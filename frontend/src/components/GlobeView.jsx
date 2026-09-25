import React, { useEffect, useRef, useState } from "react";
import {
  Cartographic,
  Color,
  GeoJsonDataSource,
  PolygonHierarchy,
  Rectangle,
  Resource,
  UrlTemplateImageryProvider,
  Cartesian3,
  sampleTerrainMostDetailed,
} from "cesium";
import { absolute, api, authHeader, withShare } from "../api.js";
import { sortForDrawing } from "../layers.js";
import { GlobeControls, GlobeHud, useGlobeHud } from "../globe/GlobeChrome.jsx";
import { Overlay, clientConfig, createViewer, flyToBbox, google3dAvailable, setBasemap } from "../globe/globeCore.js";

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/** Flat water surface for a flood layer, height calibrated against the terrain along
 *  the flood boundary (where terrain height == water level by definition). This removes
 *  the geoid/ellipsoid offset between the Copernicus DEM and Cesium World Terrain. */
async function addWaterSurface(viewer, layer, share) {
  const fc = await api(layer.geojson_export, { share });
  const polys = fc.features
    .filter((f) => f.geometry.type === "Polygon")
    .sort((a, b) => b.properties.area_km2 - a.properties.area_km2)
    .slice(0, 40);
  if (!polys.length) return [];
  const ring = polys.flatMap((f) => f.geometry.coordinates[0]);
  const step = Math.max(1, Math.floor(ring.length / 60));
  const samples = ring.filter((_, i) => i % step === 0).map(([lon, lat]) => Cartographic.fromDegrees(lon, lat));
  const sampled = await sampleTerrainMostDetailed(viewer.terrainProvider, samples);
  const height = median(sampled.map((c) => c.height).filter(Number.isFinite));
  return polys.map((f) =>
    viewer.entities.add({
      polygon: {
        hierarchy: new PolygonHierarchy(Cartesian3.fromDegreesArray(f.geometry.coordinates[0].flat())),
        height,
        material: Color.fromCssColorString("#1e78ff").withAlpha(0.55),
      },
    }),
  );
}

export default function GlobeView({ aoi, layers, layerState, share }) {
  const container = useRef(null);
  const [viewer, setViewer] = useState(null);
  const [cfg, setCfg] = useState(null);
  const [basemap, setBasemapState] = useState(null);
  const overlaysRef = useRef({}); // layer id -> Overlay | DataSource | Entity[]
  const stateRef = useRef(layerState);
  stateRef.current = layerState;
  const hud = useGlobeHud(viewer);

  useEffect(() => {
    let v;
    let cancelled = false;
    clientConfig().then(async (c) => {
      if (cancelled) return;
      setCfg(c);
      v = createViewer(container.current, c);
      setViewer(v);
      setBasemapState(await setBasemap(v, "hybrid"));
    });
    return () => {
      cancelled = true;
      if (v && !v.isDestroyed()) v.destroy();
      overlaysRef.current = {};
    };
  }, []);

  useEffect(() => {
    if (!viewer) return;
    const rect = Rectangle.fromDegrees(...aoi);
    viewer.entities.removeById("aoi");
    viewer.entities.add({
      id: "aoi",
      polyline: {
        positions: Cartesian3.fromDegreesArray([aoi[0], aoi[1], aoi[2], aoi[1], aoi[2], aoi[3], aoi[0], aoi[3], aoi[0], aoi[1]]),
        width: 2.5, material: Color.YELLOW, clampToGround: true,
      },
    });
    if (rect) flyToBbox(viewer, aoi, 1.6);
  }, [aoi, viewer]);

  const applyState = (layer, overlay) => {
    const st = stateRef.current[layer.id] ?? { visible: false, opacity: 0.85 };
    if (Array.isArray(overlay)) overlay.forEach((e) => (e.show = st.visible));
    else if (overlay instanceof Overlay) overlay.set(st.visible, st.opacity);
    else overlay.show = st.visible;
  };

  useEffect(() => {
    if (!viewer) return undefined;
    let cancelled = false;
    for (const overlay of Object.values(overlaysRef.current)) {
      if (overlay instanceof Overlay) overlay.remove();
      else if (Array.isArray(overlay)) overlay.forEach((e) => viewer.entities.remove(e));
      else viewer.dataSources.remove(overlay, true);
    }
    overlaysRef.current = {};

    (async () => {
      for (const layer of sortForDrawing(layers)) {
        let overlay;
        if (layer.tiles) {
          overlay = new Overlay(viewer, new UrlTemplateImageryProvider({
            url: new Resource({ url: absolute(withShare(layer.tiles, share)), headers: authHeader() }),
            rectangle: Rectangle.fromDegrees(...layer.bbox),
            maximumLevel: 18,
          }));
        } else if (layer.geojson_url) {
          const resource = new Resource({ url: absolute(withShare(layer.geojson_url, share)), headers: authHeader() });
          overlay = await GeoJsonDataSource.load(resource, { stroke: Color.CYAN, strokeWidth: 3, clampToGround: true });
          if (cancelled) return;
          viewer.dataSources.add(overlay);
        } else continue;
        applyState(layer, overlay);
        overlaysRef.current[layer.id] = overlay;
      }
      if (!cfg?.cesium_ion_token) return;
      for (const layer of layers.filter((l) => l.meta?.water_level_m_asl != null)) {
        try {
          const surface = await addWaterSurface(viewer, { ...layer, geojson_export: layer.downloads.geojson }, share);
          if (cancelled) return;
          applyState(layer, surface);
          overlaysRef.current[`${layer.id}__water`] = surface;
        } catch (e) {
          console.warn("water surface unavailable", e);
        }
      }
    })().catch((e) => console.error("globe overlay failed", e));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, viewer, share]);

  useEffect(() => {
    for (const layer of layers) {
      for (const key of [layer.id, `${layer.id}__water`]) {
        const overlay = overlaysRef.current[key];
        if (overlay) applyState(layer, overlay);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, layerState]);

  return (
    <div className="map-wrap">
      <div ref={container} className="map" />
      <GlobeControls viewer={viewer} cfg={cfg} basemap={basemap} onBasemap={async (id) => setBasemapState(await setBasemap(viewer, id))} camera={hud.camera} />
      <GlobeHud hud={hud} />
      {cfg && !google3dAvailable(cfg) && (
        <div className="globe-note">Add GOOGLE_MAPS_API_KEY (Google 3D) or CESIUM_ION_TOKEN (terrain, flood water) to .env</div>
      )}
    </div>
  );
}
