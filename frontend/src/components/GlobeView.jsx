import React, { useEffect, useRef, useState } from "react";
import {
  Cartographic,
  Color,
  GeoJsonDataSource,
  ImageryLayer,
  Ion,
  PolygonHierarchy,
  Rectangle,
  Resource,
  Terrain,
  UrlTemplateImageryProvider,
  Viewer,
  Cartesian3,
  sampleTerrainMostDetailed,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { absolute, api, authHeader, withShare } from "../api.js";
import { sortForDrawing } from "../layers.js";

const ION_TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN;

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
  const viewerRef = useRef(null);
  const overlaysRef = useRef({}); // layer id -> ImageryLayer | DataSource | Entity[]
  const stateRef = useRef(layerState);
  stateRef.current = layerState;
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (ION_TOKEN) Ion.defaultAccessToken = ION_TOKEN;
    const viewer = new Viewer(container.current, {
      baseLayer: new ImageryLayer(
        new UrlTemplateImageryProvider({
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          credit: "Imagery © Esri",
          maximumLevel: 19,
        }),
      ),
      terrain: ION_TOKEN ? Terrain.fromWorldTerrain() : undefined,
      baseLayerPicker: false,
      geocoder: false,
      animation: false,
      timeline: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
    });
    viewerRef.current = viewer;
    setReady(true);
    return () => {
      viewer.destroy();
      viewerRef.current = null;
      overlaysRef.current = {};
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const viewer = viewerRef.current;
    const rect = Rectangle.fromDegrees(...aoi);
    viewer.entities.removeById("aoi");
    viewer.entities.add({ id: "aoi", rectangle: { coordinates: rect, fill: false, outline: true, outlineColor: Color.YELLOW, outlineWidth: 2 } });
    viewer.camera.flyTo({ destination: rect, duration: 1.2 });
  }, [aoi, ready]);

  const applyState = (layer, overlay) => {
    const st = stateRef.current[layer.id] ?? { visible: false, opacity: 0.85 };
    if (Array.isArray(overlay)) overlay.forEach((e) => (e.show = st.visible));
    else {
      overlay.show = st.visible;
      if (overlay instanceof ImageryLayer) overlay.alpha = st.opacity;
    }
  };

  useEffect(() => {
    if (!ready) return;
    const viewer = viewerRef.current;
    let cancelled = false;
    for (const overlay of Object.values(overlaysRef.current)) {
      if (overlay instanceof ImageryLayer) viewer.imageryLayers.remove(overlay, true);
      else if (Array.isArray(overlay)) overlay.forEach((e) => viewer.entities.remove(e));
      else viewer.dataSources.remove(overlay, true);
    }
    overlaysRef.current = {};

    (async () => {
      for (const layer of sortForDrawing(layers)) {
        let overlay;
        if (layer.tiles) {
          overlay = viewer.imageryLayers.addImageryProvider(
            new UrlTemplateImageryProvider({
              url: new Resource({ url: absolute(withShare(layer.tiles, share)), headers: authHeader() }),
              rectangle: Rectangle.fromDegrees(...layer.bbox),
              maximumLevel: 18,
            }),
          );
        } else if (layer.geojson_url) {
          const resource = new Resource({ url: absolute(withShare(layer.geojson_url, share)), headers: authHeader() });
          overlay = await GeoJsonDataSource.load(resource, { stroke: Color.CYAN, strokeWidth: 3, clampToGround: true });
          if (cancelled) return;
          viewer.dataSources.add(overlay);
        } else continue;
        applyState(layer, overlay);
        overlaysRef.current[layer.id] = overlay;
      }
      if (!ION_TOKEN) return;
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
  }, [layers, ready, share]);

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
      {!ION_TOKEN && <div className="globe-note">Add a free VITE_CESIUM_ION_TOKEN for 3D terrain and flood water surfaces</div>}
    </div>
  );
}
