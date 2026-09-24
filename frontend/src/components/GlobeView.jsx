import React, { useEffect, useRef, useState } from "react";
import {
  Color,
  GeoJsonDataSource,
  ImageryLayer,
  Ion,
  Rectangle,
  SingleTileImageryProvider,
  Terrain,
  UrlTemplateImageryProvider,
  Viewer,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { sortForDrawing } from "../layers.js";

const ION_TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN;

/** Cesium 3D globe. With a (free) Cesium ion token it drapes results over
 *  Cesium World Terrain; without one it uses the smooth ellipsoid — the
 *  MapLibre "3D terrain" view works token-free either way. */
export default function GlobeView({ aoi, layers, layerState }) {
  const container = useRef(null);
  const viewerRef = useRef(null);
  const overlaysRef = useRef({}); // layer id -> ImageryLayer | DataSource
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

  // AOI outline + camera.
  useEffect(() => {
    if (!ready) return;
    const viewer = viewerRef.current;
    const rect = Rectangle.fromDegrees(...aoi);
    viewer.entities.removeById("aoi");
    viewer.entities.add({
      id: "aoi",
      rectangle: { coordinates: rect, fill: false, outline: true, outlineColor: Color.YELLOW, outlineWidth: 2 },
    });
    viewer.camera.flyTo({ destination: rect, duration: 1.2 });
  }, [aoi, ready]);

  // Latest visibility/opacity, readable from async overlay loading without stale closures.
  const stateRef = useRef(layerState);
  stateRef.current = layerState;

  const applyState = (layer, overlay) => {
    const st = stateRef.current[layer.id] ?? { visible: false, opacity: 0.85 };
    overlay.show = st.visible;
    if (overlay instanceof ImageryLayer) overlay.alpha = st.opacity;
  };

  // Result overlays (drawn bottom→top in the order the map view uses).
  useEffect(() => {
    if (!ready) return;
    const viewer = viewerRef.current;
    let cancelled = false;
    for (const overlay of Object.values(overlaysRef.current)) {
      if (overlay instanceof ImageryLayer) viewer.imageryLayers.remove(overlay, true);
      else viewer.dataSources.remove(overlay, true);
    }
    overlaysRef.current = {};

    (async () => {
      // Decode every overlay in parallel, then add them in draw order.
      const ordered = sortForDrawing(layers).filter((l) => l.image || l.geojson);
      const loaded = await Promise.all(
        ordered.map((layer) =>
          layer.image
            ? SingleTileImageryProvider.fromUrl(layer.image, { rectangle: Rectangle.fromDegrees(...layer.bbox) })
            : GeoJsonDataSource.load(layer.geojson, { stroke: Color.CYAN, strokeWidth: 3, clampToGround: true }),
        ),
      );
      if (cancelled) return;
      ordered.forEach((layer, i) => {
        let overlay = loaded[i];
        if (layer.image) overlay = viewer.imageryLayers.addImageryProvider(overlay);
        else viewer.dataSources.add(overlay);
        applyState(layer, overlay);
        overlaysRef.current[layer.id] = overlay;
      });
    })().catch((e) => console.error("globe overlay failed", e));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, ready]);

  useEffect(() => {
    for (const layer of layers) {
      const overlay = overlaysRef.current[layer.id];
      if (overlay) applyState(layer, overlay);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, layerState]);

  return (
    <div className="map-wrap">
      <div ref={container} className="map" />
      {!ION_TOKEN && <div className="globe-note">Set VITE_CESIUM_ION_TOKEN for 3D terrain on the globe</div>}
    </div>
  );
}
