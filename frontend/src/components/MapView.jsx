import React, { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import { sortForDrawing } from "../layers.js";

// Bundling breaks MapLibre's import.meta.url-relative worker lookup; vite.config.js copies it here.
maplibregl.setWorkerUrl(`${import.meta.env.BASE_URL}maplibre/maplibre-gl-worker.mjs`);

const ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

const STYLE = {
  version: 8,
  sources: {
    imagery: {
      type: "raster",
      tiles: [ESRI_IMAGERY],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    },
    // Free, token-less global elevation tiles (AWS Terrain Tiles) for 3D terrain + hillshade.
    terrain: { type: "raster-dem", tiles: [TERRARIUM], encoding: "terrarium", tileSize: 256, maxzoom: 14 },
    hillshade: { type: "raster-dem", tiles: [TERRARIUM], encoding: "terrarium", tileSize: 256, maxzoom: 14 },
  },
  layers: [
    { id: "imagery", type: "raster", source: "imagery" },
    { id: "hillshade", type: "hillshade", source: "hillshade", layout: { visibility: "none" }, paint: { "hillshade-exaggeration": 0.4 } },
  ],
};

const bboxPolygon = ([w, s, e, n]) => ({
  type: "Feature",
  geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
  properties: {},
});

const corners = ([w, s, e, n]) => [[w, n], [e, n], [e, s], [w, s]];

export default function MapView({ aoi, onAoiChange, layers, layerState, terrain }) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [drawing, setDrawing] = useState(false);

  // Create the map once.
  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current,
      style: STYLE,
      bounds: [[aoi[0], aoi[1]], [aoi[2], aoi[3]]],
      fitBoundsOptions: { padding: 80 },
      maxPitch: 80,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
    map.addControl(new maplibregl.ScaleControl(), "bottom-left");
    map.on("load", () => {
      map.addSource("aoi", { type: "geojson", data: bboxPolygon(aoi) });
      map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#ffd400", "line-width": 2, "line-dasharray": [2, 1] } });
      setReady(true);
    });
    mapRef.current = map;
    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the AOI outline in sync.
  useEffect(() => {
    if (ready) mapRef.current.getSource("aoi").setData(bboxPolygon(aoi));
  }, [aoi, ready]);

  // 3D terrain toggle.
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    if (terrain) {
      map.setTerrain({ source: "terrain", exaggeration: 1.5 });
      map.setLayoutProperty("hillshade", "visibility", "visible");
      map.easeTo({ pitch: 60, duration: 800 });
    } else {
      map.setTerrain(null);
      map.setLayoutProperty("hillshade", "visibility", "none");
      map.easeTo({ pitch: 0, bearing: 0, duration: 800 });
    }
  }, [terrain, ready]);

  // Replace result overlays whenever a new result arrives.
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    for (const l of map.getStyle().layers) if (l.id.startsWith("res-")) map.removeLayer(l.id);
    for (const id of Object.keys(map.getStyle().sources)) if (id.startsWith("res-")) map.removeSource(id);

    for (const layer of sortForDrawing(layers)) {
      const id = `res-${layer.id}`;
      if (layer.image) {
        map.addSource(id, { type: "image", url: layer.image, coordinates: corners(layer.bbox) });
        map.addLayer({ id, type: "raster", source: id, paint: { "raster-fade-duration": 0, "raster-resampling": "nearest" } }, "aoi-line");
      } else if (layer.geojson) {
        map.addSource(id, { type: "geojson", data: layer.geojson });
        map.addLayer({ id, type: "line", source: id, paint: { "line-color": "#00e5ff", "line-width": 2.5 } }, "aoi-line");
      }
    }
    if (layers.length) {
      const [w, s, e, n] = layers[0].bbox;
      map.fitBounds([[w, s], [e, n]], { padding: 80, duration: 800 });
    }
  }, [layers, ready]);

  // Visibility and opacity.
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    for (const layer of layers) {
      const id = `res-${layer.id}`;
      if (!map.getLayer(id)) continue;
      const st = layerState[layer.id] ?? { visible: false, opacity: 0.85 };
      map.setLayoutProperty(id, "visibility", st.visible ? "visible" : "none");
      map.setPaintProperty(id, layer.image ? "raster-opacity" : "line-opacity", st.opacity);
    }
  }, [layers, layerState, ready]);

  // Rectangle drawing for the area of interest.
  useEffect(() => {
    if (!ready || !drawing) return;
    const map = mapRef.current;
    let start = null;
    map.dragPan.disable();
    map.getCanvas().style.cursor = "crosshair";

    const toBbox = (a, b) => [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)];
    const down = (e) => {
      start = e.lngLat;
    };
    const move = (e) => {
      if (start) map.getSource("aoi").setData(bboxPolygon(toBbox(start, e.lngLat)));
    };
    const up = (e) => {
      if (!start) return;
      const bbox = toBbox(start, e.lngLat);
      start = null;
      if (bbox[2] - bbox[0] > 1e-4 && bbox[3] - bbox[1] > 1e-4) onAoiChange(bbox.map((v) => Number(v.toFixed(5))));
      setDrawing(false);
    };
    map.on("mousedown", down);
    map.on("mousemove", move);
    map.on("mouseup", up);
    return () => {
      map.off("mousedown", down);
      map.off("mousemove", move);
      map.off("mouseup", up);
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
    };
  }, [drawing, ready, onAoiChange]);

  const aoiFromView = () => {
    const b = mapRef.current.getBounds();
    onAoiChange([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => Number(v.toFixed(5))));
  };

  return (
    <div className="map-wrap">
      <div ref={container} className="map" />
      <div className="aoi-tools">
        <button className={drawing ? "active" : ""} onClick={() => setDrawing((d) => !d)}>
          {drawing ? "Drag a box…" : "Draw AOI"}
        </button>
        <button onClick={aoiFromView}>Use view as AOI</button>
      </div>
    </div>
  );
}
