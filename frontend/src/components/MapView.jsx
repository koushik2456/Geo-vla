import React, { useEffect, useRef, useState } from "react";
import { BASE_STYLE, addResultLayer, bboxPolygon, maplibregl, removeResultLayers, transformRequest } from "../mapStyle.js";
import { sortForDrawing } from "../layers.js";

export default function MapView({ aoi, onAoiChange, layers, layerState, terrain, share, fitKey, readOnly }) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current,
      style: BASE_STYLE,
      bounds: [[aoi[0], aoi[1]], [aoi[2], aoi[3]]],
      fitBoundsOptions: { padding: 60 },
      maxPitch: 80,
      transformRequest,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
    map.addControl(new maplibregl.ScaleControl(), "bottom-left");
    // "style.load" rather than "load": the latter waits for a fully rendered frame, which can
    // stall when basemap tiles are unreachable (e.g. offline deployments).
    map.once("style.load", () => {
      map.addSource("aoi", { type: "geojson", data: bboxPolygon(aoi) });
      map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#ffd400", "line-width": 2, "line-dasharray": [2, 1] } });
      setReady(true);
    });
    mapRef.current = map;
    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (ready) mapRef.current.getSource("aoi").setData(bboxPolygon(aoi));
  }, [aoi, ready]);

  // Fly to the AOI when it changes from outside the map (place search, loading a run).
  useEffect(() => {
    if (ready) mapRef.current.fitBounds([[aoi[0], aoi[1]], [aoi[2], aoi[3]]], { padding: 60, duration: 900 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, ready]);

  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    if (terrain) {
      map.setTerrain({ source: "terrain", exaggeration: 1.5 });
      map.setLayoutProperty("hillshade", "visibility", "visible");
      map.easeTo({ pitch: 62, duration: 800 });
    } else {
      map.setTerrain(null);
      map.setLayoutProperty("hillshade", "visibility", "none");
      map.easeTo({ pitch: 0, bearing: 0, duration: 800 });
    }
  }, [terrain, ready]);

  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    removeResultLayers(map);
    for (const layer of sortForDrawing(layers)) addResultLayer(map, layer, share, "aoi-line");
  }, [layers, ready, share]);

  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    for (const layer of layers) {
      const id = `res-${layer.id}`;
      if (!map.getLayer(id)) continue;
      const st = layerState[layer.id] ?? { visible: false, opacity: 0.85 };
      map.setLayoutProperty(id, "visibility", st.visible ? "visible" : "none");
      map.setPaintProperty(id, layer.tiles ? "raster-opacity" : "line-opacity", st.opacity);
    }
  }, [layers, layerState, ready]);

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
      {!readOnly && (
        <div className="aoi-tools">
          <button className={drawing ? "active" : ""} onClick={() => setDrawing((d) => !d)}>
            {drawing ? "Drag a box…" : "Draw area"}
          </button>
          <button onClick={aoiFromView}>Use view</button>
        </div>
      )}
    </div>
  );
}
