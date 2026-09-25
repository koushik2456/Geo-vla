import React, { useEffect, useRef, useState } from "react";
import { GripVertical } from "lucide-react";
import { BASE_STYLE, addResultLayer, maplibregl, removeResultLayers, transformRequest } from "../mapStyle.js";

/** Before/after swipe: two synchronised maps, the right one clipped by a draggable divider. */
export default function CompareView({ aoi, layers, share, left, right, onChange }) {
  const leftEl = useRef(null);
  const rightEl = useRef(null);
  const wrap = useRef(null);
  const maps = useRef({});
  const [split, setSplit] = useState(0.5);
  const [maps_, setMaps] = useState(null);

  useEffect(() => {
    const opts = { style: BASE_STYLE, bounds: [[aoi[0], aoi[1]], [aoi[2], aoi[3]]], fitBoundsOptions: { padding: 40 }, transformRequest };
    const a = new maplibregl.Map({ ...opts, container: leftEl.current });
    const b = new maplibregl.Map({ ...opts, container: rightEl.current, attributionControl: false });
    b.addControl(new maplibregl.NavigationControl(), "bottom-right");
    let syncing = false;
    const sync = (from, to) => () => {
      if (syncing) return;
      syncing = true;
      to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
      syncing = false;
    };
    a.on("move", sync(a, b));
    b.on("move", sync(b, a));
    maps.current = { a, b };
    setMaps({ a, b });
    return () => {
      a.remove();
      b.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Each map gets its layer as soon as its own style is parsed ("load" waits for a fully
  // rendered frame, which can stall while basemap tiles are unreachable).
  useEffect(() => {
    if (!maps_) return;
    const cleanups = [];
    for (const [map, id] of [[maps_.a, left], [maps_.b, right]]) {
      const apply = () => {
        removeResultLayers(map);
        const layer = layers.find((l) => l.id === id);
        if (layer) addResultLayer(map, layer, share);
      };
      if (map.isStyleLoaded()) apply();
      else {
        let done = false;
        const once = () => {
          if (done) return;
          done = true;
          apply();
        };
        map.once("style.load", once);
        map.once("idle", once);
        cleanups.push(() => {
          map.off("style.load", once);
          map.off("idle", once);
        });
      }
    }
    return () => cleanups.forEach((c) => c());
  }, [maps_, left, right, layers, share]);

  const drag = (e) => {
    const rect = wrap.current.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setSplit(Math.min(0.98, Math.max(0.02, x / rect.width)));
  };
  const startDrag = (e) => {
    e.preventDefault();
    const move = (ev) => drag(ev);
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const rasterLayers = layers.filter((l) => l.tiles);
  const picker = (value, set, label) => (
    <label className="compare-pick">
      <span>{label}</span>
      <select value={value ?? ""} onChange={(e) => set(e.target.value)}>
        {rasterLayers.map((l) => (
          <option key={l.id} value={l.id}>
            {l.name}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="map-wrap compare" ref={wrap}>
      <div ref={leftEl} className="map" />
      <div ref={rightEl} className="map" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
      <div className="swipe" style={{ left: `${split * 100}%` }} onPointerDown={startDrag} role="slider" aria-label="Before/after divider"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(split * 100)} tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setSplit((s) => Math.max(0.02, s - 0.05));
          if (e.key === "ArrowRight") setSplit((s) => Math.min(0.98, s + 0.05));
        }}>
        <span className="swipe-handle" aria-hidden="true">
          <GripVertical size={14} strokeWidth={1.75} />
        </span>
      </div>
      <div className="compare-bar glass">
        {picker(left, (v) => onChange(v, right), "Left")}
        {picker(right, (v) => onChange(left, v), "Right")}
      </div>
    </div>
  );
}
