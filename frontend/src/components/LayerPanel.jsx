import React, { useState } from "react";
import { sortForDrawing } from "../layers.js";

function Legend({ legend }) {
  if (!legend) return null;
  if (legend.type === "gradient") {
    return (
      <div className="legend">
        <span>{legend.min}</span>
        <span className="ramp" style={{ background: `linear-gradient(to right, ${legend.colors.join(",")})` }} />
        <span>
          {legend.max} {legend.unit}
        </span>
      </div>
    );
  }
  if (legend.type === "mask") return <span className="swatch" style={{ background: legend.color }} />;
  if (legend.type === "categorical") {
    return (
      <div className="legend classes">
        {Object.entries(legend.classes).map(([name, color]) => (
          <span key={name}>
            <span className="swatch" style={{ background: color }} /> {name}
          </span>
        ))}
      </div>
    );
  }
  return null;
}

export default function LayerPanel({ layers, layerState, onChange }) {
  const update = (id, patch) => onChange((s) => ({ ...s, [id]: { ...s[id], ...patch } }));
  const [open, setOpen] = useState(() => window.innerWidth > 900);

  return (
    <div className={`layer-panel ${open ? "" : "collapsed"}`}>
      <button className="layer-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        Layers ({layers.length}) {open ? "▾" : "▸"}
      </button>
      {open && (
        <ul>
          {sortForDrawing(layers).reverse().map((layer) => {
            const st = layerState[layer.id] ?? { visible: false, opacity: 0.85 };
            return (
              <li key={layer.id}>
                <label>
                  <input type="checkbox" checked={st.visible} onChange={(e) => update(layer.id, { visible: e.target.checked })} />
                  <span title={layer.id}>{layer.name}</span>
                  {layer.stats && <em>{layer.stats.area_km2} km²</em>}
                </label>
                {st.visible && (
                  <>
                    <input type="range" min="0" max="1" step="0.05" value={st.opacity} aria-label={`${layer.name} opacity`}
                      onChange={(e) => update(layer.id, { opacity: Number(e.target.value) })} />
                    <Legend legend={layer.legend} />
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
