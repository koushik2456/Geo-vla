import React, { useState } from "react";
import { ChevronDown, ChevronRight, Layers } from "lucide-react";
import { sortForDrawing } from "../layers.js";

function Legend({ legend }) {
  if (!legend) return null;
  if (legend.type === "gradient") {
    return (
      <div className="legend">
        <span className="mono">{legend.min}</span>
        <span className="ramp" style={{ background: `linear-gradient(to right, ${legend.colors.join(",")})` }} />
        <span className="mono">
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
    <div className={`layer-panel glass ${open ? "" : "collapsed"}`}>
      <button className="layer-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Collapse layers" : "Expand layers"} title="Layers">
        <Layers size={14} strokeWidth={1.75} aria-hidden="true" />
        <span>Layers ({layers.length})</span>
        {open ? <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" /> : <ChevronRight size={14} strokeWidth={1.75} aria-hidden="true" />}
      </button>
      {open && (
        <ul className="layer-tree">
          {sortForDrawing(layers).reverse().map((layer) => {
            const st = layerState[layer.id] ?? { visible: false, opacity: 0.85 };
            return (
              <li key={layer.id} className={st.visible ? "visible" : ""}>
                <label className="layer-row">
                  <input type="checkbox" checked={st.visible} onChange={(e) => update(layer.id, { visible: e.target.checked })} />
                  <span className="layer-name" title={layer.id}>{layer.name}</span>
                  {layer.stats && <em className="mono">{layer.stats.area_km2} km²</em>}
                </label>
                {st.visible && (
                  <div className="layer-controls">
                    <input type="range" min="0" max="1" step="0.05" value={st.opacity} aria-label={`${layer.name} opacity`}
                      onChange={(e) => update(layer.id, { opacity: Number(e.target.value) })} />
                    <Legend legend={layer.legend} />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
