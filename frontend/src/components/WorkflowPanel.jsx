import React, { useEffect, useMemo, useState } from "react";
import { Building2, Trees, Waves, Wheat } from "lucide-react";

const SECTOR_ICONS = {
  disaster: Waves,
  forest: Trees,
  urban: Building2,
  agriculture: Wheat,
};

/** Mirror of services/workflows.resolve_date so date pickers show real dates. */
export function resolveDate(value, today = new Date()) {
  const v = String(value).trim().toLowerCase();
  const iso = (d) => d.toISOString().slice(0, 10);
  if (v === "today") return iso(today);
  const m = v.match(/^-(\d+)([dmy])$/);
  if (!m) return v.slice(0, 10);
  const n = Number(m[1]);
  const d = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));
  if (m[2] === "d") d.setUTCDate(d.getUTCDate() - n);
  else {
    d.setUTCDate(Math.min(d.getUTCDate(), 28));
    d.setUTCMonth(d.getUTCMonth() - (m[2] === "m" ? n : 12 * n));
  }
  return iso(d);
}

export function defaultParams(wf, { relative = false } = {}) {
  return Object.fromEntries(
    wf.params.map((p) => [p.name, p.type === "date" && !relative ? resolveDate(p.default) : p.type === "numbers" ? p.default.join(", ") : p.default]),
  );
}

export function ParamFields({ wf, values, onChange, relativeDates = false }) {
  return wf.params.map((p) => {
    const set = (v) => onChange({ ...values, [p.name]: v });
    const id = `param-${wf.id}-${p.name}`;
    let input;
    if (p.type === "boolean") {
      input = <input id={id} type="checkbox" checked={!!values[p.name]} onChange={(e) => set(e.target.checked)} />;
    } else if (p.type === "date") {
      input = relativeDates ? (
        <input id={id} type="text" value={values[p.name]} onChange={(e) => set(e.target.value)} placeholder="-1y, -30d or 2024-06-01" />
      ) : (
        <input id={id} type="date" value={values[p.name]} onChange={(e) => set(e.target.value)} max={resolveDate("today")} />
      );
    } else if (p.type === "numbers") {
      input = <input id={id} type="text" value={values[p.name]} onChange={(e) => set(e.target.value)} />;
    } else {
      input = <input id={id} type="number" value={values[p.name]} min={p.min} max={p.max} step={p.type === "integer" ? 1 : "any"}
        onChange={(e) => set(e.target.value === "" ? "" : Number(e.target.value))} />;
    }
    return (
      <div className={`field ${p.type === "boolean" ? "field-check" : ""}`} key={p.name}>
        <label htmlFor={id}>{p.label}</label>
        {input}
        {p.help && <small>{p.help}</small>}
      </div>
    );
  });
}

export default function WorkflowPanel({ catalog, busy, onRun }) {
  const [sector, setSector] = useState("disaster");
  const [selected, setSelected] = useState(null);
  const [values, setValues] = useState({});
  const sectorSpec = catalog?.sectors.find((s) => s.id === sector);
  const wf = useMemo(() => sectorSpec?.workflows.find((w) => w.id === selected), [sectorSpec, selected]);

  useEffect(() => {
    if (sectorSpec && !sectorSpec.workflows.some((w) => w.id === selected)) setSelected(sectorSpec.workflows[0].id);
  }, [sectorSpec, selected]);
  useEffect(() => {
    if (wf) setValues(defaultParams(wf));
  }, [wf]);

  if (!catalog) return <p className="muted">Loading workflows…</p>;
  const submit = (e) => {
    e.preventDefault();
    const params = { ...values };
    for (const p of wf.params) if (p.type === "numbers") params[p.name] = String(params[p.name]).split(",").map(Number).filter((n) => !Number.isNaN(n));
    onRun({ workflow: wf.id, params });
  };

  return (
    <div className="workflows">
      <div className="segmented sector-tabs" role="tablist">
        {catalog.sectors.map((s) => {
          const Icon = SECTOR_ICONS[s.id];
          return (
            <button key={s.id} role="tab" aria-selected={s.id === sector} className={s.id === sector ? "active" : ""}
              onClick={() => setSector(s.id)} title={s.title}>
              {Icon && <Icon size={14} strokeWidth={1.75} aria-hidden="true" />}
              <span>{s.title}</span>
            </button>
          );
        })}
      </div>
      <div className="wf-cards">
        {sectorSpec.workflows.map((w) => (
          <button key={w.id} type="button" className={`wf-card ${w.id === selected ? "selected" : ""}`} onClick={() => setSelected(w.id)}>
            <strong>{w.title}</strong>
            <span>{w.summary}</span>
          </button>
        ))}
      </div>
      {wf && (
        <form className="wf-form" onSubmit={submit}>
          <ParamFields wf={wf} values={values} onChange={setValues} />
          {wf.outputs?.length > 0 && (
            <p className="hint wf-outputs">Outputs: {wf.outputs.join(", ")}</p>
          )}
          <button type="submit" className="primary" disabled={busy}>
            {busy ? "Running…" : "Run"}
          </button>
        </form>
      )}
    </div>
  );
}
