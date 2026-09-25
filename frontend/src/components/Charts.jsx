import React, { useMemo, useState } from "react";
import { fmt } from "../layers.js";

// Categorical slots 1-3 of the validated palette (see tokens.css --series-*); text stays in ink tokens.
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const W = 400;
const H = 210;
const PAD = { top: 12, right: 12, bottom: 44, left: 46 };

function niceTicks(min, max, count = 4) {
  if (min === max) max = min + 1;
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(Number(v.toPrecision(10)));
  return ticks;
}

function flatBar(x, y, w, h) {
  if (h <= 0) return "";
  return `M${x},${y + h}V${y}H${x + w}V${y + h}Z`;
}

function Legend({ series, colors = SERIES }) {
  if (series.length < 2) return null;
  return (
    <div className="chart-legend">
      {series.map((s, i) => (
        <span key={s.name}>
          <i style={{ background: colors[i] }} /> {s.name}
        </span>
      ))}
    </div>
  );
}

function Tooltip({ x, y, title, rows, unit }) {
  const left = x > W * 0.6;
  return (
    <div className="chart-tip" style={{ left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%`, transform: `translate(${left ? "calc(-100% - 10px)" : "10px"}, -50%)` }}>
      <strong>{title}</strong>
      {rows.map((r) => (
        <div key={r.name}>
          <i style={{ background: r.color }} /> {r.name}: <b className="mono">{fmt(r.value, 3)}</b> {unit}
        </div>
      ))}
    </div>
  );
}

function BarChart({ chart }) {
  const SERIES_C = chart.colors || SERIES;
  const [hover, setHover] = useState(null);
  const series = chart.series.slice(0, 3);
  const cats = chart.categories;
  const values = series.flatMap((s) => s.values);
  const ticks = niceTicks(0, Math.max(...values, 0));
  const top = ticks.at(-1);
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const band = plotW / cats.length;
  const barW = Math.max(2, Math.min(28, (band * 0.72) / series.length - 2));
  const y = (v) => PAD.top + plotH - (v / top) * plotH;

  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chart.title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t) + 3} className="axis" textAnchor="end">
              {fmt(t)}
            </text>
          </g>
        ))}
        {cats.map((c, ci) => {
          const groupW = series.length * (barW + 2) - 2;
          const x0 = PAD.left + ci * band + (band - groupW) / 2;
          return (
            <g key={c} onMouseEnter={() => setHover(ci)} onMouseLeave={() => setHover(null)}>
              <rect x={PAD.left + ci * band} y={PAD.top} width={band} height={plotH} fill="transparent" />
              {series.map((s, si) => (
                <path key={s.name} d={flatBar(x0 + si * (barW + 2), y(s.values[ci]), barW, y(0) - y(s.values[ci]))}
                  fill={SERIES_C[si]} opacity={hover === null || hover === ci ? 1 : 0.45} />
              ))}
              <text x={PAD.left + ci * band + band / 2} y={H - PAD.bottom + 14} className="axis" textAnchor={cats.length > 4 ? "end" : "middle"}
                transform={cats.length > 4 ? `rotate(-30 ${PAD.left + ci * band + band / 2} ${H - PAD.bottom + 14})` : undefined}>
                {c.length > 14 ? c.slice(0, 13) + "…" : c}
              </text>
            </g>
          );
        })}
        <line x1={PAD.left} x2={W - PAD.right} y1={y(0)} y2={y(0)} className="baseline" />
        <text x={12} y={PAD.top + plotH / 2} className="axis" transform={`rotate(-90 12 ${PAD.top + plotH / 2})`} textAnchor="middle">
          {chart.unit}
        </text>
      </svg>
      {hover !== null && (
        <Tooltip x={PAD.left + hover * band + band / 2} y={PAD.top + plotH / 3} title={cats[hover]} unit={chart.unit}
          rows={series.map((s, i) => ({ name: s.name, value: s.values[hover], color: SERIES_C[i] }))} />
      )}
    </div>
  );
}

function LineChart({ chart }) {
  const SERIES_C = chart.colors || SERIES;
  const [hover, setHover] = useState(null);
  const series = chart.series.slice(0, 3);
  const xs = chart.x;
  const values = series.flatMap((s) => s.values).filter((v) => v != null);
  const ticks = niceTicks(Math.min(...values), Math.max(...values));
  const [lo, hi] = [ticks[0], ticks.at(-1)];
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (xs.length === 1 ? plotW / 2 : (i / (xs.length - 1)) * plotW);
  const y = (v) => PAD.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;
  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((px - PAD.left) / plotW) * (xs.length - 1));
    setHover(Math.max(0, Math.min(xs.length - 1, i)));
  };

  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chart.title} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t) + 3} className="axis" textAnchor="end">
              {fmt(t, 3)}
            </text>
          </g>
        ))}
        {xs.map((label, i) =>
          xs.length <= 8 || i % Math.ceil(xs.length / 8) === 0 ? (
            <text key={label + i} x={x(i)} y={H - PAD.bottom + 16} className="axis" textAnchor="middle">
              {label}
            </text>
          ) : null,
        )}
        {chart.x_label && (
          <text x={PAD.left + plotW / 2} y={H - 8} className="axis" textAnchor="middle">
            {chart.x_label}
          </text>
        )}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} className="crosshair" />}
        {series.map((s, si) => {
          const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
          const area = `${x(0)},${y(lo)} ${pts} ${x(xs.length - 1)},${y(lo)}`;
          return (
            <g key={s.name}>
              <polygon fill={SERIES_C[si]} fillOpacity="0.15" stroke="none" points={area} />
              <polyline fill="none" stroke={SERIES_C[si]} strokeWidth="1.5" strokeLinejoin="round" points={pts} />
              {s.values.map((v, i) => (xs.length <= 24 || hover === i) && (
                <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 3.5 : 2.5} fill={SERIES_C[si]} stroke="var(--surface-1)" strokeWidth="1.5" />
              ))}
            </g>
          );
        })}
        <text x={12} y={PAD.top + plotH / 2} className="axis" transform={`rotate(-90 12 ${PAD.top + plotH / 2})`} textAnchor="middle">
          {chart.unit}
        </text>
      </svg>
      {hover !== null && (
        <Tooltip x={x(hover)} y={PAD.top + plotH / 3} title={xs[hover]} unit={chart.unit}
          rows={series.map((s, i) => ({ name: s.name, value: s.values[hover], color: SERIES_C[i] }))} />
      )}
    </div>
  );
}

export function DataTable({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className={typeof c === "number" ? "num" : undefined}>{fmt(c, 3)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function asTable(chart) {
  if (chart.type === "table") return { columns: chart.columns, rows: chart.rows };
  const labels = chart.type === "bar" ? chart.categories : chart.x;
  return {
    columns: [chart.type === "bar" ? "Category" : chart.x_label || "Date", ...chart.series.map((s) => `${s.name}${chart.unit ? ` (${chart.unit})` : ""}`)],
    rows: labels.map((l, i) => [l, ...chart.series.map((s) => s.values[i])]),
  };
}

export function Chart({ chart }) {
  const [table, setTable] = useState(chart.type === "table");
  const data = useMemo(() => asTable(chart), [chart]);
  const plottable = chart.type !== "table" && chart.series?.length;
  return (
    <figure className="chart">
      <figcaption>
        <span>{chart.title}</span>
        {plottable && (
          <button className="link" onClick={() => setTable((t) => !t)}>
            {table ? "Chart" : "Table"}
          </button>
        )}
      </figcaption>
      {plottable && <Legend series={chart.series.slice(0, 3)} colors={chart.colors} />}
      {table || !plottable ? <DataTable {...data} /> : chart.type === "bar" ? <BarChart chart={chart} /> : <LineChart chart={chart} />}
    </figure>
  );
}

export function KeyFigures({ figures }) {
  if (!figures?.length) return null;
  return (
    <div className="figures">
      {figures.map((f) => (
        <div className="figure" key={f.label}>
          <div className="figure-label">{f.label}</div>
          <div className="figure-value mono">
            {fmt(f.value)}
            {f.unit && <span className="figure-unit">{f.unit}</span>}
          </div>
          {f.note && <div className="figure-note">{f.note}</div>}
        </div>
      ))}
    </div>
  );
}
