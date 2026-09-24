// Draw order: imagery at the bottom, continuous rasters, then masks, then vectors on top.
const RANK = { scene: 0, dem: 1, scalar: 2, classmap: 3, mask: 4, vector: 5 };

export const sortForDrawing = (layers) =>
  layers.map((l, i) => [l, i]).sort(([a, i], [b, j]) => RANK[a.kind] - RANK[b.kind] || i - j).map(([l]) => l);

/** Initially show the latest scene plus the result layer the analysis focused on. */
export function defaultLayerState(layers, focus) {
  const finalLayer = layers.find((l) => l.id === focus) ?? layers.findLast((l) => l.kind === "mask") ?? layers.at(-1);
  const shown = new Set([layers.findLast((l) => l.kind === "scene")?.id, finalLayer?.id]);
  return Object.fromEntries(layers.map((l) => [l.id, { visible: shown.has(l.id), opacity: 0.85 }]));
}

/** Timelapse frames: layers produced by a time series, grouped by kind, ordered by date. */
export function timelapseGroups(layers) {
  const groups = {};
  for (const l of layers) {
    if (l.meta?.group === "timeseries" && l.meta?.date) (groups[l.kind] ??= []).push(l);
  }
  for (const g of Object.values(groups)) g.sort((a, b) => a.meta.date.localeCompare(b.meta.date));
  return Object.fromEntries(Object.entries(groups).filter(([, g]) => g.length > 1));
}

/** Dated scene layers — candidates for the before/after swipe. */
export const datedScenes = (layers) =>
  layers.filter((l) => l.kind === "scene" && l.meta?.date).sort((a, b) => a.meta.date.localeCompare(b.meta.date));

export const fmt = (v, digits = 2) =>
  typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : Number(v.toFixed(digits)).toString()) : String(v ?? "—");
