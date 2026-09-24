// Draw order: imagery at the bottom, continuous rasters, then masks, then vectors on top.
const RANK = { scene: 0, dem: 1, scalar: 2, classmap: 3, mask: 4, vector: 5 };

export const sortForDrawing = (layers) =>
  layers.map((l, i) => [l, i]).sort(([a, i], [b, j]) => RANK[a.kind] - RANK[b.kind] || i - j).map(([l]) => l);

/** Initially show the latest scene plus the final mask (the answer's zone), or the last layer. */
export function defaultLayerState(layers) {
  const finalLayer = layers.findLast((l) => l.kind === "mask") ?? layers.at(-1);
  const shown = new Set([layers.findLast((l) => l.kind === "scene")?.id, finalLayer?.id]);
  return Object.fromEntries(layers.map((l) => [l.id, { visible: shown.has(l.id), opacity: 0.85 }]));
}
