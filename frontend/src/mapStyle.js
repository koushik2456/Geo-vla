import * as maplibregl from "maplibre-gl";
import { absolute, authHeader, isApiUrl, withShare } from "./api.js";

// Bundling breaks MapLibre's import.meta.url-relative worker lookup; vite.config.js copies it here.
maplibregl.setWorkerUrl(`${import.meta.env.BASE_URL}maplibre/maplibre-gl-worker.mjs`);

export { maplibregl };

const ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

export const BASE_STYLE = {
  version: 8,
  sources: {
    imagery: { type: "raster", tiles: [ESRI_IMAGERY], tileSize: 256, maxzoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics" },
    // Free, token-less global elevation tiles (AWS Terrain Tiles) for 3D terrain + hillshade.
    terrain: { type: "raster-dem", tiles: [TERRARIUM], encoding: "terrarium", tileSize: 256, maxzoom: 14 },
    hillshade: { type: "raster-dem", tiles: [TERRARIUM], encoding: "terrarium", tileSize: 256, maxzoom: 14 },
  },
  layers: [
    { id: "imagery", type: "raster", source: "imagery" },
    { id: "hillshade", type: "hillshade", source: "hillshade", layout: { visibility: "none" }, paint: { "hillshade-exaggeration": 0.4 } },
  ],
};

/** Attach the signed-in user's token to requests for our own tiles and GeoJSON. */
export const transformRequest = (url) => (isApiUrl(url) ? { url, headers: authHeader() } : { url });

export const bboxPolygon = ([w, s, e, n]) => ({
  type: "Feature",
  geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
  properties: {},
});

/** Add one result layer (full-resolution tiles or GeoJSON) to a map, below `beforeId`. */
export function addResultLayer(map, layer, share, beforeId) {
  const id = `res-${layer.id}`;
  if (layer.tiles) {
    map.addSource(id, { type: "raster", tiles: [absolute(withShare(layer.tiles, share))], tileSize: 256, bounds: layer.bbox, maxzoom: 18 });
    map.addLayer({ id, type: "raster", source: id, paint: { "raster-fade-duration": 0 } }, beforeId);
  } else if (layer.geojson_url) {
    map.addSource(id, { type: "geojson", data: absolute(withShare(layer.geojson_url, share)) });
    map.addLayer({ id, type: "line", source: id, paint: { "line-color": "#00e5ff", "line-width": 2.5 } }, beforeId);
  }
  return id;
}

export function removeResultLayers(map) {
  for (const l of map.getStyle().layers) if (l.id.startsWith("res-")) map.removeLayer(l.id);
  for (const id of Object.keys(map.getStyle().sources)) if (id.startsWith("res-")) map.removeSource(id);
}
