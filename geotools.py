"""
geotools.py — Core geospatial tool functions for the Geo-VLA agent.

Each tool is registered with @tool and exposed to the reasoning LLM (agent.py)
through its JSON schema. Tools read and write named layers in a per-request
Workspace, so the LLM chains operations by passing layer ids
("s2_2023-06-01", "dem", "change_2021-06-01_2024-06-01", ...) instead of
shipping large arrays through its context window.

Tool groups:
  1. Data acquisition  — Sentinel-2 L2A, Copernicus DEM GLO-30, OpenStreetMap
  2. Derived indices   — classical raster math (NDVI, slope, flood extent)
  3. Neural inference  — the two trained models (classifier, change detector)
  4. Spatial analysis  — threshold, class mask, buffer, overlay, zonal stats
"""
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

import config
import rendering
import synthetic
from geo_utils import (
    bbox_area_km2, date_window, grid_shape, pixel_size_m, same_bbox, validate_bbox,
)
from models import EUROSAT_CLASSES

log = logging.getLogger("geo-vla.tools")

S2_RESOLUTION_M = 10.0
DEM_RESOLUTION_M = 30.0


class ToolError(Exception):
    """An error the LLM should see and can recover from (bad id, no data, ...)."""


# ---------------------------------------------------------------------------
# Workspace: named layers shared by all tool calls within one request
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    id: str
    kind: str          # scene | dem | scalar | mask | classmap | vector
    bbox: list
    data: object
    name: str
    meta: dict = field(default_factory=dict)


class Workspace:
    def __init__(self, default_bbox=None):
        self.default_bbox = validate_bbox(default_bbox) if default_bbox else None
        self.layers: dict = {}

    def add(self, layer: Layer) -> Layer:
        self.layers.pop(layer.id, None)   # re-insert so "latest" ordering stays correct
        self.layers[layer.id] = layer
        return layer

    def get(self, layer_id: str, kinds=None) -> Layer:
        if layer_id not in self.layers:
            raise ToolError(f"unknown layer '{layer_id}'. Available: {list(self.layers) or 'none yet'}")
        layer = self.layers[layer_id]
        if kinds and layer.kind not in kinds:
            raise ToolError(f"layer '{layer_id}' is a {layer.kind}; expected one of {list(kinds)}")
        return layer

    def latest(self, kind: str) -> Layer:
        for layer in reversed(list(self.layers.values())):
            if layer.kind == kind:
                return layer
        raise ToolError(f"no {kind} layer yet — fetch one first")

    def resolve(self, layer_id, kind: str) -> Layer:
        return self.get(layer_id, {kind}) if layer_id else self.latest(kind)

    def bbox(self, bbox=None) -> list:
        if bbox:
            return validate_bbox(bbox)
        if self.default_bbox:
            return self.default_bbox
        raise ToolError("no bbox given and no area of interest selected")

    def raster_shape(self, bbox) -> tuple:
        """Grid used when rasterizing vectors: the latest raster over this bbox, else 10 m."""
        for layer in reversed(list(self.layers.values())):
            if layer.kind != "vector" and same_bbox(layer.bbox, bbox):
                return _shape_of(layer)
        return grid_shape(bbox, S2_RESOLUTION_M)


def _shape_of(layer: Layer) -> tuple:
    if layer.kind == "scene":
        return layer.data["rgb"].shape[:2]
    return layer.data.shape[:2]


def _align(arr: np.ndarray, shape: tuple, categorical: bool) -> np.ndarray:
    """Resample a raster onto another grid over the same bbox."""
    if arr.shape[:2] == tuple(shape):
        return arr
    interp = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    src = arr.astype(np.uint8) if arr.dtype == bool else arr
    if src.dtype == np.int8:
        src = src.astype(np.int16)
    out = cv2.resize(src, (shape[1], shape[0]), interpolation=interp)
    return out.astype(bool) if arr.dtype == bool else out.astype(arr.dtype)


def _mask_stats(mask: np.ndarray, bbox) -> dict:
    frac = float(mask.mean())
    return {"fraction": round(frac, 4), "area_km2": round(frac * bbox_area_km2(bbox), 3)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict = {}

_BBOX_SCHEMA = {
    "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
    "description": "[min_lon, min_lat, max_lon, max_lat] in EPSG:4326. Omit to use the user's selected area.",
}


def tool(name: str, description: str, properties: dict, required=()):
    def decorator(fn):
        TOOLS[name] = {
            "fn": fn,
            "schema": {
                "name": name,
                "description": description,
                "input_schema": {"type": "object", "properties": properties, "required": list(required)},
            },
        }
        return fn
    return decorator


def tool_schemas() -> list:
    return [t["schema"] for t in TOOLS.values()]


def run_tool(ws: Workspace, name: str, tool_input: dict) -> dict:
    if name not in TOOLS:
        raise ToolError(f"unknown tool '{name}'")
    return TOOLS[name]["fn"](ws, **(tool_input or {}))


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------

COPERNICUS_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
COPERNICUS_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")

_token_cache = {"token": None, "expires": 0.0}


def _get_copernicus_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["expires"] - 60:
        return _token_cache["token"]
    resp = requests.post(
        COPERNICUS_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": config.COPERNICUS_CLIENT_ID,
            "client_secret": config.COPERNICUS_CLIENT_SECRET,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    _token_cache.update(token=body["access_token"], expires=time.time() + body.get("expires_in", 600))
    return body["access_token"]


def _cache_path(directory: str, *parts) -> str:
    os.makedirs(directory, exist_ok=True)
    key = hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]
    return os.path.join(directory, key + ".npz")


_S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "dataMask"] }],
    output: { bands: 5, sampleType: "FLOAT32" },
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.dataMask];
}
"""


def fetch_sentinel2_bands(bbox, date: str, window_days: int, max_cloud_pct: int, shape) -> dict:
    """Least-cloudy Sentinel-2 L2A mosaic around `date` from the Copernicus
    Process API. Returns {"B02","B03","B04","B08"} reflectance arrays."""
    path = _cache_path(config.IMAGERY_CACHE_DIR, "s2", bbox, date, window_days, max_cloud_pct, shape)
    if os.path.exists(path):
        with np.load(path) as cached:
            return {k: cached[k] for k in cached.files}

    from rasterio.io import MemoryFile

    start, end = date_window(date, window_days)
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                    "maxCloudCoverage": max_cloud_pct,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": shape[1], "height": shape[0],
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _S2_EVALSCRIPT,
    }
    resp = requests.post(
        COPERNICUS_PROCESS_URL,
        headers={"Authorization": f"Bearer {_get_copernicus_token()}"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    with MemoryFile(resp.content) as mem, mem.open() as src:
        arr = src.read().astype(np.float32)
    if arr[4].mean() < 0.05:
        raise ToolError(
            f"no Sentinel-2 acquisition with <= {max_cloud_pct}% cloud between {start} and {end}; "
            "try a larger window_days or max_cloud_pct"
        )
    bands = {"B02": arr[0], "B03": arr[1], "B04": arr[2], "B08": arr[3]}
    np.savez_compressed(path, **bands)
    return bands


def fetch_sentinel2_bands_ee(bbox, date: str, window_days: int, max_cloud_pct: int, shape) -> tuple:
    """Sentinel-2 composite from Google Earth Engine (cached like the Copernicus path)."""
    from services import earthengine

    path = _cache_path(config.IMAGERY_CACHE_DIR, "s2-ee", bbox, date, window_days, max_cloud_pct, shape)
    if os.path.exists(path):
        with np.load(path) as cached:
            bands = {k: cached[k] for k in cached.files}
        return bands, int(bands.pop("_scenes", np.array(0)))
    try:
        bands = earthengine.sentinel2_bands(bbox, date, window_days, max_cloud_pct, shape)
    except earthengine.EarthEngineError as exc:
        raise ToolError(str(exc)) from exc
    scenes = bands.pop("_scenes")
    np.savez_compressed(path, _scenes=np.array(scenes), **bands)
    return bands, scenes


def _bands_to_rgb(bands: dict) -> np.ndarray:
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1) * 2.5 * 255
    return np.clip(rgb, 0, 255).astype(np.uint8)


@tool(
    "fetch_sentinel2_scene",
    "Fetch a Sentinel-2 L2A scene (blue, green, red, NIR at 10 m) for a bounding box, using the "
    "least-cloudy acquisition within ±window_days of the date. Returns a scene_id "
    "('s2_<date>') that later tools reference. Fetch one scene per date you need.",
    {
        "bbox": _BBOX_SCHEMA,
        "date": {"type": "string", "description": "ISO date, e.g. 2024-06-01"},
        "window_days": {"type": "integer", "description": "Search window around the date (default 20)."},
        "max_cloud_pct": {"type": "integer", "description": "Maximum scene cloud cover % (default 30)."},
    },
    required=["date"],
)
def fetch_sentinel2_scene(ws: Workspace, date: str, bbox=None, window_days: int = 20, max_cloud_pct: int = 30) -> dict:
    bbox = ws.bbox(bbox)
    shape = grid_shape(bbox, S2_RESOLUTION_M)
    if config.data_live() and config.imagery_source() == "earthengine":
        bands, scenes = fetch_sentinel2_bands_ee(bbox, date, window_days, max_cloud_pct, shape)
        source = f"Google Earth Engine — Sentinel-2 L2A (cloud-masked median of {scenes} scenes)"
    elif config.data_live():
        bands = fetch_sentinel2_bands(bbox, date, window_days, max_cloud_pct, shape)
        source = "Copernicus Data Space — Sentinel-2 L2A (least-cloud mosaic)"
    else:
        bands = synthetic.synthetic_sentinel2(bbox, date, shape)
        source = "synthetic (offline demo — set Copernicus credentials for real imagery)"
    layer = ws.add(Layer(
        id=f"s2_{date}", kind="scene", bbox=bbox, name=f"Sentinel-2 {date}",
        data={"bands": bands, "rgb": _bands_to_rgb(bands)}, meta={"date": date},
    ))
    start, end = date_window(date, window_days)
    return {"scene_id": layer.id, "search_window": [start, end], "shape": list(shape),
            "resolution_m": S2_RESOLUTION_M, "source": source}


def _dem_tile_urls(bbox) -> list:
    min_lon, min_lat, max_lon, max_lat = bbox
    urls = []
    for lat in range(int(np.floor(min_lat)), int(np.floor(max_lat - 1e-9)) + 1):
        for lon in range(int(np.floor(min_lon)), int(np.floor(max_lon - 1e-9)) + 1):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
    return urls


def fetch_dem_array(bbox) -> np.ndarray:
    """Copernicus DEM GLO-30 over bbox, mosaicking every 1°×1° COG tile it touches."""
    path = _cache_path(config.DEM_TILE_CACHE_DIR, "dem", bbox)
    if os.path.exists(path):
        with np.load(path) as cached:
            return cached["dem"]

    import rasterio
    from rasterio.merge import merge

    sources = []
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for url in _dem_tile_urls(bbox):
            try:
                sources.append(rasterio.open(url))
            except rasterio.errors.RasterioIOError:
                log.info("DEM tile missing (ocean?): %s", url)
        if not sources:
            raise ToolError("no Copernicus DEM tiles cover this bbox (open ocean?)")
        try:
            mosaic, _ = merge(sources, bounds=tuple(bbox), nodata=_DEM_NODATA)
        finally:
            for src in sources:
                src.close()
    dem = _fill_dem_gaps(mosaic[0].astype(np.float32))
    rows, cols = dem.shape
    if max(rows, cols) > 1024:
        scale = 1024 / max(rows, cols)
        dem = cv2.resize(dem, (int(cols * scale), int(rows * scale)), interpolation=cv2.INTER_AREA)
    np.savez_compressed(path, dem=dem)
    return dem


_DEM_NODATA = -32768.0


def _fill_dem_gaps(dem: np.ndarray, seam_px: int = 3) -> np.ndarray:
    """GLO-30 tiles are offset by half a pixel, so a mosaic can have empty edge
    rows/columns. Fill thin seams from the nearest valid pixel; larger gaps are
    missing (ocean) tiles and become sea level."""
    invalid = dem == _DEM_NODATA
    if not invalid.any():
        return dem
    if invalid.all():
        return np.zeros_like(dem)
    dist, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL numbers every valid (zero-input) pixel in row-major order from 1.
    nearest = dem[~invalid][labels - 1]
    return np.where(invalid, np.where(dist <= seam_px, nearest, 0.0), dem).astype(np.float32)


@tool(
    "fetch_dem",
    "Fetch Copernicus DEM GLO-30 elevation (metres) for a bounding box. Stored as layer 'dem'.",
    {"bbox": _BBOX_SCHEMA},
)
def fetch_dem(ws: Workspace, bbox=None) -> dict:
    bbox = ws.bbox(bbox)
    if config.data_live():
        try:
            dem = fetch_dem_array(bbox)
            source = "Copernicus DEM GLO-30 (AWS open data COGs)"
        except Exception as exc:
            if not config.earthengine_configured():
                raise
            from services import earthengine
            log.info("AWS DEM unavailable (%s); using Earth Engine", exc)
            try:
                dem = earthengine.dem(bbox, grid_shape(bbox, DEM_RESOLUTION_M, max_px=1024))
            except earthengine.EarthEngineError as ee_exc:
                raise ToolError(str(ee_exc)) from ee_exc
            source = "Copernicus DEM GLO-30 (Google Earth Engine)"
    else:
        dem = synthetic.synthetic_dem(bbox, grid_shape(bbox, DEM_RESOLUTION_M, max_px=512))
        source = "synthetic (offline demo)"
    ws.add(Layer(id="dem", kind="dem", bbox=bbox, data=dem, name="Elevation (DEM)", meta={"unit": "m"}))
    return {"dem_id": "dem", "shape": list(dem.shape), "min_m": round(float(dem.min()), 1),
            "max_m": round(float(dem.max()), 1), "mean_m": round(float(dem.mean()), 1), "source": source}


_OSM_QUERIES = {
    "water": 'way["waterway"~"^(river|stream|canal)$"]({s},{w},{n},{e});way["natural"="water"]({s},{w},{n},{e});',
    "roads": 'way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});',
    "buildings": 'way["building"]({s},{w},{n},{e});',
}


def fetch_osm_geometries(bbox, feature_type: str) -> list:
    """OSM ways from the Overpass API as shapely geometries (lon/lat)."""
    from shapely.geometry import LineString, Polygon

    w, s, e, n = bbox
    query = f"[out:json][timeout:60];({_OSM_QUERIES[feature_type].format(s=s, w=w, n=n, e=e)});out geom;"
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=90,
                         headers={"User-Agent": "Geo-VLA research prototype"})
    resp.raise_for_status()
    geoms = []
    for el in resp.json().get("elements", []):
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        closed = coords[0] == coords[-1] and len(coords) >= 4
        is_area = feature_type == "buildings" or el.get("tags", {}).get("natural") == "water"
        geoms.append(Polygon(coords) if closed and is_area else LineString(coords))
    return geoms


@tool(
    "fetch_osm_features",
    "Fetch OpenStreetMap vector features for a bounding box: 'water' (rivers, streams, canals, "
    "lakes), 'roads' (major roads) or 'buildings'. Stored as layer 'osm_<feature_type>'. "
    "Use buffer_features on the result for proximity questions ('near rivers').",
    {"bbox": _BBOX_SCHEMA, "feature_type": {"type": "string", "enum": ["water", "roads", "buildings"]}},
    required=["feature_type"],
)
def fetch_osm_features(ws: Workspace, feature_type: str, bbox=None) -> dict:
    from shapely.geometry import LineString

    bbox = ws.bbox(bbox)
    if feature_type not in _OSM_QUERIES:
        raise ToolError(f"feature_type must be one of {list(_OSM_QUERIES)}")
    if config.data_live():
        geoms = fetch_osm_geometries(bbox, feature_type)
        source = "OpenStreetMap (Overpass API)"
    else:
        if feature_type == "water":
            geoms = [LineString(synthetic.synthetic_waterways(bbox))]
        elif feature_type == "roads":
            geoms = [LineString(line) for line in synthetic.synthetic_roads(bbox)]
        else:
            geoms = []
        source = "synthetic (offline demo — buildings are not simulated)"
    layer_id = f"osm_{feature_type}"
    ws.add(Layer(id=layer_id, kind="vector", bbox=bbox, data=geoms, name=f"OSM {feature_type}"))
    return {"layer_id": layer_id, "feature_count": len(geoms), "source": source}


# ---------------------------------------------------------------------------
# 2. Derived index calculations (classical raster math, no neural network)
# ---------------------------------------------------------------------------

def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index, range [-1, 1]."""
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def compute_slope(dem: np.ndarray, pixel_size_m=30.0) -> np.ndarray:
    """Terrain slope in degrees (finite differences). pixel_size_m is a scalar
    or (dx_m, dy_m) — geographic DEMs have different x/y spacing."""
    dx, dy = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    gy, gx = np.gradient(dem.astype(np.float32), dy, dx)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))


def flood_extent(dem: np.ndarray, water_level_m: float) -> np.ndarray:
    """Boolean flood mask under a bathtub (fixed water-level) model."""
    return dem <= water_level_m


@tool(
    "compute_ndvi",
    "Compute NDVI = (NIR - Red) / (NIR + Red) for a fetched Sentinel-2 scene. Stored as "
    "'ndvi_<date>'. Values > 0.4 usually indicate healthy vegetation.",
    {"scene_id": {"type": "string", "description": "Scene id; defaults to the latest scene."}},
)
def compute_ndvi_tool(ws: Workspace, scene_id: str = None) -> dict:
    scene = ws.resolve(scene_id, "scene")
    ndvi = compute_ndvi(scene.data["bands"]["B08"], scene.data["bands"]["B04"])
    date = scene.meta["date"]
    ws.add(Layer(id=f"ndvi_{date}", kind="scalar", bbox=scene.bbox, data=ndvi,
                 name=f"NDVI {date}", meta={"cmap": "ndvi", "vmin": -0.2, "vmax": 0.9, "date": date}))
    return {"layer_id": f"ndvi_{date}", "mean": round(float(ndvi.mean()), 3),
            "median": round(float(np.median(ndvi)), 3),
            "vegetated_fraction_ndvi_gt_0_4": round(float((ndvi > 0.4).mean()), 4)}


@tool(
    "compute_slope",
    "Compute terrain slope in degrees from a DEM layer. Stored as 'slope'.",
    {"dem_id": {"type": "string", "description": "DEM layer id (default 'dem')."}},
)
def compute_slope_tool(ws: Workspace, dem_id: str = None) -> dict:
    dem = ws.resolve(dem_id, "dem")
    slope = compute_slope(dem.data, pixel_size_m(dem.bbox, dem.data.shape))
    ws.add(Layer(id="slope", kind="scalar", bbox=dem.bbox, data=slope, name="Slope (°)",
                 meta={"cmap": "slope", "vmin": 0, "vmax": 35, "unit": "°"}))
    return {"layer_id": "slope", "mean_slope_deg": round(float(slope.mean()), 2),
            "p90_slope_deg": round(float(np.percentile(slope, 90)), 2),
            "max_slope_deg": round(float(slope.max()), 2),
            "steep_fraction_gt_15deg": round(float((slope > 15).mean()), 4)}


@tool(
    "flood_extent",
    "Bathtub flood model: mark every DEM cell at or below a water level. mode='above_min' "
    "interprets water_level_m as a rise above the lowest point in the area (e.g. a river "
    "rising 5 m); mode='absolute' treats it as metres above sea level.",
    {
        "water_level_m": {"type": "number"},
        "mode": {"type": "string", "enum": ["above_min", "absolute"], "description": "Default 'above_min'."},
        "dem_id": {"type": "string"},
    },
    required=["water_level_m"],
)
def flood_extent_tool(ws: Workspace, water_level_m: float, mode: str = "above_min", dem_id: str = None) -> dict:
    dem = ws.resolve(dem_id, "dem")
    level = float(water_level_m) + (float(dem.data.min()) if mode == "above_min" else 0.0)
    mask = flood_extent(dem.data, level)
    layer_id = f"flood_{water_level_m:g}m"
    ws.add(Layer(id=layer_id, kind="mask", bbox=dem.bbox, data=mask,
                 name=f"Flood extent (+{water_level_m:g} m)" if mode == "above_min" else f"Flood ≤ {level:g} m",
                 meta={"style": "flood", "water_level_m_asl": round(level, 2)}))
    return {"layer_id": layer_id, "water_level_m_asl": round(level, 2), **_mask_stats(mask, dem.bbox)}


# ---------------------------------------------------------------------------
# 3. Trained neural network inference (lazy-loaded; classical fallback)
# ---------------------------------------------------------------------------

_models: dict = {}


CHECKPOINT_FILES = {"classifier": "resnet50_eurosat.pth", "change": "siamese_unet_levircd.pth"}


def _checkpoint(name: str):
    path = os.path.join(config.MODEL_CHECKPOINT_DIR, name)
    return path if os.path.exists(path) else None


def _sidecar(key: str) -> dict:
    ckpt = _checkpoint(CHECKPOINT_FILES[key])
    sidecar = ckpt and ckpt + ".version.json"
    if sidecar and os.path.exists(sidecar):
        with open(sidecar) as f:
            return json.load(f)
    return {}


def model_version(key: str):
    """Version label of the active checkpoint (written by the model registry on promotion)."""
    return _sidecar(key).get("version") or ("unversioned" if _checkpoint(CHECKPOINT_FILES[key]) else None)


def classifier_input_size() -> int:
    """Side length the active classifier was trained at (224 unless the registry recorded otherwise)."""
    return int(_sidecar("classifier").get("input_size") or 224)


def reload_models() -> None:
    """Forget loaded models so the next call picks up a newly promoted checkpoint."""
    _models.clear()


def _load_model(key: str):
    """Load a trained model once. Returns (model, device) or (None, None) when
    the checkpoint (or torch) is unavailable."""
    if key in _models:
        return _models[key]
    ckpt = _checkpoint(CHECKPOINT_FILES[key])
    if ckpt is None:
        _models[key] = (None, None)
        return _models[key]
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if key == "classifier":
            from models.classifier import SceneClassifier
            model = SceneClassifier(num_classes=len(EUROSAT_CLASSES), pretrained=False)
        else:
            from models.change_detector import SiameseChangeDetector
            model = SiameseChangeDetector(pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        log.info("loaded %s checkpoint %s on %s", key, ckpt, device)
        _models[key] = (model, device)
    except Exception as exc:  # torch missing or incompatible checkpoint → classical fallback
        log.warning("could not load %s model (%s); using classical fallback", key, exc)
        _models[key] = (None, None)
    return _models[key]


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 → float32 3xHxW ImageNet-normalized."""
    return ((rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)


def _classify_cnn(model, device, rgb: np.ndarray, patch: int = 64) -> np.ndarray:
    """Tile the scene into 64×64 px patches (EuroSAT's native 640 m footprint at
    10 m), classify each with the ResNet-50, and paint the labels back."""
    import torch
    import torch.nn.functional as F

    rows, cols = rgb.shape[:2]
    pr, pc = int(np.ceil(rows / patch)), int(np.ceil(cols / patch))
    padded = np.pad(rgb, ((0, pr * patch - rows), (0, pc * patch - cols), (0, 0)), mode="reflect")
    tiles = padded.reshape(pr, patch, pc, patch, 3).transpose(0, 2, 1, 3, 4).reshape(-1, patch, patch, 3)
    labels = []
    with torch.no_grad():
        for i in range(0, len(tiles), 64):
            batch = torch.from_numpy(np.stack([_normalize(t) for t in tiles[i:i + 64]])).to(device)
            size = classifier_input_size()
            batch = F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)
            labels.append(model(batch).argmax(1).cpu().numpy())
    grid = np.concatenate(labels).reshape(pr, pc)
    return np.kron(grid, np.ones((patch, patch), dtype=np.int64))[:rows, :cols].astype(np.int8)


def _classify_spectral(bands: dict) -> np.ndarray:
    """Rule-based per-pixel land cover from reflectance (fallback when no checkpoint)."""
    b, g, r, nir = (bands[k] for k in ("B02", "B03", "B04", "B08"))
    ndvi = compute_ndvi(nir, r)
    idx = {c: i for i, c in enumerate(EUROSAT_CLASSES)}
    out = np.full(r.shape, idx["HerbaceousVegetation"], dtype=np.int8)
    out[(ndvi > 0.35)] = idx["AnnualCrop"]
    out[(ndvi > 0.6) & (g < 0.06)] = idx["Forest"]
    out[(ndvi < 0.3) & (r > g)] = idx["AnnualCrop"]                    # bare / ploughed soil
    out[(ndvi < 0.25) & ((b + g + r) / 3 > 0.09)] = idx["Residential"]
    out[(nir < 0.05) & (g > nir)] = idx["River"]
    return out


@tool(
    "classify_scene",
    "Land-cover classification of a Sentinel-2 scene into the 10 EuroSAT classes (AnnualCrop, "
    "Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, "
    "River, SeaLake) using the trained ResNet-50 on 64×64 px patches. Stored as "
    "'landcover_<date>'; returns class fractions.",
    {"scene_id": {"type": "string", "description": "Scene id; defaults to the latest scene."}},
)
def classify_scene(ws: Workspace, scene_id: str = None) -> dict:
    scene = ws.resolve(scene_id, "scene")
    model, device = _load_model("classifier")
    if model is not None:
        classmap = _classify_cnn(model, device, scene.data["rgb"])
        method = f"ResNet-50 fine-tuned on EuroSAT, model {model_version('classifier')} (64 px patches)"
    else:
        classmap = _classify_spectral(scene.data["bands"])
        method = "spectral rules (fallback — no resnet50_eurosat.pth checkpoint)"
    date = scene.meta["date"]
    layer_id = f"landcover_{date}"
    ws.add(Layer(id=layer_id, kind="classmap", bbox=scene.bbox, data=classmap, name=f"Land cover {date}",
                 meta={"date": date}))
    counts = np.bincount(classmap.ravel().astype(np.int64), minlength=len(EUROSAT_CLASSES)) / classmap.size
    fractions = {c: round(float(f), 4) for c, f in zip(EUROSAT_CLASSES, counts) if f >= 0.005}
    return {"layer_id": layer_id, "dominant_class": EUROSAT_CLASSES[int(counts.argmax())],
            "class_fractions": dict(sorted(fractions.items(), key=lambda kv: -kv[1])), "method": method}


def _change_cnn(model, device, rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
    import torch

    rows, cols = rgb1.shape[:2]
    pad = ((0, (-rows) % 32), (0, (-cols) % 32), (0, 0))
    t1 = torch.from_numpy(_normalize(np.pad(rgb1, pad, mode="reflect")))[None].to(device)
    t2 = torch.from_numpy(_normalize(np.pad(rgb2, pad, mode="reflect")))[None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t1, t2)).squeeze().cpu().numpy()
    return prob[:rows, :cols] > 0.5


def _change_cva(bands1: dict, bands2: dict) -> np.ndarray:
    """Change-vector analysis: Otsu threshold on the spectral difference magnitude."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, opening

    diff = np.sqrt(sum((bands2[k] - bands1[k]) ** 2 for k in ("B02", "B03", "B04", "B08")))
    thresh = max(float(threshold_otsu(diff)), 0.08)   # floor avoids flagging pure noise
    return opening(diff > thresh, disk(1)).astype(bool)


@tool(
    "detect_change",
    "Pixel-level change detection between two fetched scenes of the same area at different "
    "dates, using the trained Siamese U-Net. Stored as a mask 'change_<date1>_<date2>'. "
    "For *what* changed (e.g. forest loss), compare class masks with overlay_layers instead.",
    {"scene_id_1": {"type": "string", "description": "Earlier scene id"},
     "scene_id_2": {"type": "string", "description": "Later scene id"}},
    required=["scene_id_1", "scene_id_2"],
)
def detect_change(ws: Workspace, scene_id_1: str, scene_id_2: str) -> dict:
    s1, s2 = ws.get(scene_id_1, {"scene"}), ws.get(scene_id_2, {"scene"})
    if not same_bbox(s1.bbox, s2.bbox):
        raise ToolError("scenes cover different bboxes; fetch both dates over the same area")
    model, device = _load_model("change")
    if model is not None:
        mask = _change_cnn(model, device, s1.data["rgb"], s2.data["rgb"])
        method = f"Siamese U-Net fine-tuned on LEVIR-CD, model {model_version('change')}"
    else:
        mask = _change_cva(s1.data["bands"], s2.data["bands"])
        method = "change-vector analysis + Otsu (fallback — no siamese_unet_levircd.pth checkpoint)"
    layer_id = f"change_{s1.meta['date']}_{s2.meta['date']}"
    ws.add(Layer(id=layer_id, kind="mask", bbox=s1.bbox, data=mask,
                 name=f"Change {s1.meta['date']} → {s2.meta['date']}", meta={"style": "change"}))
    return {"layer_id": layer_id, **_mask_stats(mask, s1.bbox), "method": method}


# ---------------------------------------------------------------------------
# 4. Spatial analysis (buffer / overlay / zonal statistics)
# ---------------------------------------------------------------------------

@tool(
    "threshold_layer",
    "Turn a continuous layer (NDVI, slope, DEM) into a boolean mask, e.g. slope > 15 or ndvi < 0.2.",
    {
        "layer_id": {"type": "string"},
        "operator": {"type": "string", "enum": ["gt", "ge", "lt", "le"]},
        "value": {"type": "number"},
    },
    required=["layer_id", "operator", "value"],
)
def threshold_layer(ws: Workspace, layer_id: str, operator: str, value: float) -> dict:
    layer = ws.get(layer_id, {"scalar", "dem"})
    ops = {"gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal}
    if operator not in ops:
        raise ToolError(f"operator must be one of {list(ops)}")
    mask = ops[operator](layer.data, value)
    new_id = f"{layer_id}_{operator}_{value:g}"
    ws.add(Layer(id=new_id, kind="mask", bbox=layer.bbox, data=mask,
                 name=f"{layer.name} {operator} {value:g}", meta={"style": "threshold"}))
    return {"layer_id": new_id, **_mask_stats(mask, layer.bbox)}


CLASS_GROUPS = {
    "built_up": ["Residential", "Industrial", "Highway"],
    "cropland": ["AnnualCrop", "PermanentCrop"],
    "vegetation": ["Forest", "HerbaceousVegetation", "Pasture"],
    "water": ["River", "SeaLake"],
}


def _resolve_classes(class_name=None, class_names=None, group=None) -> list:
    names = list(class_names or []) + ([class_name] if class_name else []) + CLASS_GROUPS.get(group or "", [])
    if group and group not in CLASS_GROUPS:
        raise ToolError(f"group must be one of {list(CLASS_GROUPS)}")
    bad = [n for n in names if n not in EUROSAT_CLASSES]
    if bad or not names:
        raise ToolError(f"unknown or missing classes {bad}; choose from {EUROSAT_CLASSES}")
    return names


@tool(
    "class_mask",
    "Extract a boolean mask of land-cover classes from a classification layer: one class "
    "(class_name='Forest'), several (class_names), or a group — built_up (Residential, "
    "Industrial, Highway), cropland, vegetation, water. The id is '<class or group>_<date>', "
    "e.g. 'forest_2021-06-01', 'built_up_2024-06-01'. Combine dates with overlay_layers.",
    {"layer_id": {"type": "string"},
     "class_name": {"type": "string", "enum": EUROSAT_CLASSES},
     "class_names": {"type": "array", "items": {"type": "string", "enum": EUROSAT_CLASSES}},
     "group": {"type": "string", "enum": list(CLASS_GROUPS)}},
    required=["layer_id"],
)
def class_mask(ws: Workspace, layer_id: str, class_name: str = None, class_names: list = None,
               group: str = None) -> dict:
    layer = ws.get(layer_id, {"classmap"})
    names = _resolve_classes(class_name, class_names, group)
    mask = np.isin(layer.data, [EUROSAT_CLASSES.index(n) for n in names])
    label = group or ("_".join(n.lower() for n in names) if len(names) > 1 else names[0].lower())
    new_id = f"{label}_{layer_id.removeprefix('landcover_')}"
    pretty = group.replace("_", "-").capitalize() if group else " + ".join(names)
    ws.add(Layer(id=new_id, kind="mask", bbox=layer.bbox, data=mask,
                 name=f"{pretty} ({layer.name})", meta={"style": "class", "date": layer.meta.get("date")}))
    return {"layer_id": new_id, "classes": names, **_mask_stats(mask, layer.bbox)}


@tool(
    "buffer_features",
    "Create a proximity zone: every pixel within distance_m of a vector layer (e.g. 'osm_water') "
    "or of the True pixels of a mask layer. Stored as mask 'buffer_<layer>_<d>m'.",
    {"layer_id": {"type": "string"}, "distance_m": {"type": "number"}},
    required=["layer_id", "distance_m"],
)
def buffer_features(ws: Workspace, layer_id: str, distance_m: float) -> dict:
    layer = ws.get(layer_id, {"vector", "mask"})
    if layer.kind == "vector":
        shape = ws.raster_shape(layer.bbox)
        mask = _rasterize_buffer(layer.data, layer.bbox, shape, distance_m)
    else:
        dx, dy = pixel_size_m(layer.bbox, layer.data.shape)
        # Distance transform in pixel units; use the mean pixel size to convert.
        dist_px = cv2.distanceTransform((~layer.data).astype(np.uint8), cv2.DIST_L2, 5)
        mask = dist_px * (dx + dy) / 2 <= distance_m
    new_id = f"buffer_{layer_id}_{distance_m:g}m"
    ws.add(Layer(id=new_id, kind="mask", bbox=layer.bbox, data=mask,
                 name=f"Within {distance_m:g} m of {layer.name}", meta={"style": "buffer"}))
    return {"layer_id": new_id, **_mask_stats(mask, layer.bbox)}


def _rasterize_buffer(geoms: list, bbox, shape, distance_m: float) -> np.ndarray:
    if not geoms:
        return np.zeros(shape, dtype=bool)
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    series = gpd.GeoSeries(geoms, crs="EPSG:4326")
    utm = series.estimate_utm_crs()
    buffered = series.to_crs(utm).buffer(distance_m).to_crs("EPSG:4326")
    transform = from_bounds(*bbox, shape[1], shape[0])
    return rasterize(((g, 1) for g in buffered if not g.is_empty), out_shape=shape,
                     transform=transform, fill=0, dtype="uint8").astype(bool)


@tool(
    "overlay_layers",
    "Combine two boolean mask layers: 'intersect' (A and B), 'union' (A or B), or 'difference' "
    "(A and not B — e.g. forest_2021 minus forest_2024 = forest loss). Different grids over the "
    "same area are resampled automatically.",
    {
        "layer_a": {"type": "string"},
        "layer_b": {"type": "string"},
        "operation": {"type": "string", "enum": ["intersect", "union", "difference"]},
        "name": {"type": "string", "description": "Optional short id for the result, e.g. 'forest_loss'."},
    },
    required=["layer_a", "layer_b", "operation"],
)
def overlay_layers(ws: Workspace, layer_a: str, layer_b: str, operation: str, name: str = None) -> dict:
    a, b = ws.get(layer_a, {"mask"}), ws.get(layer_b, {"mask"})
    if not same_bbox(a.bbox, b.bbox):
        raise ToolError("layers cover different bboxes")
    mb = _align(b.data, a.data.shape, categorical=True)
    ops = {"intersect": np.logical_and, "union": np.logical_or,
           "difference": lambda x, y: np.logical_and(x, ~y)}
    if operation not in ops:
        raise ToolError(f"operation must be one of {list(ops)}")
    mask = ops[operation](a.data, mb)
    new_id = name or f"{operation}_{layer_a}_{layer_b}"
    symbol = {"intersect": "∩", "union": "∪", "difference": "−"}[operation]
    label = name.replace("_", " ").capitalize() if name else f"{a.name} {symbol} {b.name}"
    ws.add(Layer(id=new_id, kind="mask", bbox=a.bbox, data=mask, name=label, meta={"style": "overlay"}))
    share = float(mask.sum() / a.data.sum()) if a.data.any() else 0.0
    return {"layer_id": new_id, **_mask_stats(mask, a.bbox), "fraction_of_layer_a": round(share, 4)}


@tool(
    "zonal_stats",
    "Statistics of a value layer inside a zone mask — e.g. mean slope inside the forest-loss "
    "zone, or the land-cover mix inside a flood zone.",
    {"value_layer": {"type": "string", "description": "scalar, DEM or land-cover layer id"},
     "zone_layer": {"type": "string", "description": "mask layer id"}},
    required=["value_layer", "zone_layer"],
)
def zonal_stats(ws: Workspace, value_layer: str, zone_layer: str) -> dict:
    values = ws.get(value_layer, {"scalar", "dem", "classmap"})
    zone = ws.get(zone_layer, {"mask"})
    if not same_bbox(values.bbox, zone.bbox):
        raise ToolError("layers cover different bboxes")
    categorical = values.kind == "classmap"
    data = _align(values.data, zone.data.shape, categorical=categorical)
    inside = data[zone.data]
    result = {"zone_pixels": int(inside.size), **_mask_stats(zone.data, zone.bbox)}
    if inside.size == 0:
        return {**result, "note": "zone is empty"}
    if categorical:
        counts = np.bincount(inside.astype(np.int64), minlength=len(EUROSAT_CLASSES)) / inside.size
        result["class_fractions"] = {c: round(float(f), 4) for c, f in zip(EUROSAT_CLASSES, counts) if f >= 0.005}
    else:
        result.update(mean=round(float(inside.mean()), 3), median=round(float(np.median(inside)), 3),
                      min=round(float(inside.min()), 3), max=round(float(inside.max()), 3),
                      std=round(float(inside.std()), 3))
    return result


@tool(
    "time_series",
    "Track an indicator across several dates (2-12): 'ndvi' (mean NDVI) or 'class_fraction' "
    "(share of a land-cover class or group, e.g. group='built_up'). Fetches and analyses a "
    "scene per date (skipping dates without usable imagery) and returns the series; the "
    "scenes become a timelapse in the UI. Optionally restrict to a zone mask.",
    {
        "dates": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 12},
        "index": {"type": "string", "enum": ["ndvi", "class_fraction"]},
        "class_name": {"type": "string", "enum": EUROSAT_CLASSES},
        "group": {"type": "string", "enum": list(CLASS_GROUPS)},
        "zone_layer": {"type": "string", "description": "Optional mask layer to restrict the statistic to."},
        "bbox": _BBOX_SCHEMA,
    },
    required=["dates", "index"],
)
def time_series(ws: Workspace, dates: list, index: str, class_name: str = None, group: str = None,
                zone_layer: str = None, bbox=None) -> dict:
    if not 2 <= len(dates) <= 12:
        raise ToolError("give between 2 and 12 dates")
    if index not in ("ndvi", "class_fraction"):
        raise ToolError("index must be 'ndvi' or 'class_fraction'")
    targets = _resolve_classes(class_name, None, group) if index == "class_fraction" else None
    zone = ws.get(zone_layer, {"mask"}) if zone_layer else None
    series, skipped = [], []
    for date in sorted(dates):
        try:
            fetch_sentinel2_scene(ws, date=date, bbox=bbox)
        except ToolError as exc:
            skipped.append({"date": date, "reason": str(exc)})
            continue
        scene = ws.get(f"s2_{date}")
        scene.meta["group"] = "timeseries"
        if index == "ndvi":
            compute_ndvi_tool(ws, scene_id=scene.id)
            values = ws.get(f"ndvi_{date}").data
        else:
            classify_scene(ws, scene_id=scene.id)
            values = np.isin(ws.get(f"landcover_{date}").data, [EUROSAT_CLASSES.index(n) for n in targets])
        ws.get(f"{'ndvi' if index == 'ndvi' else 'landcover'}_{date}").meta["group"] = "timeseries"
        if zone is not None:
            values = values[_align(zone.data, values.shape, categorical=True)]
        value = float(values.mean()) if values.size else float("nan")
        series.append({"date": date, "value": round(value, 4)})
    if len(series) < 2:
        raise ToolError(f"fewer than 2 dates had usable imagery: {skipped}")
    label = "mean NDVI" if index == "ndvi" else f"fraction {group or class_name}"
    first, last = series[0]["value"], series[-1]["value"]
    return {"indicator": label, "series": series, "change": round(last - first, 4),
            "relative_change_pct": round(100 * (last - first) / first, 1) if first else None,
            "skipped": skipped}


def _mask_sampler(mask: np.ndarray, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    rows, cols = mask.shape

    def inside(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        c = ((lons - min_lon) / (max_lon - min_lon) * cols).astype(int)
        r = ((max_lat - lats) / (max_lat - min_lat) * rows).astype(int)
        ok = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        out = np.zeros(lons.shape, dtype=bool)
        out[ok] = mask[r[ok], c[ok]]
        return out
    return inside


@tool(
    "features_in_zone",
    "Count OSM features (roads, rivers, buildings) that intersect a zone mask and measure the "
    "length (km) of lines inside it — e.g. kilometres of road inside a flood zone, or "
    "buildings inside a change zone.",
    {"layer_id": {"type": "string", "description": "vector layer, e.g. 'osm_roads'"},
     "zone_layer": {"type": "string", "description": "mask layer id"}},
    required=["layer_id", "zone_layer"],
)
def features_in_zone(ws: Workspace, layer_id: str, zone_layer: str) -> dict:
    import geopandas as gpd
    import shapely
    from pyproj import Transformer

    vec = ws.get(layer_id, {"vector"})
    zone = ws.get(zone_layer, {"mask"})
    if not vec.data:
        return {"features_total": 0, "features_in_zone": 0, "length_km_in_zone": 0.0}
    inside = _mask_sampler(zone.data, zone.bbox)
    series = gpd.GeoSeries(vec.data, crs="EPSG:4326")
    utm = series.estimate_utm_crs()
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    metric = series.to_crs(utm)
    hit_count, length_m = 0, 0.0
    step = 10.0  # metres between sample points along lines
    for geom in metric:
        line = geom.boundary if geom.geom_type == "Polygon" else geom
        if line.is_empty or line.length == 0:
            continue
        n = max(int(line.length // step), 1)
        pts = shapely.line_interpolate_point(line, (np.arange(n) + 0.5) * line.length / n)
        xy = shapely.get_coordinates(pts)
        lon, lat = to_ll.transform(xy[:, 0], xy[:, 1])
        hits = inside(np.asarray(lon), np.asarray(lat))
        if geom.geom_type == "Polygon":
            c = geom.centroid
            clon, clat = to_ll.transform(c.x, c.y)
            hits = np.append(hits, inside(np.array([clon]), np.array([clat])))
        elif hits.any():
            length_m += line.length * hits.mean()
        hit_count += bool(hits.any())
    return {"features_total": len(vec.data), "features_in_zone": hit_count,
            "length_km_in_zone": round(length_m / 1000, 3)}


# ---------------------------------------------------------------------------
# Layer styling + rendering for the web map
# ---------------------------------------------------------------------------

def layer_style(layer: Layer) -> dict:
    """Style computed once from the full-resolution data (cached on the layer)."""
    if "_style" not in layer.meta:
        data = layer.data["rgb"] if layer.kind == "scene" else layer.data
        layer.meta["_style"] = rendering.style_for(layer.kind, data, layer.meta)
    return layer.meta["_style"]


def vector_geojson(layer: Layer) -> dict:
    from shapely.geometry import mapping
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(g), "properties": {}} for g in layer.data]}


def layer_summary(layer: Layer) -> dict:
    """JSON-safe description of a layer (no pixel data)."""
    meta = {k: v for k, v in layer.meta.items() if not k.startswith("_")}
    out = {"id": layer.id, "name": layer.name, "kind": layer.kind, "bbox": layer.bbox,
           "legend": layer_style(layer)["legend"], "meta": meta}
    if layer.kind == "mask":
        out["stats"] = _mask_stats(layer.data, layer.bbox)
    return out


def render_layer(layer: Layer) -> dict:
    """Layer summary plus an inline preview image (used by the synchronous /query API)."""
    out = layer_summary(layer)
    if layer.kind == "vector":
        out["geojson"] = vector_geojson(layer)
    else:
        data = layer.data["rgb"] if layer.kind == "scene" else layer.data
        out["image"] = rendering.data_url(rendering.preview_png(layer.kind, data, layer_style(layer)))
    return out
