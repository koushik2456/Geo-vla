"""
services/terrain.py — 3D terrain model for an analysis.

Serves the data the 3D studio (frontend/src/components/TerrainView.jsx) renders:

  terrain_model()   height grid (≤ 256×256, float32, base64), extent in metres,
                    elevation range, and contour lines (major/minor, labelled)
  texture_png()     derived surface textures from the DEM:
                      height       normalised elevation (greyscale)
                      normal       tangent-space normal map (RGB = xyz), the classic purple/blue look
                      ao           ambient occlusion (multi-scale local depressions)
                      hillshade    Lambertian shading, sun from the north-west
                      hypsometric  elevation colour ramp × hillshade

Analyses that never loaded a DEM (e.g. urban growth) get one fetched on demand
and cached next to the run, so every analysis can be explored in 3D.
"""
import base64
import os

import cv2
import numpy as np

import geotools
import rendering
from geo_utils import bbox_size_m
from services import storage

MAX_GRID = 256
TEXTURE_PX = 1024
HYPSOMETRIC = [(0.0, (38, 30, 140)), (0.2, (84, 48, 190)), (0.4, (150, 70, 180)), (0.6, (225, 110, 120)),
               (0.8, (250, 170, 80)), (1.0, (255, 236, 150))]


def dem_for(run: dict) -> np.ndarray:
    dem_layer = next((l for l in run["layers"] if l["kind"] == "dem"), None)
    if dem_layer:
        return storage.load_array(run["id"], dem_layer["id"]).astype(np.float32)
    cache = os.path.join(storage.run_dir(run["id"]), "_terrain_dem.npy")
    if os.path.exists(cache):
        return np.load(cache)
    ws = geotools.Workspace(run["bbox"])
    geotools.fetch_dem(ws)
    dem = ws.get("dem").data.astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, dem)
    return dem


def _resize(arr: np.ndarray, max_px: int) -> np.ndarray:
    rows, cols = arr.shape
    scale = min(1.0, max_px / max(rows, cols))
    if scale >= 1.0:
        return arr
    return cv2.resize(arr, (max(2, int(cols * scale)), max(2, int(rows * scale))), interpolation=cv2.INTER_AREA)


def _nice_interval(span: float, target: int = 20) -> float:
    raw = max(span / target, 1e-6)
    mag = 10 ** np.floor(np.log10(raw))
    return float(min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag))


def contours(grid: np.ndarray, interval: float, target_shape: tuple = None) -> list:
    """Contour polylines per level as [[row, col], …] in the coordinates of `target_shape`
    (the terrain mesh grid), even when traced on a coarser grid. Every 5th level is major."""
    from skimage.measure import find_contours
    target_shape = target_shape or grid.shape
    scale = np.array([(target_shape[0] - 1) / max(grid.shape[0] - 1, 1), (target_shape[1] - 1) / max(grid.shape[1] - 1, 1)])
    lo = np.ceil(grid.min() / interval) * interval
    out = []
    for level in np.arange(lo, grid.max(), interval):
        major = round(level / interval) % 5 == 0
        lines = [np.round(c * scale, 2).tolist() for c in find_contours(grid, level) if len(c) >= 4]
        if lines:
            out.append({"level": round(float(level), 2), "major": bool(major), "lines": lines})
    return out


def terrain_model(run: dict) -> dict:
    dem = dem_for(run)
    grid = _resize(dem, MAX_GRID)
    width_m, height_m = bbox_size_m(run["bbox"])
    span = float(grid.max() - grid.min())
    interval = _nice_interval(span)
    water = [{"layer_id": l["id"], "name": l["name"], "level": l["meta"]["water_level_m_asl"]}
             for l in run["layers"] if l.get("meta", {}).get("water_level_m_asl") is not None]
    return {
        "bbox": run["bbox"], "rows": grid.shape[0], "cols": grid.shape[1],
        "heights": base64.b64encode(grid.astype("<f4").tobytes()).decode(),
        "min": round(float(grid.min()), 2), "max": round(float(grid.max()), 2),
        "mean": round(float(grid.mean()), 2),
        "width_m": round(width_m, 1), "height_m": round(height_m, 1),
        "contour_interval": interval, "contours": contours(grid, interval),
        # Dense set for the "contour lines" render mode (topographic-survey look).
        "dense_contours": contours(_resize(grid, 160), _nice_interval(span, 60), grid.shape),
        "water": water, "source_shape": list(dem.shape),
        "textures": ["hypsometric", "hillshade", "height", "normal", "ao"],
    }


# -- derived textures ----------------------------------------------------------------------------

def _gradients(dem: np.ndarray, bbox) -> tuple:
    w, h = bbox_size_m(bbox)
    dx, dy = w / dem.shape[1], h / dem.shape[0]
    gy, gx = np.gradient(dem, dy, dx)
    return gx, gy


def hillshade(dem, bbox, azimuth=315.0, altitude=45.0) -> np.ndarray:
    gx, gy = _gradients(dem, bbox)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def normal_map(dem, bbox, strength: float = 4.0) -> np.ndarray:
    gx, gy = _gradients(dem, bbox)
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(dem)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def ambient_occlusion(dem) -> np.ndarray:
    """Multi-scale approximation: pixels lower than their neighbourhood receive less sky light."""
    span = max(float(dem.max() - dem.min()), 1e-6)
    occ = np.zeros_like(dem)
    for k in (3, 9, 27, 81):
        k = min(k, (min(dem.shape) // 2) * 2 - 1)
        if k < 3:
            continue
        occ += np.clip(cv2.blur(dem, (k, k)) - dem, 0, None) / span
    return np.clip(1 - 6.0 * occ, 0, 1) ** 1.5


def texture_png(run: dict, kind: str) -> bytes:
    dem = _resize(dem_for(run), TEXTURE_PX)
    dem = cv2.resize(dem, (dem.shape[1] * 2, dem.shape[0] * 2), interpolation=cv2.INTER_CUBIC) \
        if max(dem.shape) < TEXTURE_PX // 2 else dem
    norm = (dem - dem.min()) / max(float(dem.max() - dem.min()), 1e-6)
    if kind == "height":
        rgb = np.dstack([(norm * 255).astype(np.uint8)] * 3)
    elif kind == "normal":
        rgb = normal_map(dem, run["bbox"])
    elif kind == "ao":
        rgb = np.dstack([(ambient_occlusion(dem) * 255).astype(np.uint8)] * 3)
    elif kind == "hillshade":
        rgb = np.dstack([(hillshade(dem, run["bbox"]) * 255).astype(np.uint8)] * 3)
    elif kind == "hypsometric":
        tint = rendering._apply_cmap(norm, HYPSOMETRIC).astype(np.float32)
        shade = 0.55 + 0.45 * hillshade(dem, run["bbox"])[..., None]
        rgb = np.clip(tint * shade, 0, 255).astype(np.uint8)
    else:
        raise KeyError(kind)
    alpha = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    return rendering.png_bytes(np.concatenate([rgb, alpha], axis=-1))
