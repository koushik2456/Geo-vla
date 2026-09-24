"""
rendering.py — Colour styling for layers, shared by every output.

A layer's *style* (colour ramp, value range, mask colour, classes present) is
computed once from the full-resolution data and stored with the layer. The
same style then drives:
  * map tiles        (services/tiles.py — any zoom, full resolution)
  * previews         (small PNG/JPEG for thumbnails and the sync /query API)
  * PDF report maps  (services/reports.py)
so colours and legends are identical everywhere.
"""
import base64
import io

import numpy as np
from PIL import Image

from models import EUROSAT_CLASSES

# (position, (r, g, b)) colour stops
COLORMAPS = {
    "ndvi": [(0.0, (165, 0, 38)), (0.35, (244, 109, 67)), (0.5, (255, 255, 191)), (0.7, (166, 217, 106)), (1.0, (0, 104, 55))],
    "terrain": [(0.0, (51, 102, 153)), (0.25, (94, 158, 94)), (0.5, (230, 220, 140)), (0.75, (153, 102, 51)), (1.0, (255, 255, 255))],
    "slope": [(0.0, (255, 255, 204)), (0.33, (254, 204, 92)), (0.66, (240, 59, 32)), (1.0, (128, 0, 38))],
    "viridis": [(0.0, (68, 1, 84)), (0.5, (33, 145, 140)), (1.0, (253, 231, 37))],
}

MASK_COLORS = {
    "change": (255, 64, 64),
    "flood": (30, 120, 255),
    "buffer": (255, 200, 0),
    "overlay": (255, 0, 255),
    "threshold": (0, 230, 200),
    "class": (120, 255, 120),
    "default": (255, 128, 0),
}

CLASS_COLORS = {
    "AnnualCrop": (230, 200, 90), "Forest": (20, 110, 40), "HerbaceousVegetation": (150, 200, 90),
    "Highway": (120, 120, 120), "Industrial": (200, 60, 200), "Pasture": (190, 230, 120),
    "PermanentCrop": (220, 150, 60), "Residential": (230, 60, 60), "River": (40, 120, 230),
    "SeaLake": (20, 60, 160),
}

SCALAR_ALPHA, MASK_ALPHA, CLASS_ALPHA = 200, 170, 190
MAX_PREVIEW_PX = 512


def rgb_css(color) -> str:
    return "rgb(%d,%d,%d)" % tuple(color)


# -- style (computed once per layer) -------------------------------------------

def style_for(kind: str, data, meta: dict) -> dict:
    """Rendering parameters + legend for a layer, from its full-resolution data."""
    if kind == "scene":
        return {"legend": {"type": "none"}}
    if kind in ("dem", "scalar"):
        arr = np.asarray(data, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        cmap = "terrain" if kind == "dem" else meta.get("cmap", "viridis")
        vmin = meta.get("vmin")
        vmax = meta.get("vmax")
        vmin = float(np.percentile(finite, 2)) if vmin is None and finite.size else (vmin or 0.0)
        vmax = float(np.percentile(finite, 98)) if vmax is None and finite.size else (vmax or 1.0)
        return {"cmap": cmap, "vmin": vmin, "vmax": vmax,
                "legend": {"type": "gradient", "min": round(vmin, 3), "max": round(vmax, 3),
                           "unit": meta.get("unit", ""),
                           "colors": [rgb_css(s[1]) for s in COLORMAPS[cmap]]}}
    if kind == "mask":
        color = MASK_COLORS.get(meta.get("style", "default"), MASK_COLORS["default"])
        return {"color": list(color), "legend": {"type": "mask", "color": rgb_css(color)}}
    if kind == "classmap":
        present = np.unique(np.asarray(data)).tolist()
        classes = {EUROSAT_CLASSES[i]: rgb_css(CLASS_COLORS[EUROSAT_CLASSES[i]])
                   for i in present if 0 <= i < len(EUROSAT_CLASSES)}
        return {"legend": {"type": "categorical", "classes": classes}}
    return {"legend": {"type": "vector"}}


# -- colouring (any window of the data) -------------------------------------------

def _apply_cmap(norm: np.ndarray, stops) -> np.ndarray:
    pos = np.array([s[0] for s in stops])
    rgb = np.stack([np.interp(norm, pos, [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    return rgb.astype(np.uint8)


_CLASS_LUT = np.array([(*CLASS_COLORS[c], CLASS_ALPHA) for c in EUROSAT_CLASSES], dtype=np.uint8)


def colorize(kind: str, data: np.ndarray, style: dict, valid: np.ndarray = None) -> np.ndarray:
    """RGBA uint8 image for `data` (a full layer or a resampled tile window).
    `valid` marks pixels inside the layer footprint; others are transparent."""
    if kind == "scene":
        rgba = np.concatenate([data.astype(np.uint8), np.full(data.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    elif kind in ("dem", "scalar"):
        arr = data.astype(np.float32)
        norm = np.clip((arr - style["vmin"]) / max(style["vmax"] - style["vmin"], 1e-9), 0, 1)
        rgb = _apply_cmap(np.nan_to_num(norm), COLORMAPS[style["cmap"]])
        alpha = np.where(np.isfinite(arr), SCALAR_ALPHA, 0).astype(np.uint8)[..., None]
        rgba = np.concatenate([rgb, alpha], axis=-1)
    elif kind == "mask":
        rgba = np.zeros(data.shape + (4,), dtype=np.uint8)
        rgba[data.astype(bool)] = (*style["color"], MASK_ALPHA)
    elif kind == "classmap":
        rgba = _CLASS_LUT[np.clip(data.astype(np.int64), 0, len(EUROSAT_CLASSES) - 1)]
    else:
        raise ValueError(f"cannot colorize layer kind {kind}")
    if valid is not None:
        rgba = rgba.copy()
        rgba[~valid] = 0
    return rgba


def downsample(arr: np.ndarray, max_px: int = MAX_PREVIEW_PX) -> np.ndarray:
    """Nearest-neighbour thinning so previews stay small."""
    step = int(np.ceil(max(arr.shape[:2]) / max_px))
    return arr[::step, ::step] if step > 1 else arr


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def preview_png(kind: str, data: np.ndarray, style: dict, max_px: int = MAX_PREVIEW_PX) -> bytes:
    return png_bytes(colorize(kind, downsample(data, max_px), style))


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
