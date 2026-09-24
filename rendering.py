"""rendering.py — Turn raster layers into RGBA PNG overlays for the web map."""
import base64
import io

import numpy as np
from PIL import Image

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


MAX_RENDER_PX = 512


def _downsample(arr: np.ndarray, categorical: bool = False) -> np.ndarray:
    """Cap overlay size so API responses stay small; the map stretches it to the bbox."""
    step = int(np.ceil(max(arr.shape[:2]) / MAX_RENDER_PX))
    if step <= 1:
        return arr
    if categorical or arr.ndim == 3:
        return arr[::step, ::step]
    rows, cols = (arr.shape[0] // step) * step, (arr.shape[1] // step) * step
    return arr[:rows, :cols].reshape(rows // step, step, cols // step, step).mean(axis=(1, 3))


def _apply_cmap(norm: np.ndarray, stops) -> np.ndarray:
    pos = np.array([s[0] for s in stops])
    rgb = np.stack([np.interp(norm, pos, [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    return rgb.astype(np.uint8)


def _to_png_b64(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_rgb(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(_downsample(rgb).astype(np.uint8), mode="RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_scalar(arr: np.ndarray, cmap: str, vmin: float = None, vmax: float = None, alpha: int = 200) -> tuple:
    arr = _downsample(arr.astype(np.float32))
    finite = np.isfinite(arr)
    vmin = float(np.nanpercentile(arr[finite], 2)) if vmin is None else vmin
    vmax = float(np.nanpercentile(arr[finite], 98)) if vmax is None else vmax
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgb = _apply_cmap(np.nan_to_num(norm), COLORMAPS[cmap])
    a = np.where(finite, alpha, 0).astype(np.uint8)[..., None]
    legend = {"type": "gradient", "min": round(vmin, 3), "max": round(vmax, 3),
              "colors": ["rgb(%d,%d,%d)" % s[1] for s in COLORMAPS[cmap]]}
    return _to_png_b64(np.concatenate([rgb, a], axis=-1)), legend


def render_mask(mask: np.ndarray, style: str = "default", alpha: int = 170) -> tuple:
    color = MASK_COLORS.get(style, MASK_COLORS["default"])
    mask = _downsample(mask, categorical=True)
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    legend = {"type": "mask", "color": "rgb(%d,%d,%d)" % color}
    return _to_png_b64(rgba), legend


def render_classmap(classmap: np.ndarray, classes: list, alpha: int = 190) -> tuple:
    classmap = _downsample(classmap, categorical=True)
    rgba = np.zeros(classmap.shape + (4,), dtype=np.uint8)
    present = {}
    for idx, name in enumerate(classes):
        sel = classmap == idx
        if sel.any():
            color = CLASS_COLORS.get(name, (200, 200, 200))
            rgba[sel] = (*color, alpha)
            present[name] = "rgb(%d,%d,%d)" % color
    return _to_png_b64(rgba), {"type": "categorical", "classes": present}
