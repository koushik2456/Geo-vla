"""geo_utils.py — Small geometry helpers shared by the tools (bbox math, grids)."""
import math
from datetime import date, timedelta

import numpy as np

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


def validate_bbox(bbox) -> list:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError(f"invalid bbox {bbox}: expected min < max within lon/lat range")
    if (max_lon - min_lon) > 2.0 or (max_lat - min_lat) > 2.0:
        raise ValueError("bbox too large (max 2 degrees per side) — zoom in on a smaller area")
    return [min_lon, min_lat, max_lon, max_lat]


def bbox_size_m(bbox) -> tuple:
    """(width_m, height_m) of a lon/lat bbox, using the mid-latitude scale."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    width = (max_lon - min_lon) * M_PER_DEG_LON_EQ * math.cos(mid_lat)
    height = (max_lat - min_lat) * M_PER_DEG_LAT
    return width, height


def bbox_area_km2(bbox) -> float:
    w, h = bbox_size_m(bbox)
    return w * h / 1e6


def grid_shape(bbox, resolution_m: float, min_px: int = 64, max_px: int = 768) -> tuple:
    """(rows, cols) for a raster covering bbox at ~resolution_m, clamped to sane sizes."""
    w, h = bbox_size_m(bbox)
    cols = int(np.clip(round(w / resolution_m), min_px, max_px))
    rows = int(np.clip(round(h / resolution_m), min_px, max_px))
    return rows, cols


def pixel_size_m(bbox, shape) -> tuple:
    """(dx_m, dy_m) for a raster of `shape` covering bbox."""
    w, h = bbox_size_m(bbox)
    return w / shape[1], h / shape[0]


def same_bbox(a, b, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def parse_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def date_window(value: str, window_days: int) -> tuple:
    d = parse_date(value)
    return (d - timedelta(days=window_days)).isoformat(), (d + timedelta(days=window_days)).isoformat()


def lonlat_grid(bbox, shape) -> tuple:
    """Pixel-centre lon/lat arrays (row 0 = north)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    rows, cols = shape
    lons = min_lon + (np.arange(cols) + 0.5) / cols * (max_lon - min_lon)
    lats = max_lat - (np.arange(rows) + 0.5) / rows * (max_lat - min_lat)
    return np.meshgrid(lons, lats)
