"""
services/tiles.py — Full-resolution XYZ (Web Mercator) map tiles for run layers.

Layers are stored on a regular lon/lat grid. For each 256×256 tile we compute
the lon/lat of every output pixel, map it to fractional row/column in the
layer, and resample with cv2.remap (nearest for masks and classes, bilinear
for imagery and continuous values). Detail is therefore limited only by the
source data (10 m Sentinel-2, 30 m DEM) at any zoom level.
"""
import math

import cv2
import numpy as np

import rendering

TILE = 256
_EMPTY = None


def empty_tile() -> bytes:
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = rendering.png_bytes(np.zeros((TILE, TILE, 4), dtype=np.uint8))
    return _EMPTY


def tile_bounds(z: int, x: int, y: int) -> tuple:
    n = 2 ** z
    lon = lambda px: px / n * 360.0 - 180.0  # noqa: E731
    lat = lambda py: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * py / n))))  # noqa: E731
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def render_tile(data: np.ndarray, kind: str, style: dict, bbox, z: int, x: int, y: int) -> bytes:
    if not (0 <= z <= 22 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise ValueError("tile out of range")
    if not _intersects(tile_bounds(z, x, y), bbox):
        return empty_tile()

    n = TILE * 2 ** z
    px = x * TILE + np.arange(TILE) + 0.5
    py = y * TILE + np.arange(TILE) + 0.5
    lons = px / n * 360.0 - 180.0
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * py / n))))
    min_lon, min_lat, max_lon, max_lat = bbox
    rows, cols = data.shape[:2]
    col = ((lons - min_lon) / (max_lon - min_lon) * cols - 0.5).astype(np.float32)
    row = ((max_lat - lats) / (max_lat - min_lat) * rows - 0.5).astype(np.float32)
    map_x, map_y = np.meshgrid(col, row)
    valid = (map_x >= -0.5) & (map_x <= cols - 0.5) & (map_y >= -0.5) & (map_y <= rows - 0.5)

    categorical = kind in ("mask", "classmap")
    src = data
    if src.dtype == bool:
        src = src.astype(np.uint8)
    elif src.dtype == np.int8:
        src = src.astype(np.int16)
    interp = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    sampled = cv2.remap(src, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)
    if kind == "mask":
        sampled = sampled.astype(bool)
    return rendering.png_bytes(rendering.colorize(kind, sampled, style, valid))
