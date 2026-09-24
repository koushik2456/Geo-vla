"""
synthetic.py — Deterministic synthetic Earth-observation data for offline demos.

When Copernicus credentials are missing (or GEO_VLA_OFFLINE=1), the data
tools fall back to this module. It builds a small, internally consistent
"world" for any bounding box:

  * a DEM with hills and a river valley,
  * a river centreline and two roads through the town (synthetic OSM features),
  * Sentinel-2-like surface reflectance (B02, B03, B04, B08) whose land cover
    evolves with the acquisition date: the town grows and forest patches near
    the river are cleared over 2019–2025.

Everything is seeded from the bbox, so the same query always gives the same
answer and change detection between two dates finds real (synthetic) change.
"""
import hashlib

import numpy as np

from geo_utils import parse_date

# Surface reflectance (B02 blue, B03 green, B04 red, B08 NIR) per cover type.
_REFLECTANCE = {
    "water":  (0.060, 0.050, 0.030, 0.020),
    "forest": (0.020, 0.040, 0.020, 0.350),
    "crop":   (0.040, 0.075, 0.050, 0.300),
    "bare":   (0.080, 0.100, 0.130, 0.200),
    "urban":  (0.100, 0.110, 0.120, 0.160),
}


def _seed(bbox) -> int:
    key = ",".join(f"{v:.3f}" for v in bbox).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class _World:
    def __init__(self, bbox):
        self.bbox = bbox
        rng = np.random.default_rng(_seed(bbox))
        self.river_phase = rng.uniform(0, 2 * np.pi)
        self.river_x0 = rng.uniform(0.3, 0.7)
        self.hills = [
            (rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.08, 0.25), rng.uniform(120, 650))
            for _ in range(6)
        ]
        self.town = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75))
        # Forest-clearing patches: (u, v, radius, year cleared)
        self.clearings = [
            (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(0.03, 0.07), rng.uniform(2019.5, 2025.0))
            for _ in range(10)
        ]
        self.base_elev = rng.uniform(20, 400)

    # Normalised coordinates: u = west→east [0,1], v = north→south [0,1]
    def river_u(self, v):
        return self.river_x0 + 0.12 * np.sin(2 * np.pi * 1.1 * v + self.river_phase) + 0.04 * np.sin(2 * np.pi * 3.3 * v)

    def _uv(self, shape):
        rows, cols = shape
        u = (np.arange(cols) + 0.5) / cols
        v = (np.arange(rows) + 0.5) / rows
        return np.meshgrid(u, v)

    def river_distance(self, shape):
        u, v = self._uv(shape)
        return np.abs(u - self.river_u(v))

    def elevation(self, shape):
        u, v = self._uv(shape)
        z = np.full(shape, self.base_elev, dtype=np.float32)
        for hu, hv, r, amp in self.hills:
            z += amp * np.exp(-(((u - hu) ** 2 + (v - hv) ** 2) / (2 * r ** 2)))
        d = self.river_distance(shape)
        z += 150 * d                                   # valley floor rises away from the river
        z -= 60 * np.exp(-((d / 0.05) ** 2))           # incised channel
        z += 3 * np.sin(40 * u) * np.cos(37 * v)       # micro-relief
        return z.astype(np.float32)

    def land_cover(self, shape, year: float):
        """Integer cover map: 0 water, 1 forest, 2 crop, 3 bare, 4 urban."""
        u, v = self._uv(shape)
        z = self.elevation(shape)
        d = self.river_distance(shape)
        cover = np.full(shape, 2, dtype=np.int8)                       # crops by default
        forest = z > np.percentile(z, 45)
        cover[forest] = 1
        for cu, cv, r, yr in self.clearings:
            if year >= yr:
                patch = ((u - cu) ** 2 + (v - cv) ** 2) < r ** 2
                cover[patch & forest] = 3                              # cleared → bare soil
        tu, tv = self.town
        town_r = 0.06 + 0.012 * max(0.0, year - 2018)
        cover[((u - tu) ** 2 + (v - tv) ** 2) < town_r ** 2] = 4
        cover[d < 0.012] = 0
        return cover


def _year(date_str: str) -> float:
    d = parse_date(date_str)
    return d.year + (d.timetuple().tm_yday - 1) / 365.0


def synthetic_dem(bbox, shape) -> np.ndarray:
    return _World(bbox).elevation(shape)


def synthetic_sentinel2(bbox, date_str: str, shape) -> dict:
    """Returns {"B02","B03","B04","B08"} float32 reflectance arrays."""
    world = _World(bbox)
    cover = world.land_cover(shape, _year(date_str))
    date_seed = int(hashlib.sha256(date_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(_seed(bbox) ^ date_seed)
    names = ["water", "forest", "crop", "bare", "urban"]
    bands = {}
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        lut = np.array([_REFLECTANCE[n][i] for n in names], dtype=np.float32)
        arr = lut[cover]
        # Field-to-field variation for crops (seasonal greenness) + sensor noise.
        season = 0.5 + 0.5 * np.sin(2 * np.pi * (_year(date_str) % 1) - np.pi / 2)
        if band == "B08":
            arr = np.where(cover == 2, arr * (0.7 + 0.5 * season), arr)
        arr = arr * (1 + rng.normal(0, 0.04, shape)).astype(np.float32)
        bands[band] = np.clip(arr, 0, 1).astype(np.float32)
    return bands


def synthetic_waterways(bbox, n: int = 200) -> list:
    """River centreline as a list of (lon, lat) coordinates."""
    world = _World(bbox)
    min_lon, min_lat, max_lon, max_lat = bbox
    v = np.linspace(0, 1, n)
    u = world.river_u(v)
    lons = min_lon + u * (max_lon - min_lon)
    lats = max_lat - v * (max_lat - min_lat)
    return list(zip(lons.tolist(), lats.tolist()))


def synthetic_roads(bbox) -> list:
    """Two straight roads crossing at the synthetic town, as lists of (lon, lat)."""
    world = _World(bbox)
    min_lon, min_lat, max_lon, max_lat = bbox
    tu, tv = world.town
    to_ll = lambda u, v: (min_lon + u * (max_lon - min_lon), max_lat - v * (max_lat - min_lat))  # noqa: E731
    return [[to_ll(0.0, tv - 0.1), to_ll(1.0, tv + 0.1)], [to_ll(tu + 0.05, 0.0), to_ll(tu - 0.05, 1.0)]]
