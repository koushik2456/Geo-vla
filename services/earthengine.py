"""
services/earthengine.py — Google Earth Engine as an imagery source.

Earth Engine hosts the full Sentinel-2 archive (COPERNICUS/S2_SR_HARMONIZED) and
Copernicus DEM, and composites them server-side, so we ask for exactly the grid
we need and receive a NumPy array (ee.data.computePixels). Pixels are cloud
masked with the Scene Classification Layer and median-composited over the
search window, which gives cleaner mosaics than a single least-cloud scene.

Setup (either):
  * personal account:  pip install earthengine-api && earthengine authenticate
                       then EE_PROJECT=<your registered cloud project>
  * service account:   EE_PROJECT, EE_SERVICE_ACCOUNT=<email>, EE_PRIVATE_KEY_FILE=<key.json>
"""
import logging
import threading
from datetime import timedelta

import numpy as np

import config
from geo_utils import date_window, parse_date

log = logging.getLogger("geo-vla.earthengine")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEM_COLLECTION = "COPERNICUS/DEM/GLO30"
# Scene Classification Layer classes to drop: cloud shadow, cloud (medium/high), cirrus, saturated
_SCL_MASKED = (1, 3, 8, 9, 10)
_BANDS = {"B02": "B2", "B03": "B3", "B04": "B4", "B08": "B8"}

_lock = threading.Lock()
_state = {"ee": None}


class EarthEngineError(RuntimeError):
    pass


def _ee():
    """Initialised `ee` module (once per process)."""
    with _lock:
        if _state["ee"] is not None:
            return _state["ee"]
        try:
            import ee
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise EarthEngineError("earthengine-api is not installed (pip install earthengine-api)") from exc
        try:
            if config.EE_SERVICE_ACCOUNT and config.EE_PRIVATE_KEY_FILE:
                creds = ee.ServiceAccountCredentials(config.EE_SERVICE_ACCOUNT, config.EE_PRIVATE_KEY_FILE)
                ee.Initialize(creds, project=config.EE_PROJECT)
            else:
                ee.Initialize(project=config.EE_PROJECT)
        except Exception as exc:
            raise EarthEngineError(
                f"Earth Engine initialisation failed ({exc}). Run `earthengine authenticate` or set "
                "EE_SERVICE_ACCOUNT + EE_PRIVATE_KEY_FILE, and check EE_PROJECT is registered for Earth Engine."
            ) from exc
        _state["ee"] = ee
        return ee


def reset():
    """Forget the initialised module (tests)."""
    with _lock:
        _state["ee"] = None


def _grid(bbox, shape) -> dict:
    w, s, e, n = bbox
    rows, cols = shape
    return {
        "dimensions": {"width": int(cols), "height": int(rows)},
        "affineTransform": {"scaleX": (e - w) / cols, "shearX": 0, "translateX": w,
                            "shearY": 0, "scaleY": -(n - s) / rows, "translateY": n},
        "crsCode": "EPSG:4326",
    }


def _mask_clouds(img):
    scl = img.select("SCL")
    keep = scl.neq(_SCL_MASKED[0])
    for cls in _SCL_MASKED[1:]:
        keep = keep.And(scl.neq(cls))
    return img.updateMask(keep)


def sentinel2_bands(bbox, date: str, window_days: int, max_cloud_pct: int, shape) -> dict:
    """Cloud-masked median Sentinel-2 L2A composite → {"B02","B03","B04","B08"} reflectance (0–1)."""
    ee = _ee()
    start, end = date_window(date, window_days)
    end_exclusive = (parse_date(end) + timedelta(days=1)).isoformat()
    region = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(region)
           .filterDate(start, end_exclusive)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct)))
    count = col.size().getInfo()
    if not count:
        raise EarthEngineError(f"no Sentinel-2 scene with <= {max_cloud_pct}% cloud between {start} and {end}")
    image = (col.map(_mask_clouds).median()
             .select(list(_BANDS.values())).divide(10000).unmask(-1).toFloat())
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": _grid(bbox, shape)})
    bands = {ours: np.asarray(arr[theirs], dtype=np.float32) for ours, theirs in _BANDS.items()}
    invalid = bands["B04"] < 0
    if invalid.mean() > 0.9:
        raise EarthEngineError(f"clouds cover this area in every scene between {start} and {end}; widen the window")
    if invalid.any():  # fill remaining cloud holes from the band median so indices stay finite
        for k, v in bands.items():
            v[invalid] = float(np.median(v[~invalid]))
    log.info("Earth Engine S2 composite of %d scenes, %s..%s, %s px", count, start, end, shape)
    bands["_scenes"] = count
    return bands


def dem(bbox, shape) -> np.ndarray:
    """Copernicus DEM GLO-30 elevation (m) on the requested grid."""
    ee = _ee()
    image = ee.ImageCollection(DEM_COLLECTION).select("DEM").mosaic().unmask(0).toFloat()
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": _grid(bbox, shape)})
    return np.asarray(arr["DEM"], dtype=np.float32)
