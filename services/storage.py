"""
services/storage.py — Persist a run's layers to disk and load them back.

Layout: data/runs/<run_id>/<layer_id>.npz (rasters) or .geojson (vectors).
Scenes keep the display RGB (uint8) plus reflectance bands as uint16
(reflectance × 10 000) so exports and re-analysis stay possible.
"""
import json
import os
import re
import shutil
from functools import lru_cache

import numpy as np

import config
import geotools

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,160}$")


def check_id(value: str) -> str:
    if not _ID_RE.match(value or "") or ".." in value:
        raise ValueError("invalid id")
    return value


def run_dir(run_id: str) -> str:
    return os.path.join(config.RUNS_DIR, check_id(run_id))


def save_layer(run_id: str, layer: geotools.Layer) -> None:
    d = run_dir(run_id)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, check_id(layer.id))
    if layer.kind == "vector":
        with open(base + ".geojson", "w") as f:
            json.dump(geotools.vector_geojson(layer), f)
        return
    if layer.kind == "scene":
        bands = {k: np.clip(v * 10000, 0, 65535).astype(np.uint16) for k, v in layer.data["bands"].items()}
        np.savez_compressed(base + ".npz", rgb=layer.data["rgb"], **bands)
    else:
        np.savez_compressed(base + ".npz", data=layer.data)


@lru_cache(maxsize=24)
def load_array(run_id: str, layer_id: str, key: str = "data") -> np.ndarray:
    """One array of a stored raster layer (cached: tiles hit the same layer repeatedly)."""
    path = os.path.join(run_dir(run_id), check_id(layer_id) + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(layer_id)
    with np.load(path) as f:
        return f[key]


def display_array(run_id: str, layer: dict) -> np.ndarray:
    return load_array(run_id, layer["id"], "rgb" if layer["kind"] == "scene" else "data")


def load_bands(run_id: str, layer_id: str) -> dict:
    return {b: load_array(run_id, layer_id, b).astype(np.float32) / 10000 for b in ("B02", "B03", "B04", "B08")}


def load_geojson(run_id: str, layer_id: str) -> dict:
    path = os.path.join(run_dir(run_id), check_id(layer_id) + ".geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(layer_id)
    with open(path) as f:
        return json.load(f)


def delete_run(run_id: str) -> None:
    load_array.cache_clear()
    shutil.rmtree(run_dir(run_id), ignore_errors=True)
