"""
services/exports.py — Downloads for GIS software and records.

  GeoTIFF  every raster layer, EPSG:4326, deflate-compressed
           (scenes: 4 bands of surface reflectance × 10 000; masks: 0/1;
            land cover: class codes with names in the metadata)
  GeoJSON  vector layers, plus masks and land-cover maps vectorised to polygons
           with an area_km2 attribute
  ZIP      the PDF report + every layer + a machine-readable summary.json
"""
import io
import json
import zipfile

import numpy as np

import rendering
from models import EUROSAT_CLASSES
from services import reports, storage

MAX_POLYGONS = 5000


def preview_png(run: dict, layer: dict, max_px: int = 512) -> bytes:
    data = storage.display_array(run["id"], layer)
    return rendering.preview_png(layer["kind"], data, layer["style"], max_px)


def geotiff_bytes(run: dict, layer: dict) -> bytes:
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    kind = layer["kind"]
    if kind == "vector":
        raise ValueError("vector layers are exported as GeoJSON")
    tags = {"layer": layer["name"], "source": "Geo-VLA", "run": run["id"]}
    if kind == "scene":
        bands = [storage.load_array(run["id"], layer["id"], b) for b in ("B02", "B03", "B04", "B08")]
        stack, dtype, nodata = np.stack(bands), "uint16", None
        descriptions = ["B02 blue", "B03 green", "B04 red", "B08 NIR"]
        tags["scale"] = "reflectance x 10000"
    else:
        arr = storage.load_array(run["id"], layer["id"])
        if kind == "mask":
            stack, dtype, nodata = arr.astype(np.uint8)[None], "uint8", None
        elif kind == "classmap":
            stack, dtype, nodata = arr.astype(np.uint8)[None], "uint8", 255
            tags["classes"] = json.dumps(dict(enumerate(EUROSAT_CLASSES)))
        else:
            stack, dtype, nodata = arr.astype(np.float32)[None], "float32", np.nan
        descriptions = [layer["name"]]
    count, rows, cols = stack.shape
    profile = {"driver": "GTiff", "width": cols, "height": rows, "count": count, "dtype": dtype,
               "crs": "EPSG:4326", "transform": from_bounds(*layer["bbox"], cols, rows),
               "compress": "deflate", "nodata": nodata}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(stack.astype(dtype))
            dst.update_tags(**tags)
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)
        return mem.read()


def _polygonize(arr: np.ndarray, bbox, mask_value_fn) -> list:
    """Vectorise a categorical raster into (geometry, value) features with areas."""
    import geopandas as gpd
    from rasterio.features import shapes
    from rasterio.transform import from_bounds
    from shapely.geometry import shape

    rows, cols = arr.shape
    transform = from_bounds(*bbox, cols, rows)
    data = arr.astype(np.uint8)
    geoms, values = [], []
    for geom, value in shapes(data, mask=mask_value_fn(data) if mask_value_fn else None, transform=transform):
        geoms.append(shape(geom))
        values.append(int(value))
    if not geoms:
        return []
    series = gpd.GeoSeries(geoms, crs="EPSG:4326")
    areas = series.to_crs(series.estimate_utm_crs()).area / 1e6
    order = np.argsort(-areas.values)[:MAX_POLYGONS]
    tolerance = (bbox[2] - bbox[0]) / cols / 2
    return [(geoms[i].simplify(tolerance, preserve_topology=True), values[i], round(float(areas.iloc[i]), 5))
            for i in order]


def geojson_bytes(run: dict, layer: dict) -> bytes:
    kind = layer["kind"]
    if kind == "vector":
        fc = storage.load_geojson(run["id"], layer["id"])
    elif kind == "mask":
        arr = storage.load_array(run["id"], layer["id"])
        feats = _polygonize(arr, layer["bbox"], lambda d: d == 1)
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": g.__geo_interface__, "properties": {"layer": layer["name"], "area_km2": a}}
            for g, _, a in feats]}
    elif kind == "classmap":
        arr = storage.load_array(run["id"], layer["id"])
        feats = _polygonize(arr, layer["bbox"], None)
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": g.__geo_interface__,
             "properties": {"class": EUROSAT_CLASSES[v] if v < len(EUROSAT_CLASSES) else v, "area_km2": a}}
            for g, v, a in feats]}
    else:
        raise ValueError("continuous rasters (imagery, NDVI, slope, elevation) are exported as GeoTIFF")
    fc["properties"] = {"layer": layer["name"], "run": run["id"], "crs": "EPSG:4326"}
    return json.dumps(fc).encode()


def bundle_zip(run: dict, author=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.pdf", reports.report_pdf(run, author=author))
        for layer in run["layers"]:
            if layer["kind"] != "vector":
                z.writestr(f"rasters/{layer['id']}.tif", geotiff_bytes(run, layer))
            if layer["kind"] in ("vector", "mask"):
                z.writestr(f"vectors/{layer['id']}.geojson", geojson_bytes(run, layer))
        summary = {k: run[k] for k in ("id", "title", "instruction", "workflow", "params", "bbox", "place",
                                       "answer", "planner", "model", "data_mode", "insights", "trace",
                                       "created_at", "finished_at")}
        z.writestr("summary.json", json.dumps(summary, indent=2, default=str))
        z.writestr("README.txt", (
            "Geo-VLA export\n\n"
            "report.pdf        official summary report\n"
            "rasters/*.tif     GeoTIFF layers (EPSG:4326); open in QGIS / ArcGIS\n"
            "vectors/*.geojson vector layers and vectorised masks (area_km2 per polygon)\n"
            "summary.json      answer, key figures, metrics and the full reasoning trace\n"))
    return buf.getvalue()


report_pdf = reports.report_pdf
