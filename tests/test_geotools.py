import numpy as np
import pytest

import geotools as g
from geo_utils import bbox_area_km2, validate_bbox

BBOX = [77.50, 12.90, 77.56, 12.96]


@pytest.fixture
def ws():
    return g.Workspace(BBOX)


def run(ws, tool_name, **kw):
    return g.run_tool(ws, tool_name, kw)


# -- pure raster math ---------------------------------------------------------

def test_ndvi_range_and_values():
    nir, red = np.array([[0.5, 0.0]]), np.array([[0.1, 0.0]])
    ndvi = g.compute_ndvi(nir, red)
    assert ndvi[0, 0] == pytest.approx(0.4 / 0.6)
    assert ndvi[0, 1] == 0.0  # zero denominator is guarded


def test_slope_of_inclined_plane():
    # Rises 30 m per 30 m pixel in x → 45°.
    dem = np.tile(np.arange(10, dtype=np.float32) * 30, (10, 1))
    assert np.allclose(g.compute_slope(dem, 30.0), 45.0, atol=1e-3)
    # Anisotropic pixels: 60 m in x → 26.57°.
    assert np.allclose(g.compute_slope(dem, (60.0, 30.0)), np.degrees(np.arctan(0.5)), atol=1e-3)


def test_flood_extent_bathtub():
    dem = np.array([[1.0, 5.0], [10.0, 2.0]])
    assert g.flood_extent(dem, 2.0).tolist() == [[True, False], [False, True]]


def test_dem_gap_filling():
    dem = np.full((6, 6), 100.0, dtype=np.float32)
    dem[:, 3:] = 200.0
    dem[-1, :] = g._DEM_NODATA          # thin seam → nearest value
    filled = g._fill_dem_gaps(dem)
    assert filled[-1, 0] == 100.0 and filled[-1, 5] == 200.0
    ocean = np.full((20, 20), g._DEM_NODATA, dtype=np.float32)
    ocean[:, :2] = 50.0                 # large gap (missing tile) → sea level
    assert g._fill_dem_gaps(ocean)[0, 19] == 0.0


def test_bbox_validation():
    assert validate_bbox(BBOX) == BBOX
    with pytest.raises(ValueError):
        validate_bbox([1, 1, 0, 0])
    with pytest.raises(ValueError):
        validate_bbox([0, 0, 5, 5])  # too large
    assert bbox_area_km2(BBOX) == pytest.approx(6.5 * 6.6, rel=0.05)


# -- tools over synthetic data ------------------------------------------------

def test_fetch_scene_is_deterministic(ws):
    out = run(ws, "fetch_sentinel2_scene", date="2022-06-01")
    assert out["scene_id"] == "s2_2022-06-01" and "synthetic" in out["source"]
    a = ws.get("s2_2022-06-01").data["bands"]["B08"].copy()
    run(ws, "fetch_sentinel2_scene", date="2022-06-01")
    assert np.array_equal(a, ws.get("s2_2022-06-01").data["bands"]["B08"])


def test_deforestation_chain_finds_forest_loss(ws):
    for d in ("2019-07-01", "2025-07-01"):
        run(ws, "fetch_sentinel2_scene", date=d)
        run(ws, "classify_scene", scene_id=f"s2_{d}")
        run(ws, "class_mask", layer_id=f"landcover_{d}", class_name="Forest")
    loss = run(ws, "overlay_layers", layer_a="forest_2019-07-01", layer_b="forest_2025-07-01",
               operation="difference", name="forest_loss")
    assert loss["layer_id"] == "forest_loss"
    assert loss["area_km2"] > 0  # the synthetic world clears forest patches over time
    change = run(ws, "detect_change", scene_id_1="s2_2019-07-01", scene_id_2="s2_2025-07-01")
    assert 0 < change["fraction"] < 0.5
    assert "fallback" in change["method"]


def test_buffer_overlay_and_zonal_stats(ws):
    run(ws, "fetch_sentinel2_scene", date="2024-07-01")
    run(ws, "fetch_osm_features", feature_type="water")
    buf = run(ws, "buffer_features", layer_id="osm_water", distance_m=500)
    wide = run(ws, "buffer_features", layer_id="osm_water", distance_m=1500)
    assert 0 < buf["fraction"] < wide["fraction"] < 1
    run(ws, "fetch_dem")
    run(ws, "compute_slope")
    stats = run(ws, "zonal_stats", value_layer="slope", zone_layer=buf["layer_id"])
    assert stats["zone_pixels"] > 0 and stats["min"] <= stats["mean"] <= stats["max"]
    run(ws, "classify_scene")
    lc = run(ws, "zonal_stats", value_layer="landcover_2024-07-01", zone_layer=buf["layer_id"])
    assert sum(lc["class_fractions"].values()) == pytest.approx(1.0, abs=0.05)


def test_flood_modes(ws):
    dem = run(ws, "fetch_dem")
    rel = run(ws, "flood_extent", water_level_m=10)
    assert rel["water_level_m_asl"] == pytest.approx(dem["min_m"] + 10, abs=0.1)
    absolute = run(ws, "flood_extent", water_level_m=dem["max_m"] + 1, mode="absolute")
    assert absolute["fraction"] == 1.0


def test_threshold_and_mask_buffer(ws):
    run(ws, "fetch_dem")
    run(ws, "compute_slope")
    steep = run(ws, "threshold_layer", layer_id="slope", operator="gt", value=5)
    assert steep["layer_id"] == "slope_gt_5"
    grown = run(ws, "buffer_features", layer_id="slope_gt_5", distance_m=200)
    assert grown["fraction"] >= steep["fraction"]


def test_errors_are_tool_errors(ws):
    with pytest.raises(g.ToolError, match="unknown layer"):
        run(ws, "compute_ndvi", scene_id="s2_1999-01-01")
    with pytest.raises(g.ToolError, match="no scene layer"):
        run(ws, "classify_scene")
    with pytest.raises(g.ToolError, match="unknown tool"):
        g.run_tool(ws, "launch_rocket", {})
    with pytest.raises(g.ToolError):
        g.Workspace().bbox()


def test_render_every_layer_kind(ws):
    run(ws, "fetch_sentinel2_scene", date="2024-07-01")
    run(ws, "compute_ndvi")
    run(ws, "classify_scene")
    run(ws, "fetch_osm_features", feature_type="roads")
    run(ws, "buffer_features", layer_id="osm_roads", distance_m=300)
    run(ws, "fetch_dem")
    for layer in ws.layers.values():
        out = g.render_layer(layer)
        assert out["bbox"] == BBOX
        assert ("image" in out and out["image"].startswith("data:image/")) or out["kind"] == "vector"


def test_tool_schemas_are_well_formed():
    schemas = g.tool_schemas()
    assert len(schemas) == 15
    for s in schemas:
        assert s["input_schema"]["type"] == "object"
        assert set(s["input_schema"]["required"]) <= set(s["input_schema"]["properties"])


def test_live_mode_without_imagery_credentials_refuses_instead_of_faking(monkeypatch):
    import config
    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "_DATA_MODE", "live")
    monkeypatch.setattr(config, "COPERNICUS_CLIENT_ID", "")
    monkeypatch.setattr(config, "EE_PROJECT", "")
    assert config.data_mode() == "live" and config.imagery_status() == "missing"
    ws = g.Workspace([77.60, 12.90, 77.64, 12.94])
    with pytest.raises(g.ToolError, match="COPERNICUS_CLIENT_ID"):
        g.fetch_sentinel2_scene(ws, "2024-01-01")
    assert "s2_2024-01-01" not in ws.layers


def test_overpass_tries_mirrors_then_caches(monkeypatch, tmp_path):
    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [{"geometry": [{"lon": 77.6, "lat": 12.9}, {"lon": 77.61, "lat": 12.91}], "tags": {}}]}

    def post(url, **_):
        calls.append(url)
        return Resp(504 if "overpass-api.de" in url else 200)

    monkeypatch.setattr(g, "OSM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(g, "OVERPASS_URLS", ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"])
    monkeypatch.setattr(g.requests, "post", post)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    assert len(g.fetch_osm_geometries([77.6, 12.9, 77.62, 12.92], "roads")) == 1
    assert calls == ["https://overpass-api.de/api/interpreter"] * 2 + ["https://overpass.kumi.systems/api/interpreter"]
    assert len(g.fetch_osm_geometries([77.6, 12.9, 77.62, 12.92], "roads")) == 1 and len(calls) == 3   # cached
    monkeypatch.setattr(g.requests, "post", lambda url, **_: Resp(504))
    with pytest.raises(g.ToolError, match="Overpass"):
        g.fetch_osm_geometries([77.0, 12.0, 77.02, 12.02], "water")
