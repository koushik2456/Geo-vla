"""Globe explorer (place details, click-to-classify), Google geocoding and Earth Engine imagery."""
import json
import os
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
import geotools
from main import app
from services import earthengine, geocode

client = TestClient(app)
LON, LAT = 77.67, 12.93  # Bellandur Lake, Bengaluru


def test_client_config_without_keys():
    cfg = client.get("/client-config").json()
    assert cfg["google_maps_key"] is None and cfg["geocoder"] == "gazetteer" and cfg["imagery_source"] == "synthetic"


def test_point_details_offline():
    d = client.get(f"/explore/point?lon={LON}&lat={LAT}").json()
    assert d["place"]["name"].startswith("Bellandur") or "Bellandur" in d["place"]["formatted_address"]
    assert d["utm"] == {"zone": "43N", "epsg": 32643}
    assert d["elevation"]["m"] is not None
    assert client.get("/explore/point?lon=10&lat=89").status_code == 400


def test_classify_point_fallback_without_checkpoint():
    r = client.post("/explore/classify", json={"lon": LON, "lat": LAT, "date": "2025-01-15"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert not d["neural"] and "network" not in d and "spectral rules" in d["method"]
    assert len(d["context"]["grid"]) == 9 and d["patch"]["size_m"] == 640
    assert len(d["prediction"]["top5"]) == 5 and sum(t["p"] for t in d["prediction"]["top5"]) <= 1.0001
    sig = d["spectral"]["signature"]
    assert [s["nm"] for s in sig] == [490, 560, 665, 842] and -1 <= d["spectral"]["ndvi"] <= 1
    w, s, e, n = d["context"]["bbox"]
    assert w < LON < e and s < LAT < n
    assert client.post("/explore/classify", json={"lon": LON, "lat": 95}).status_code == 422


@pytest.fixture
def classifier_checkpoint():
    torch = pytest.importorskip("torch")
    from models.classifier import SceneClassifier

    os.makedirs(config.MODEL_CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_CHECKPOINT_DIR, geotools.CHECKPOINT_FILES["classifier"])
    torch.manual_seed(0)
    torch.save(SceneClassifier(pretrained=False).state_dict(), path)
    with open(path + ".version.json", "w") as f:
        json.dump({"version": "v7", "input_size": 64}, f)
    geotools.reload_models()
    yield
    for p in (path, path + ".version.json"):
        os.remove(p)
    geotools.reload_models()


def test_classify_point_explains_the_network(classifier_checkpoint):
    d = client.post("/explore/classify", json={"lon": LON, "lat": LAT}).json()
    assert d["neural"] and "v7" in d["method"]
    net = d["network"]
    assert net["input"]["tensor_shape"] == [1, 3, 64, 64]
    assert [s["stage"] for s in net["stages"]] == ["stem", "layer1", "layer2", "layer3", "layer4"]
    assert net["stages"][-1]["shape"][0] == 2048 and net["gradcam"].startswith("data:image/png;base64,")
    assert net["embedding"]["dim"] == 2048 and len(net["logits"]) == 10
    top = d["prediction"]["top5"]
    assert top == sorted(top, key=lambda t: -t["p"]) and d["prediction"]["class"] == top[0]["class"]


# -- Google geocoding ------------------------------------------------------------------------

class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


GOOGLE_REVERSE = {"status": "OK", "results": [{
    "formatted_address": "Bellandur, Bengaluru, Karnataka 560103, India",
    "types": ["sublocality"], "plus_code": {"global_code": "7J4VWMM9+2C"},
    "address_components": [
        {"long_name": "Bellandur", "types": ["sublocality_level_1", "sublocality"]},
        {"long_name": "Bengaluru", "types": ["locality"]},
        {"long_name": "Bangalore Urban", "types": ["administrative_area_level_2"]},
        {"long_name": "Karnataka", "types": ["administrative_area_level_1"]},
        {"long_name": "India", "types": ["country"]},
        {"long_name": "560103", "types": ["postal_code"]},
    ]}]}


def test_google_reverse_and_search(monkeypatch):
    calls = []

    def fake_get(url, params=None, **_):
        calls.append((url, params))
        if "latlng" in params:
            return _Resp(GOOGLE_REVERSE)
        return _Resp({"status": "OK", "results": [{
            **GOOGLE_REVERSE["results"][0],
            "geometry": {"location": {"lat": 12.93, "lng": 77.67},
                         "viewport": {"southwest": {"lat": 12.9, "lng": 77.64}, "northeast": {"lat": 12.96, "lng": 77.7}}}}]})

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "GOOGLE_GEOCODING_KEY", "test-key")
    monkeypatch.setattr(geocode.requests, "get", fake_get)
    r = geocode.reverse(LON, LAT)
    assert r["source"] == "Google Geocoding" and r["plus_code"] == "7J4VWMM9+2C"
    assert r["components"] == {"suburb": "Bellandur", "locality": "Bengaluru", "district": "Bangalore Urban",
                               "state": "Karnataka", "country": "India", "postcode": "560103"}
    assert calls[0][1]["key"] == "test-key" and calls[0][1]["latlng"] == f"{LAT},{LON}"
    hits = geocode.search("Bellandur")
    assert hits[0]["source"] == "Google Geocoding" and hits[0]["bbox"] == [77.64, 12.9, 77.7, 12.96]


def test_reverse_falls_back_to_nominatim_then_gazetteer(monkeypatch):
    def fake_get(url, params=None, **_):
        if url == config.GOOGLE_GEOCODE_URL:
            return _Resp({"status": "REQUEST_DENIED", "error_message": "API key invalid"})
        return _Resp({"display_name": "Bellandur, Bengaluru, Karnataka, India", "name": "Bellandur",
                      "category": "place", "type": "suburb",
                      "address": {"suburb": "Bellandur", "city": "Bengaluru", "state": "Karnataka", "country": "India"}})

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "GOOGLE_GEOCODING_KEY", "bad")
    monkeypatch.setattr(geocode.requests, "get", fake_get)
    r = geocode.reverse(LON, LAT)
    assert r["source"] == "OpenStreetMap Nominatim" and r["components"]["locality"] == "Bengaluru"

    def down(*_, **__):
        raise ConnectionError("offline")

    monkeypatch.setattr(geocode.requests, "get", down)
    assert geocode.reverse(LON, LAT)["source"].startswith("bundled gazetteer")


# -- Google Earth Engine -------------------------------------------------------------------------

def _fake_ee(scene_count=3, cloudy=False):
    """Minimal stand-in for the earthengine-api surface we use."""
    log = {"filters": [], "requests": []}

    class Chain:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, attr):
            def call(*args, **kwargs):
                log["filters"].append((attr, args))
                return self
            return call

        def size(self):
            return types.SimpleNamespace(getInfo=lambda: scene_count)

    def compute_pixels(req):
        log["requests"].append(req)
        dims = req["grid"]["dimensions"]
        names = ["B2", "B3", "B4", "B8"] if req["expression"].name == "s2" else ["DEM"]
        arr = np.zeros((dims["height"], dims["width"]), dtype=[(n, "<f4") for n in names])
        for i, n in enumerate(names):
            arr[n] = -1 if cloudy else 0.05 * (i + 1)
        return arr

    ee = types.ModuleType("ee")
    ee.Initialize = lambda *a, **k: log.setdefault("init", (a, k))
    ee.ServiceAccountCredentials = lambda *a: ("creds", a)
    ee.Geometry = types.SimpleNamespace(Rectangle=lambda *a: ("rect", a))
    ee.Filter = types.SimpleNamespace(lte=lambda *a: ("lte", a))
    ee.ImageCollection = lambda name: Chain("s2" if "S2" in name else "dem")
    ee.data = types.SimpleNamespace(computePixels=compute_pixels)
    return ee, log


@pytest.fixture
def earth_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IMAGERY_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "EE_PROJECT", "my-ee-project")
    monkeypatch.setattr(config, "data_live", lambda: True)
    earthengine.reset()
    yield monkeypatch
    earthengine.reset()


def test_imagery_source_selection(monkeypatch):
    monkeypatch.setattr(config, "EE_PROJECT", "p")
    monkeypatch.setattr(config, "_IMAGERY_SOURCE", "auto")
    monkeypatch.setattr(config, "COPERNICUS_CLIENT_ID", "")
    assert config.imagery_source() == "earthengine"
    monkeypatch.setattr(config, "COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setattr(config, "COPERNICUS_CLIENT_SECRET", "secret")
    assert config.imagery_source() == "copernicus"
    monkeypatch.setattr(config, "_IMAGERY_SOURCE", "earthengine")
    assert config.imagery_source() == "earthengine"


def test_sentinel2_from_earth_engine(earth_engine):
    ee, log = _fake_ee()
    earth_engine.setitem(sys.modules, "ee", ee)
    earth_engine.setattr(config, "imagery_source", lambda: "earthengine")
    ws = geotools.Workspace([77.60, 12.90, 77.64, 12.94])
    out = geotools.fetch_sentinel2_scene(ws, "2024-03-01", window_days=15, max_cloud_pct=25)
    assert "Google Earth Engine" in out["source"] and "3 scenes" in out["source"]
    bands = ws.get("s2_2024-03-01").data["bands"]
    assert set(bands) == {"B02", "B03", "B04", "B08"} and bands["B04"].shape == tuple(out["shape"])
    assert np.allclose(bands["B08"], 0.2)
    grid = log["requests"][0]["grid"]
    assert grid["crsCode"] == "EPSG:4326" and grid["affineTransform"]["translateX"] == 77.60
    assert grid["affineTransform"]["scaleY"] < 0 and ("filterDate", ("2024-02-15", "2024-03-17")) in log["filters"]
    assert log["init"][1] == {"project": "my-ee-project"}
    # second fetch comes from the cache
    geotools.fetch_sentinel2_scene(geotools.Workspace([77.60, 12.90, 77.64, 12.94]), "2024-03-01", window_days=15, max_cloud_pct=25)
    assert len(log["requests"]) == 1


def test_earth_engine_errors_become_tool_errors(earth_engine):
    earth_engine.setattr(config, "imagery_source", lambda: "earthengine")
    for kwargs in ({"scene_count": 0}, {"cloudy": True}):
        earthengine.reset()
        ee, _ = _fake_ee(**kwargs)
        earth_engine.setitem(sys.modules, "ee", ee)
        with pytest.raises(geotools.ToolError):
            geotools.fetch_sentinel2_scene(geotools.Workspace([77.60, 12.90, 77.64, 12.94]), "2024-04-01")
