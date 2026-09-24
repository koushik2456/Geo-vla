from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
BBOX = [77.50, 12.90, 77.56, 12.96]


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["planner"] == "offline" and body["data_mode"] == "synthetic"


def test_tools_endpoint():
    names = {t["name"] for t in client.get("/tools").json()}
    assert {"fetch_sentinel2_scene", "detect_change", "buffer_features", "overlay_layers"} <= names


def test_query_roundtrip():
    resp = client.post("/query", json={"instruction": "Show NDVI and land cover", "bbox": BBOX})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] and body["layers"] and body["trace"]


def test_query_validation():
    assert client.post("/query", json={"instruction": "   ", "bbox": BBOX}).status_code == 400
    assert client.post("/query", json={"instruction": "x", "bbox": [1, 2, 0, 0]}).status_code == 422
