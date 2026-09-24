"""Training studio: real (tiny) training job → registry → promote → hot-reload → upload."""
import io
import time

import pytest
from fastapi.testclient import TestClient

import geotools
from main import app
from services import auth

torch = pytest.importorskip("torch")
client = TestClient(app)
BBOX = [77.50, 12.90, 77.56, 12.96]


@pytest.fixture(scope="module")
def admin():
    auth.create_user("studio_admin", "password123", "admin")
    token = client.post("/auth/login", json={"username": "studio_admin", "password": "password123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_datasets_listed(admin):
    ids = {d["id"] for d in client.get("/training/datasets", headers=admin).json()}
    assert ids == {"eurosat", "levir", "synthetic"}


def test_job_validation(admin):
    bad = client.post("/training/jobs", headers=admin, json={"model": "classifier", "dataset": "levir"})
    assert bad.status_code == 400
    bad = client.post("/training/jobs", headers=admin,
                      json={"model": "change_detector", "dataset": "synthetic", "params": {"epochs": 0}})
    assert bad.status_code == 400


def test_train_promote_and_hot_reload(admin):
    job = client.post("/training/jobs", headers=admin, json={
        "model": "change_detector", "dataset": "synthetic",
        "params": {"epochs": 1, "batch_size": 4, "samples": 8, "pretrained": False}}).json()
    assert job["status"] == "running"
    deadline = time.time() + 300
    while time.time() < deadline:
        job = client.get(f"/training/jobs/{job['id']}", headers=admin).json()
        if job["status"] != "running":
            break
        time.sleep(1)
    assert job["status"] == "done", (job.get("error"), job.get("log"))
    assert job["version"] == "v1" and job["progress"]["fraction"] == 1.0
    assert job["history"] and "val_f1" in job["history"][0]
    assert job["result"]["test"]["f1"] >= 0

    registry = client.get("/models", headers=admin).json()["change_detector"]
    assert registry["versions"][0]["version"] == "v1" and registry["versions"][0]["metrics"]["test_f1"] is not None

    ws = geotools.Workspace(BBOX)
    for d in ("2019-07-01", "2025-07-01"):
        geotools.run_tool(ws, "fetch_sentinel2_scene", {"date": d})
    before = geotools.run_tool(ws, "detect_change", {"scene_id_1": "s2_2019-07-01", "scene_id_2": "s2_2025-07-01"})
    assert "fallback" in before["method"]

    promoted = client.post("/models/change_detector/versions/v1/promote", headers=admin).json()
    assert promoted["active_checkpoint"] == "v1" and promoted["versions"][0]["active"]
    after = geotools.run_tool(ws, "detect_change", {"scene_id_1": "s2_2019-07-01", "scene_id_2": "s2_2025-07-01"})
    assert "Siamese U-Net" in after["method"] and "v1" in after["method"]
    assert client.get("/health").json()["checkpoints"]["change_detector"] == "v1"

    assert client.delete("/models/change_detector/versions/v1", headers=admin).status_code == 400  # active
    client.post("/models/change_detector/deactivate", headers=admin)
    again = geotools.run_tool(ws, "detect_change", {"scene_id_1": "s2_2019-07-01", "scene_id_2": "s2_2025-07-01"})
    assert "fallback" in again["method"]


def test_upload_checkpoint(admin):
    from models.change_detector import SiameseChangeDetector

    buf = io.BytesIO()
    torch.save(SiameseChangeDetector(pretrained=False).state_dict(), buf)
    ok = client.post("/models/change_detector/upload", headers=admin, data={"notes": "from Colab"},
                     files={"file": ("colab.pth", buf.getvalue(), "application/octet-stream")})
    assert ok.status_code == 200, ok.text
    wrong = client.post("/models/classifier/upload", headers=admin,
                        files={"file": ("colab.pth", buf.getvalue(), "application/octet-stream")})
    assert wrong.status_code == 400 and "architecture" in wrong.json()["detail"]
    notpth = client.post("/models/classifier/upload", headers=admin, files={"file": ("x.txt", b"hi", "text/plain")})
    assert notpth.status_code == 400
