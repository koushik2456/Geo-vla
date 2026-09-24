"""Product API: accounts, background runs, tiles, exports, sharing, projects, workflows, monitoring."""
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from main import app
from services import auth, geocode, monitoring, runs, workflows

client = TestClient(app)
BBOX = [77.50, 12.90, 77.56, 12.96]


def _login(username, role="official"):
    try:
        auth.create_user(username, "password123", role)
    except ValueError:
        pass  # already created by an earlier test
    token = client.post("/auth/login", json={"username": username, "password": "password123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _center_tile(bbox, z=15):
    import math
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return {"z": z, "x": x, "y": y}


def _wait(run_id, headers=None, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}", headers=headers or {}).json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError("run did not finish")


# -- accounts ----------------------------------------------------------------------------------

def test_password_hashing_and_sessions():
    stored = auth.hash_password("correct horse")
    assert auth.verify_password("correct horse", stored) and not auth.verify_password("wrong", stored)
    headers = _login("asha")
    me = client.get("/auth/me", headers=headers).json()["user"]
    assert me["username"] == "asha" and me["role"] == "official" and "password_hash" not in me
    client.post("/auth/logout", headers=headers)
    assert client.get("/auth/me", headers=headers).json()["user"] is None


def test_login_rejects_bad_password_and_signup_validates():
    assert client.post("/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401
    assert client.post("/auth/signup", json={"username": "ab", "password": "password123"}).status_code == 400
    ok = client.post("/auth/signup", json={"username": "citizen1", "password": "password123"})
    assert ok.status_code == 200 and ok.json()["user"]["role"] == "public"


def test_roles_are_enforced():
    public = _login("pub1", "public")
    official = _login("off1", "official")
    admin = _login("adm1", "admin")
    assert client.get("/admin/users", headers=official).status_code == 403
    assert client.get("/admin/users", headers=admin).status_code == 200
    assert client.get("/monitors", headers=public).status_code == 403
    assert client.get("/monitors", headers=official).status_code == 200
    assert client.get("/training/jobs", headers=official).status_code == 403
    uid = next(u["id"] for u in client.get("/admin/users", headers=admin).json() if u["username"] == "pub1")
    assert client.patch(f"/admin/users/{uid}", json={"role": "official"}, headers=admin).json()["role"] == "official"


# -- runs ---------------------------------------------------------------------------------------

def test_background_run_with_tiles_and_exports():
    headers = _login("ravi")
    resp = client.post("/runs", json={"bbox": BBOX, "workflow": "deforestation",
                                      "params": {"start_date": "2019-07-01", "end_date": "2025-07-01"}},
                       headers=headers)
    assert resp.status_code == 202
    run = _wait(resp.json()["id"], headers)
    assert run["status"] == "done", run["error"]
    assert run["insights"]["key_figures"] and run["insights"]["metrics"]["forest_loss_km2"] > 0
    assert run["insights"]["focus_layer"] == "forest_loss"
    assert any(c["type"] == "bar" for c in run["insights"]["charts"])
    assert [t["tool"] for t in run["trace"] if t["type"] == "tool_call"][:2] == ["fetch_sentinel2_scene"] * 2

    loss = next(l for l in run["layers"] if l["id"] == "forest_loss")
    assert loss["stats"]["area_km2"] == run["insights"]["metrics"]["forest_loss_km2"] and "tiles" in loss
    scene = run["layers"][0]
    tile = client.get(scene["tiles"].format(**_center_tile(BBOX)), headers=headers)
    empty = client.get(scene["tiles"].format(z=3, x=0, y=0), headers=headers)  # outside the AOI
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    assert empty.status_code == 200 and len(tile.content) > 5 * len(empty.content)

    base = f"/runs/{run['id']}"
    tif = client.get(f"{base}/layers/forest_loss.tif", headers=headers)
    assert tif.status_code == 200 and tif.content[:2] in (b"II", b"MM")
    gj = client.get(f"{base}/layers/forest_loss.geojson", headers=headers).json()
    assert gj["features"] and "area_km2" in gj["features"][0]["properties"]
    assert client.get(f"{base}/layers/slope.geojson", headers=headers).status_code == 400
    pdf = client.get(f"{base}/report.pdf", headers=headers)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    zf = zipfile.ZipFile(io.BytesIO(client.get(f"{base}/export.zip", headers=headers).content))
    assert {"report.pdf", "summary.json", "rasters/forest_loss.tif", "vectors/forest_loss.geojson"} <= set(zf.namelist())


def test_private_runs_and_share_links():
    owner, other = _login("owner1"), _login("other1")
    run_id = client.post("/runs", json={"bbox": BBOX, "workflow": "crop_health"}, headers=owner).json()["id"]
    run = _wait(run_id, owner)
    assert run["can_edit"]
    assert client.get(f"/runs/{run_id}", headers=other).status_code == 404
    assert client.get(f"/runs/{run_id}").status_code == 404
    token = client.post(f"/runs/{run_id}/share", headers=owner).json()["share_token"]
    shared = client.get(f"/share/{token}").json()
    assert shared["id"] == run_id and not shared["can_edit"] and shared["share_token"] is None
    layer = run["layers"][0]
    assert client.get(layer["tiles"].format(**_center_tile(BBOX)) + f"?share={token}").status_code == 200
    client.delete(f"/runs/{run_id}/share", headers=owner)
    assert client.get(f"/share/{token}").status_code == 404
    assert client.delete(f"/runs/{run_id}", headers=other).status_code == 404
    assert client.delete(f"/runs/{run_id}", headers=owner).status_code == 200


def test_anonymous_free_text_run_and_validation():
    run_id = client.post("/runs", json={"bbox": BBOX, "instruction": "NDVI within 500 m of roads"}).json()["id"]
    run = _wait(run_id)
    assert run["status"] == "done" and run["planner"] == "offline"
    assert "synthetic" in run["answer"]
    assert client.post("/runs", json={"bbox": BBOX}).status_code == 400
    assert client.post("/runs", json={"bbox": BBOX, "workflow": "nope"}).status_code == 400
    assert client.post("/runs", json={"bbox": [1, 1, 0, 0], "instruction": "x"}).status_code == 422
    assert client.get("/runs/../../etc/passwd").status_code == 404


def test_projects_and_history():
    headers = _login("planner1")
    pid = client.post("/projects", json={"name": "Lake survey"}, headers=headers).json()["id"]
    run_id = client.post("/runs", json={"bbox": BBOX, "workflow": "water_encroachment", "project_id": pid},
                         headers=headers).json()["id"]
    _wait(run_id, headers)
    client.patch(f"/runs/{run_id}", json={"saved": True, "title": "Bellandur 2020-25"}, headers=headers)
    listed = client.get(f"/runs?project_id={pid}", headers=headers).json()
    assert listed[0]["title"] == "Bellandur 2020-25" and listed[0]["saved"] == 1
    assert client.get("/projects", headers=headers).json()[0]["run_count"] == 1
    other = _login("planner2")
    assert client.post("/runs", json={"bbox": BBOX, "instruction": "x", "project_id": pid},
                       headers=other).status_code == 404


# -- workflows ----------------------------------------------------------------------------------------

def test_catalog_covers_all_sectors():
    cat = client.get("/workflows").json()
    assert {s["id"] for s in cat["sectors"]} == {"disaster", "forest", "urban", "agriculture"}
    assert all(s["workflows"] for s in cat["sectors"])


def test_relative_dates_and_param_validation():
    from datetime import date
    today = date(2026, 9, 24)
    assert workflows.resolve_date("-3y", today) == "2023-09-24"
    assert workflows.resolve_date("-6m", today) == "2026-03-24"
    assert workflows.resolve_date("-30d", today) == "2026-08-25"
    with pytest.raises(ValueError):
        workflows.get("flood_risk").coerce({"water_rise_m": -1})


@pytest.mark.parametrize("wid", list(workflows.WORKFLOWS))
def test_every_workflow_runs(wid):
    run = runs.execute(runs.create(BBOX, workflow=wid)["id"])
    assert run["status"] == "done", run["error"]
    assert run["answer"] and run["insights"]["metrics"]
    assert set(workflows.get(wid).metrics) <= set(run["insights"]["metrics"])


# -- monitoring and place search -----------------------------------------------------------------------

def test_monitor_triggers_alert():
    headers = _login("warden")
    user = client.get("/auth/me", headers=headers).json()["user"]
    m = monitoring.create(user, "Forest watch", "deforestation", {"start_date": "-5y", "end_date": "-45d"},
                          BBOX, "weekly", {"metric": "forest_loss_km2", "op": "gt", "value": 0.01}, run_now=False)
    result = monitoring.check(m)
    assert result["triggered"] and result["value"] > 0.01
    alerts = client.get("/alerts", headers=headers).json()
    assert alerts["unread"] == 1 and "forest_loss_km2" in alerts["alerts"][0]["message"]
    listed = client.get("/monitors", headers=headers).json()[0]
    assert listed["last_value"] == result["value"] and listed["history"]
    quiet = monitoring.create(user, "Quiet", "deforestation", {}, BBOX, "weekly",
                              {"metric": "forest_loss_km2", "op": "gt", "value": 1e6}, run_now=False)
    assert not monitoring.check(quiet)["triggered"]
    client.post("/alerts/read", headers=headers)
    assert client.get("/alerts", headers=headers).json()["unread"] == 0


def test_monitor_validation():
    headers = _login("warden2")
    bad = client.post("/monitors", headers=headers, json={
        "name": "x", "workflow": "deforestation", "bbox": BBOX, "rule": {"metric": "nope", "op": "gt", "value": 1}})
    assert bad.status_code == 400


def test_scheduler_runs_due_monitors():
    user = auth.create_user("sched_user", "password123", "official")
    monitoring.create(user, "Due", "crop_health", {}, BBOX, "daily",
                      {"metric": "stressed_cropland_km2", "op": "ge", "value": 0}, run_now=True)
    assert monitoring.Scheduler(0).tick() >= 1
    assert monitoring.due_monitors() == []


def test_offline_geocoder():
    hits = geocode.search("Bengaluru")
    assert hits[0]["name"] == "Bengaluru" and hits[0]["source"] == "gazetteer"
    assert len(hits[0]["analysis_bbox"]) == 4
    assert client.get("/geocode", params={"q": "chilika"}).json()[0]["type"] == "lake"
    assert geocode.search("x") == []
    assert json.loads(json.dumps(hits))  # JSON-safe
