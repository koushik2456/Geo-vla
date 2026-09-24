"""3D terrain model + derived textures."""
import base64
import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from services import runs, terrain

client = TestClient(app)
BBOX = [77.50, 12.90, 77.60, 13.00]


def _run(workflow, **params):
    return runs.execute(runs.create(BBOX, workflow=workflow, params=params)["id"])


def test_terrain_model_and_contours_align_with_mesh():
    run = _run("flood_risk", water_rise_m=40)
    m = terrain.terrain_model(run)
    heights = np.frombuffer(base64.b64decode(m["heights"]), dtype="<f4")
    assert heights.size == m["rows"] * m["cols"] and m["min"] < m["max"]
    assert m["water"] and m["min"] < m["water"][0]["level"] < m["max"]
    for key in ("contours", "dense_contours"):
        assert m[key], key
        pts = np.array([p for c in m[key] for line in c["lines"] for p in line])
        # contours traced on coarser grids must be rescaled to the mesh grid
        assert pts[:, 0].max() > 0.9 * (m["rows"] - 1) or pts[:, 1].max() > 0.9 * (m["cols"] - 1)
        assert pts[:, 0].max() <= m["rows"] - 1 + 1e-6 and pts[:, 1].max() <= m["cols"] - 1 + 1e-6
    assert len(m["dense_contours"]) > len(m["contours"])
    assert any(c["major"] for c in m["contours"])


def test_terrain_on_demand_dem_and_textures():
    run = _run("urban_growth")                      # never loads a DEM itself
    assert terrain.terrain_model(run)["rows"] > 10
    for kind in ("hypsometric", "hillshade", "height", "normal", "ao"):
        img = Image.open(io.BytesIO(terrain.texture_png(run, kind)))
        assert img.size[0] >= 256
    normal = np.array(Image.open(io.BytesIO(terrain.texture_png(run, "normal"))))
    assert normal[..., 2].mean() > 128             # normal maps point mostly "up" (blue)


def test_terrain_api():
    rid = client.post("/runs", json={"bbox": BBOX, "workflow": "crop_health"}).json()["id"]
    for _ in range(200):
        if client.get(f"/runs/{rid}").json()["status"] == "done":
            break
        time.sleep(0.2)
    body = client.get(f"/runs/{rid}/terrain").json()
    assert body["contour_interval"] > 0 and body["textures"]
    assert client.get(f"/runs/{rid}/terrain/normal.png").headers["content-type"] == "image/png"
    assert client.get(f"/runs/{rid}/terrain/nope.png").status_code == 404
    layer = client.get(f"/runs/{rid}").json()["layers"][0]
    big = client.get(layer["preview_url"] + "?size=1024")
    assert big.status_code == 200 and Image.open(io.BytesIO(big.content)).size[0] > 512
