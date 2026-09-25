"""
services/explorer.py — Point explorer for the globe: what is here, and what does the network see?

`point_info` describes a clicked location (address, elevation, UTM zone).
`classify_point` fetches a 1.92 km Sentinel-2 window around the point, splits it
into the 3×3 grid of 640 m EuroSAT-sized patches, classifies each with the
active ResNet-50, and explains the centre prediction with Grad-CAM, feature
maps at every stage, the embedding, and the patch's spectral signature.
"""
import base64
import math
from datetime import date, timedelta

import cv2
import numpy as np

import config
import geotools
import rendering
from geo_utils import M_PER_DEG_LAT, M_PER_DEG_LON_EQ
from models import EUROSAT_CLASSES
from services import geocode

PATCH_PX = 64                  # EuroSAT patch: 64 px × 10 m = 640 m
GRID = 3                       # 3×3 neighbourhood around the clicked patch
CONTEXT_PX = PATCH_PX * GRID
CLASS_COLORS = {c: "#%02x%02x%02x" % rendering.CLASS_COLORS[c] for c in EUROSAT_CLASSES}  # same as map layers
STAGES = [("stem", "relu"), ("layer1", "layer1"), ("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4")]


def _png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _half_deg(lat: float, metres: float) -> tuple:
    return metres / (M_PER_DEG_LON_EQ * max(math.cos(math.radians(lat)), 0.01)), metres / M_PER_DEG_LAT


def _check(lon: float, lat: float):
    if not (-180 <= lon <= 180 and -85 <= lat <= 85):
        raise ValueError("lon must be within ±180 and lat within ±85")


def utm_zone(lon: float, lat: float) -> dict:
    zone = int((lon + 180) // 6) % 60 + 1
    return {"zone": f"{zone}{'N' if lat >= 0 else 'S'}", "epsg": (32600 if lat >= 0 else 32700) + zone}


def point_info(lon: float, lat: float) -> dict:
    _check(lon, lat)
    hx, hy = _half_deg(lat, 300)
    ws = geotools.Workspace([lon - hx, lat - hy, lon + hx, lat + hy])
    elevation = None
    try:
        dem = geotools.fetch_dem(ws)
        grid = ws.get("dem").data
        elevation = {"m": round(float(grid[grid.shape[0] // 2, grid.shape[1] // 2]), 1),
                     "local_min_m": dem["min_m"], "local_max_m": dem["max_m"], "source": dem["source"]}
    except Exception as exc:  # ocean, network: the place card still works
        elevation = {"m": None, "error": str(exc)}
    return {"lon": lon, "lat": lat, "place": geocode.reverse(lon, lat), "elevation": elevation,
            "utm": utm_zone(lon, lat), "data_mode": config.data_mode(),
            "imagery_source": config.imagery_source() if config.data_live() else "synthetic"}


# ---------------------------------------------------------------------------
# Neural network exploration
# ---------------------------------------------------------------------------

def _default_date() -> str:
    return (date.today() - timedelta(days=30)).isoformat()


def _fetch_context(lon: float, lat: float, day: str):
    hx, hy = _half_deg(lat, CONTEXT_PX * geotools.S2_RESOLUTION_M / 2)
    bbox = [lon - hx, lat - hy, lon + hx, lat + hy]
    ws = geotools.Workspace(bbox)
    info = geotools.fetch_sentinel2_scene(ws, day, window_days=45 if config.data_live() else 20, max_cloud_pct=40)
    scene = ws.latest("scene")
    rgb, bands = scene.data["rgb"], scene.data["bands"]
    if rgb.shape[:2] != (CONTEXT_PX, CONTEXT_PX):  # bbox rounding can give 191–193 px
        rgb = cv2.resize(rgb, (CONTEXT_PX, CONTEXT_PX), interpolation=cv2.INTER_LINEAR)
        bands = {k: cv2.resize(v, (CONTEXT_PX, CONTEXT_PX), interpolation=cv2.INTER_LINEAR) for k, v in bands.items()}
    return bbox, rgb, bands, info["source"]


def _tiles(arr: np.ndarray) -> list:
    return [arr[r * PATCH_PX:(r + 1) * PATCH_PX, c * PATCH_PX:(c + 1) * PATCH_PX]
            for r in range(GRID) for c in range(GRID)]


def _spectral(bands: dict) -> dict:
    mean = {k: float(np.mean(v)) for k, v in bands.items()}
    b, g, r, nir = mean["B02"], mean["B03"], mean["B04"], mean["B08"]
    return {
        "signature": [{"band": "B02 blue", "nm": 490, "reflectance": round(b, 4)},
                      {"band": "B03 green", "nm": 560, "reflectance": round(g, 4)},
                      {"band": "B04 red", "nm": 665, "reflectance": round(r, 4)},
                      {"band": "B08 NIR", "nm": 842, "reflectance": round(nir, 4)}],
        "ndvi": round((nir - r) / (nir + r + 1e-6), 3),
        "ndwi": round((g - nir) / (g + nir + 1e-6), 3),
        "brightness": round((b + g + r) / 3, 4),
    }


def _feature_strip(fmap: np.ndarray, n: int = 6, size: int = 72) -> str:
    """fmap (C,H,W) → strip of the n most active channels, each min-max scaled, viridis."""
    order = np.argsort(-fmap.mean(axis=(1, 2)))[:n]
    tiles = []
    for ch in order:
        m = fmap[ch]
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        img = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tiles.append(np.full((size, 3, 3), 12, np.uint8))
    return _png(np.concatenate(tiles[:-1], axis=1)), [int(c) for c in order]


def _cnn_explain(model, device, patches: list) -> dict:
    import torch

    size = geotools.classifier_input_size()
    x = torch.from_numpy(np.stack([geotools._normalize(p) for p in patches])).to(device)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    net = model.model
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).cpu().numpy()

    # Centre patch again with gradients: Grad-CAM on layer4, activations at every stage.
    acts, grads = {}, {}
    hooks = []
    for label, attr in STAGES:
        def hook(_m, _i, out, label=label):
            acts[label] = out
            if label == "layer4":
                out.register_hook(lambda g: grads.__setitem__("layer4", g))
        hooks.append(getattr(net, attr).register_forward_hook(hook))
    pooled = {}
    hooks.append(net.avgpool.register_forward_hook(lambda _m, _i, out: pooled.__setitem__("z", out)))
    centre = x[GRID * GRID // 2:GRID * GRID // 2 + 1]
    try:
        with torch.enable_grad():
            logits = model(centre)
            target = int(logits.argmax(1))
            model.zero_grad(set_to_none=True)
            logits[0, target].backward()
    finally:
        for h in hooks:
            h.remove()

    a = acts["layer4"].detach()[0]
    weights = grads["layer4"][0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * a).sum(0)).cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    stages = []
    for label, _ in STAGES:
        fmap = acts[label].detach()[0].cpu().numpy()
        img, channels = _feature_strip(fmap)
        stages.append({"stage": label, "shape": list(fmap.shape), "image": img, "channels": channels,
                       "mean_activation": round(float(fmap.mean()), 4),
                       "sparsity": round(float((fmap <= 0).mean()), 3)})
    z = pooled["z"].detach().flatten().cpu().numpy()
    return {"probs": probs, "cam": cam, "stages": stages, "input_size": size,
            "logits": [round(float(v), 3) for v in logits.detach()[0].cpu().numpy()],
            "embedding": {"dim": int(z.size), "l2_norm": round(float(np.linalg.norm(z)), 3),
                          "active_fraction": round(float((z > 0).mean()), 3),
                          "top_units": [int(i) for i in np.argsort(-z)[:8]]}}


def _spectral_rules_probs(bands_tiles: list) -> np.ndarray:
    """Fallback: class shares of the rule-based per-pixel map, used as pseudo-probabilities."""
    out = []
    for tb in bands_tiles:
        cm = geotools._classify_spectral(tb)
        counts = np.bincount(cm.ravel().astype(np.int64), minlength=len(EUROSAT_CLASSES)).astype(np.float64)
        out.append(counts / counts.sum())
    return np.array(out)


def _overlay_cam(patch_rgb: np.ndarray, cam: np.ndarray, size: int = 224) -> str:
    base = cv2.resize(patch_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return _png(cv2.addWeighted(base, 0.55, heat, 0.45, 0))


def _context_image(rgb: np.ndarray, labels: list, scale: int = 2) -> str:
    img = cv2.resize(rgb, (CONTEXT_PX * scale, CONTEXT_PX * scale), interpolation=cv2.INTER_CUBIC)
    step = PATCH_PX * scale
    for i, name in enumerate(labels):
        r, c = divmod(i, GRID)
        hexc = CLASS_COLORS[name].lstrip("#")
        color = tuple(int(hexc[j:j + 2], 16) for j in (0, 2, 4))
        thick = 3 if i == GRID * GRID // 2 else 1
        cv2.rectangle(img, (c * step + 1, r * step + 1), ((c + 1) * step - 2, (r + 1) * step - 2), color, thick)
    return _png(img)


def classify_point(lon: float, lat: float, day: str = None) -> dict:
    _check(lon, lat)
    day = day or _default_date()
    bbox, rgb, bands, source = _fetch_context(lon, lat, day)
    patches = _tiles(rgb)
    band_tiles = [dict(zip(bands, t)) for t in zip(*(_tiles(v) for v in bands.values()))]
    model, device = geotools._load_model("classifier")
    centre = GRID * GRID // 2
    explain = None
    if model is not None:
        explain = _cnn_explain(model, device, patches)
        probs = explain["probs"]
        method = f"ResNet-50 (EuroSAT), model {geotools.model_version('classifier')}"
    else:
        probs = _spectral_rules_probs(band_tiles)
        method = "spectral rules (no trained classifier yet: train one in the Training studio for Grad-CAM)"

    labels = [EUROSAT_CLASSES[int(p.argmax())] for p in probs]
    w, s, e, n = bbox
    dx, dy = (e - w) / GRID, (n - s) / GRID
    grid = []
    for i, p in enumerate(probs):
        r, c = divmod(i, GRID)
        grid.append({"row": r, "col": c, "class": labels[i], "confidence": round(float(p.max()), 3),
                     "color": CLASS_COLORS[labels[i]],
                     "bbox": [round(w + c * dx, 6), round(n - (r + 1) * dy, 6), round(w + (c + 1) * dx, 6), round(n - r * dy, 6)]})
    top = np.argsort(-probs[centre])[:5]
    p = probs[centre]
    out = {
        "lon": lon, "lat": lat, "date": day, "imagery": source, "method": method, "neural": model is not None,
        "prediction": {"class": labels[centre], "confidence": round(float(p.max()), 3), "color": CLASS_COLORS[labels[centre]],
                       "entropy_bits": round(float(-(p * np.log2(p + 1e-12)).sum()), 3),
                       "top5": [{"class": EUROSAT_CLASSES[i], "p": round(float(p[i]), 4), "color": CLASS_COLORS[EUROSAT_CLASSES[i]]}
                                for i in top]},
        "patch": {"size_px": PATCH_PX, "size_m": PATCH_PX * geotools.S2_RESOLUTION_M, "bbox": grid[centre]["bbox"],
                  "image": _png(cv2.resize(patches[centre], (224, 224), interpolation=cv2.INTER_NEAREST))},
        "context": {"bbox": [round(v, 6) for v in bbox], "image": _context_image(rgb, labels), "grid": grid},
        "spectral": _spectral(band_tiles[centre]),
    }
    if explain:
        out["network"] = {
            "input": {"size": explain["input_size"], "normalisation": "ImageNet mean/std",
                      "tensor_shape": [1, 3, explain["input_size"], explain["input_size"]]},
            "gradcam": _overlay_cam(patches[centre], explain["cam"]),
            "stages": explain["stages"],
            "logits": dict(zip(EUROSAT_CLASSES, explain["logits"])),
            "embedding": explain["embedding"],
        }
    return out
