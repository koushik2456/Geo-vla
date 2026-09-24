"""
services/geocode.py — Find a village, city, district, lake or forest by name.

Uses OpenStreetMap Nominatim when reachable; otherwise (or additionally) a
bundled gazetteer of Indian cities, lakes, forests and hazard-prone areas.
Large places (districts, states) are returned with their full extent plus an
`analysis_bbox` centred on them and capped at MAX_ANALYSIS_DEG, because
10 m analyses of whole districts are slow and coarse.
"""
import json
import logging
import os

import requests

import config

log = logging.getLogger("geo-vla.geocode")
MAX_ANALYSIS_DEG = 0.25
_GAZETTEER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gazetteer.json")
_gazetteer = None


def _analysis_bbox(bbox: list) -> list:
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    hw, hh = min((e - w) / 2, MAX_ANALYSIS_DEG / 2), min((n - s) / 2, MAX_ANALYSIS_DEG / 2)
    return [round(cx - hw, 5), round(cy - hh, 5), round(cx + hw, 5), round(cy + hh, 5)]


def _result(name, label, kind, bbox, lat, lon, source) -> dict:
    return {"name": name, "label": label, "type": kind, "bbox": bbox, "center": [lon, lat],
            "analysis_bbox": _analysis_bbox(bbox), "clipped": _analysis_bbox(bbox) != bbox, "source": source}


def gazetteer_search(query: str, limit: int = 8) -> list:
    global _gazetteer
    if _gazetteer is None:
        with open(_GAZETTEER_PATH) as f:
            _gazetteer = json.load(f)
    q = query.strip().lower()
    hits = [p for p in _gazetteer if q in p["name"].lower() or q in p["state"].lower()]
    hits.sort(key=lambda p: (not p["name"].lower().startswith(q), p["name"]))
    return [_result(p["name"], f"{p['name']}, {p['state']}", p["type"], p["bbox"], p["lat"], p["lon"], "gazetteer")
            for p in hits[:limit]]


def nominatim_search(query: str, limit: int = 8) -> list:
    resp = requests.get(config.NOMINATIM_URL, timeout=10,
                        params={"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 0},
                        headers={"User-Agent": "Geo-VLA geospatial decision support (research prototype)"})
    resp.raise_for_status()
    out = []
    for r in resp.json():
        s, n, w, e = (float(v) for v in r["boundingbox"])
        out.append(_result(r.get("name") or r["display_name"].split(",")[0], r["display_name"],
                           r.get("type") or r.get("category", "place"), [w, s, e, n],
                           float(r["lat"]), float(r["lon"]), "OpenStreetMap Nominatim"))
    return out


def search(query: str, limit: int = 8) -> list:
    query = query.strip()
    if len(query) < 2:
        return []
    results = []
    if not config.OFFLINE:
        try:
            results = nominatim_search(query, limit)
        except Exception as exc:  # network or rate limit → offline list
            log.info("Nominatim unavailable (%s); using the bundled gazetteer", exc)
    return results or gazetteer_search(query, limit)
