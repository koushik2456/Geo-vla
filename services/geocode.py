"""
services/geocode.py — Find a village, city, district, lake or forest by name.

Uses the Google Geocoding API when GOOGLE_MAPS_API_KEY is set (most accurate
addresses, plus codes), OpenStreetMap Nominatim otherwise, and a bundled
gazetteer of Indian cities, lakes, forests and hazard-prone areas offline.
`reverse()` turns a clicked point back into a place description.
Large places (districts, states) are returned with their full extent plus an
`analysis_bbox` centred on them and capped at MAX_ANALYSIS_DEG, because
10 m analyses of whole districts are slow and coarse.
"""
import json
import logging
import math
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


def _places() -> list:
    global _gazetteer
    if _gazetteer is None:
        with open(_GAZETTEER_PATH) as f:
            _gazetteer = json.load(f)
    return _gazetteer


def gazetteer_search(query: str, limit: int = 8) -> list:
    q = query.strip().lower()
    hits = [p for p in _places() if q in p["name"].lower() or q in p["state"].lower()]
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


def _google(params: dict) -> list:
    resp = requests.get(config.GOOGLE_GEOCODE_URL, timeout=10, params={**params, "key": config.GOOGLE_GEOCODING_KEY})
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(f"Google Geocoding: {body.get('status')} {body.get('error_message', '')}".strip())
    return body.get("results", [])


def google_search(query: str, limit: int = 8) -> list:
    out = []
    for r in _google({"address": query})[:limit]:
        vp = r["geometry"].get("bounds") or r["geometry"]["viewport"]
        bbox = [vp["southwest"]["lng"], vp["southwest"]["lat"], vp["northeast"]["lng"], vp["northeast"]["lat"]]
        loc = r["geometry"]["location"]
        name = r["address_components"][0]["long_name"] if r.get("address_components") else r["formatted_address"]
        out.append(_result(name, r["formatted_address"], (r.get("types") or ["place"])[0], bbox,
                           loc["lat"], loc["lng"], "Google Geocoding"))
    return out


def search(query: str, limit: int = 8) -> list:
    query = query.strip()
    if len(query) < 2:
        return []
    results = []
    if not config.OFFLINE:
        providers = ([("Google", google_search)] if config.GOOGLE_GEOCODING_KEY else []) + [("Nominatim", nominatim_search)]
        for name, fn in providers:
            try:
                results = fn(query, limit)
            except Exception as exc:  # network, quota or key problem → next provider
                log.info("%s geocoding unavailable (%s)", name, exc)
            if results:
                break
    return results or gazetteer_search(query, limit)


# ---------------------------------------------------------------------------
# Reverse geocoding (point → place)
# ---------------------------------------------------------------------------

_GOOGLE_COMPONENTS = {
    "route": "road", "sublocality": "suburb", "sublocality_level_1": "suburb", "locality": "locality",
    "administrative_area_level_3": "subdistrict", "administrative_area_level_2": "district",
    "administrative_area_level_1": "state", "country": "country", "postal_code": "postcode",
}
_NOMINATIM_COMPONENTS = {
    "road": "road", "suburb": "suburb", "neighbourhood": "suburb", "village": "locality", "town": "locality",
    "city": "locality", "county": "subdistrict", "state_district": "district", "state": "state",
    "country": "country", "postcode": "postcode",
}


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def google_reverse(lon: float, lat: float) -> dict:
    results = _google({"latlng": f"{lat},{lon}"})
    if not results:
        return None
    top = results[0]
    comps = {}
    for r in results[:4]:  # the first result can be a street address missing the admin levels
        for c in r.get("address_components", []):
            for t in c["types"]:
                if t in _GOOGLE_COMPONENTS:
                    comps.setdefault(_GOOGLE_COMPONENTS[t], c["long_name"])
    plus = top.get("plus_code") or {}
    return {"source": "Google Geocoding", "formatted_address": top["formatted_address"],
            "name": comps.get("locality") or comps.get("suburb") or top["formatted_address"].split(",")[0],
            "components": comps, "plus_code": plus.get("global_code"), "types": top.get("types", [])}


def nominatim_reverse(lon: float, lat: float) -> dict:
    resp = requests.get(config.NOMINATIM_REVERSE_URL, timeout=10,
                        params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
                        headers={"User-Agent": "Geo-VLA geospatial decision support (research prototype)"})
    resp.raise_for_status()
    r = resp.json()
    if "error" in r:
        return None
    addr = r.get("address", {})
    comps = {}
    for k, ours in _NOMINATIM_COMPONENTS.items():
        if k in addr:
            comps.setdefault(ours, addr[k])
    return {"source": "OpenStreetMap Nominatim", "formatted_address": r.get("display_name", ""),
            "name": r.get("name") or comps.get("locality") or r.get("display_name", "").split(",")[0],
            "components": comps, "plus_code": None, "types": [t for t in (r.get("category"), r.get("type")) if t]}


def gazetteer_reverse(lon: float, lat: float) -> dict:
    best = min(_places(), key=lambda p: haversine_km(lon, lat, p["lon"], p["lat"]))
    dist = haversine_km(lon, lat, best["lon"], best["lat"])
    inside = best["bbox"][0] <= lon <= best["bbox"][2] and best["bbox"][1] <= lat <= best["bbox"][3]
    label = best["name"] if inside else f"{dist:.1f} km from {best['name']}"
    return {"source": "bundled gazetteer (offline)", "formatted_address": f"{label}, {best['state']}, India",
            "name": label, "components": {"locality": best["name"], "state": best["state"], "country": "India"},
            "plus_code": None, "types": [best["type"]], "nearest_km": round(dist, 2)}


def reverse(lon: float, lat: float) -> dict:
    """Describe the place at a point, with the most accurate provider available."""
    if not config.OFFLINE:
        providers = ([("Google", google_reverse)] if config.GOOGLE_GEOCODING_KEY else []) + [("Nominatim", nominatim_reverse)]
        for name, fn in providers:
            try:
                found = fn(lon, lat)
            except Exception as exc:
                log.info("%s reverse geocoding unavailable (%s)", name, exc)
                continue
            if found:
                return found
    return gazetteer_reverse(lon, lat)
