"""
services/workflows.py — Ready-made analyses for government departments.

Each workflow is a fixed, parameterised tool chain (no LLM involved), so the
same inputs always give the same method — important when numbers go into an
official report. Workflows run through the same tools and trace as the agent,
and every one returns:

  answer       plain-language summary
  key_figures  headline numbers for the dashboard and PDF
  metrics      machine-readable values that monitoring rules can alert on
  charts       chart specs rendered by the UI and the PDF report

Sectors: disaster management, forest & environment, urban planning, agriculture.
Dates accept ISO strings or relative values: "today", "-30d", "-6m", "-3y".
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from geo_utils import bbox_area_km2
from planner import Plan

SECTORS = {
    "disaster": {"title": "Disaster management", "icon": "flood",
                 "audience": "District disaster management authorities, relief planners"},
    "forest": {"title": "Forest & environment", "icon": "forest",
               "audience": "Forest departments, pollution control boards, environment officers"},
    "urban": {"title": "Urban planning", "icon": "city",
              "audience": "Municipal corporations, town planning, development authorities"},
    "agriculture": {"title": "Agriculture", "icon": "crop",
                    "audience": "Agriculture officers, crop insurance, farmers"},
}


def resolve_date(value: str, today: date = None) -> str:
    """ISO date, or relative: today, -30d, -6m, -3y (days / months / years before today)."""
    today = today or date.today()
    value = str(value).strip().lower()
    if value == "today":
        return today.isoformat()
    m = re.fullmatch(r"-(\d+)([dmy])", value)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "d":
            return (today - timedelta(days=n)).isoformat()
        months = n if unit == "m" else 12 * n
        y, mo = divmod(today.year * 12 + today.month - 1 - months, 12)
        return date(y, mo + 1, min(today.day, 28)).isoformat()
    return date.fromisoformat(value[:10]).isoformat()


@dataclass
class Param:
    name: str
    label: str
    type: str                 # date | number | integer | boolean | numbers
    default: object
    help: str = ""
    min: float = None
    max: float = None

    def spec(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@dataclass
class Workflow:
    id: str
    sector: str
    title: str
    summary: str
    params: list
    outputs: list = field(default_factory=list)   # what the official gets
    metrics: list = field(default_factory=list)   # alertable metric names (for monitoring)

    def spec(self) -> dict:
        return {"id": self.id, "sector": self.sector, "title": self.title, "summary": self.summary,
                "params": [p.spec() for p in self.params], "outputs": self.outputs, "metrics": self.metrics}

    def coerce(self, raw: dict, today: date = None) -> dict:
        raw = raw or {}
        out = {}
        for p in self.params:
            v = raw.get(p.name, p.default)
            if p.type == "date":
                v = resolve_date(v, today)
            elif p.type == "number":
                v = float(v)
            elif p.type == "integer":
                v = int(v)
            elif p.type == "boolean":
                v = v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}
            elif p.type == "numbers":
                v = sorted({float(x) for x in (v if isinstance(v, list) else str(v).split(","))})
            if p.min is not None and v is not None and not isinstance(v, (list, str)) and v < p.min:
                raise ValueError(f"{p.label} must be ≥ {p.min}")
            if p.max is not None and v is not None and not isinstance(v, (list, str)) and v > p.max:
                raise ValueError(f"{p.label} must be ≤ {p.max}")
            out[p.name] = v
        return out


# -- helpers -----------------------------------------------------------------------

def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _km2(x) -> str:
    return f"{x:,.2f} km²"


def _fig(label, value, unit="", note=""):
    return {"label": label, "value": value, "unit": unit, "note": note}


def _class_bar(title: str, fractions: dict, area_km2: float) -> dict:
    items = sorted(fractions.items(), key=lambda kv: -kv[1])
    return {"type": "bar", "title": title, "unit": "km²", "categories": [k for k, _ in items],
            "series": [{"name": "Area", "values": [round(v * area_km2, 3) for _, v in items]}]}


def _two_date_bar(title: str, lc1: dict, lc2: dict, d1: str, d2: str, area_km2: float) -> dict:
    classes = sorted(set(lc1) | set(lc2), key=lambda c: -(lc1.get(c, 0) + lc2.get(c, 0)))
    return {"type": "bar", "title": title, "unit": "km²", "categories": classes,
            "series": [{"name": d1, "values": [round(lc1.get(c, 0) * area_km2, 3) for c in classes]},
                       {"name": d2, "values": [round(lc2.get(c, 0) * area_km2, 3) for c in classes]}]}


def _get(results: dict, key: str, field_: str, default=0.0):
    return results.get(key, {}).get(field_, default)


# -- workflow definitions --------------------------------------------------------------------

class FloodRisk(Workflow):
    def __init__(self):
        super().__init__(
            "flood_risk", "disaster", "Flood risk assessment",
            "Which areas, settlements, farmland and roads flood if water rises by a given height?",
            [Param("water_rise_m", "Water rise above lowest point (m)", "number", 5.0, min=0.5, max=100),
             Param("date", "Imagery date for land cover", "date", "-45d", "Recent cloud-free date")],
            ["Flood extent map", "Settlements and cropland at risk", "Roads flooded (km)", "Land cover affected"],
            ["flooded_km2", "flooded_pct", "built_up_at_risk_km2", "cropland_at_risk_km2", "roads_flooded_km"],
        )

    def build(self, p):
        d, lvl = p["date"], p["water_rise_m"]
        plan = Plan(rationale=f"Workflow: flood risk at +{lvl:g} m, land cover from {d}")
        flood = f"flood_{lvl:g}m"
        plan.add("dem", "fetch_dem")
        plan.add("flood", "flood_extent", water_level_m=lvl, mode="above_min")
        plan.add("scene", "fetch_sentinel2_scene", date=d)
        plan.add("lc", "classify_scene", scene_id=f"s2_{d}")
        plan.add("flood_lc", "zonal_stats", value_layer=f"landcover_{d}", zone_layer=flood)
        plan.add("built", "class_mask", layer_id=f"landcover_{d}", group="built_up")
        plan.add("built_risk", "overlay_layers", layer_a=f"built_up_{d}", layer_b=flood, operation="intersect",
                 name="built_up_at_risk")
        plan.add("crop", "class_mask", layer_id=f"landcover_{d}", group="cropland")
        plan.add("crop_risk", "overlay_layers", layer_a=f"cropland_{d}", layer_b=flood, operation="intersect",
                 name="cropland_at_risk")
        plan.add("roads", "fetch_osm_features", _optional=True, feature_type="roads")
        plan.add("roads_flooded", "features_in_zone", _optional=True, layer_id="osm_roads", zone_layer=flood)
        return plan

    def summarize(self, p, r, bbox):
        f = r["flood"]
        metrics = {"flooded_km2": f["area_km2"], "flooded_pct": round(100 * f["fraction"], 2),
                   "built_up_at_risk_km2": _get(r, "built_risk", "area_km2"),
                   "cropland_at_risk_km2": _get(r, "crop_risk", "area_km2"),
                   "roads_flooded_km": _get(r, "roads_flooded", "length_km_in_zone", None)}
        lines = [f"A {p['water_rise_m']:g} m rise (to {f['water_level_m_asl']} m above sea level) would flood "
                 f"{_km2(f['area_km2'])} ({_pct(f['fraction'])} of the area).",
                 f"Built-up land at risk: {_km2(metrics['built_up_at_risk_km2'])}; "
                 f"cropland at risk: {_km2(metrics['cropland_at_risk_km2'])}."]
        if metrics["roads_flooded_km"] is not None:
            lines.append(f"Major roads inside the flood zone: {metrics['roads_flooded_km']:.1f} km "
                         f"({r['roads_flooded']['features_in_zone']} road segments).")
        figures = [_fig("Flooded area", f["area_km2"], "km²", _pct(f["fraction"]) + " of area"),
                   _fig("Built-up at risk", metrics["built_up_at_risk_km2"], "km²"),
                   _fig("Cropland at risk", metrics["cropland_at_risk_km2"], "km²"),
                   _fig("Water level", f["water_level_m_asl"], "m a.s.l.")]
        if metrics["roads_flooded_km"] is not None:
            figures.append(_fig("Roads flooded", metrics["roads_flooded_km"], "km"))
        charts = []
        if r.get("flood_lc", {}).get("class_fractions"):
            charts.append(_class_bar("Land cover inside the flood zone", r["flood_lc"]["class_fractions"],
                                     f["area_km2"]))
        return lines, figures, metrics, charts, "built_up_at_risk"


class FloodScenarios(Workflow):
    def __init__(self):
        super().__init__(
            "flood_scenarios", "disaster", "Flood scenarios (multiple water levels)",
            "Compare flooded area and built-up exposure across several water-level scenarios.",
            [Param("levels", "Water rises to compare (m, comma-separated)", "numbers", [1, 2, 5, 10]),
             Param("date", "Imagery date for land cover", "date", "-45d")],
            ["Flood extent per scenario", "Exposure curve (area vs water level)"],
            ["max_flooded_km2", "max_built_up_at_risk_km2"],
        )

    def build(self, p):
        d = p["date"]
        plan = Plan(rationale=f"Workflow: flood scenarios {', '.join(f'{x:g}' for x in p['levels'])} m")
        plan.add("dem", "fetch_dem")
        plan.add("scene", "fetch_sentinel2_scene", date=d)
        plan.add("lc", "classify_scene", scene_id=f"s2_{d}")
        plan.add("built", "class_mask", layer_id=f"landcover_{d}", group="built_up")
        for lvl in p["levels"]:
            plan.add(f"flood_{lvl:g}", "flood_extent", water_level_m=lvl, mode="above_min")
            plan.add(f"built_{lvl:g}", "overlay_layers", layer_a=f"flood_{lvl:g}m", layer_b=f"built_up_{d}",
                     operation="intersect", name=f"built_up_flooded_{lvl:g}m")
        return plan

    def summarize(self, p, r, bbox):
        levels = p["levels"]
        flooded = [_get(r, f"flood_{x:g}", "area_km2") for x in levels]
        built = [_get(r, f"built_{x:g}", "area_km2") for x in levels]
        lines = [f"+{x:g} m → {_km2(a)} flooded, {_km2(b)} built-up exposed." for x, a, b in zip(levels, flooded, built)]
        figures = [_fig(f"Flooded at +{x:g} m", a, "km²") for x, a in zip(levels, flooded)]
        charts = [{"type": "line", "title": "Exposure curve", "unit": "km²", "x_label": "Water rise (m)",
                   "x": [f"{x:g}" for x in levels],
                   "series": [{"name": "Flooded area", "values": flooded},
                              {"name": "Built-up flooded", "values": built}]}]
        metrics = {"max_flooded_km2": max(flooded), "max_built_up_at_risk_km2": max(built)}
        return lines, figures, metrics, charts, f"flood_{levels[-1]:g}m"


class DamageAssessment(Workflow):
    def __init__(self):
        super().__init__(
            "damage_assessment", "disaster", "Post-event damage assessment",
            "Compare imagery before and after a flood, cyclone or landslide to map what changed.",
            [Param("pre_date", "Before the event", "date", "-60d"),
             Param("post_date", "After the event", "date", "-10d")],
            ["Change map", "Built-up and cropland affected", "Roads in changed areas"],
            ["changed_km2", "built_up_affected_km2", "cropland_affected_km2"],
        )

    def build(self, p):
        a, b = p["pre_date"], p["post_date"]
        plan = Plan(rationale=f"Workflow: damage assessment {a} → {b}")
        plan.add("pre", "fetch_sentinel2_scene", date=a, window_days=10)
        plan.add("post", "fetch_sentinel2_scene", date=b, window_days=10)
        plan.add("change", "detect_change", scene_id_1=f"s2_{a}", scene_id_2=f"s2_{b}")
        plan.add("lc", "classify_scene", scene_id=f"s2_{a}")
        plan.add("built", "class_mask", layer_id=f"landcover_{a}", group="built_up")
        plan.add("crop", "class_mask", layer_id=f"landcover_{a}", group="cropland")
        plan.add("built_hit", "overlay_layers", layer_a=f"change_{a}_{b}", layer_b=f"built_up_{a}",
                 operation="intersect", name="built_up_affected")
        plan.add("crop_hit", "overlay_layers", layer_a=f"change_{a}_{b}", layer_b=f"cropland_{a}",
                 operation="intersect", name="cropland_affected")
        plan.add("change_lc", "zonal_stats", value_layer=f"landcover_{a}", zone_layer=f"change_{a}_{b}")
        plan.add("roads", "fetch_osm_features", _optional=True, feature_type="roads")
        plan.add("roads_hit", "features_in_zone", _optional=True, layer_id="osm_roads", zone_layer=f"change_{a}_{b}")
        return plan

    def summarize(self, p, r, bbox):
        c = r["change"]
        metrics = {"changed_km2": c["area_km2"], "built_up_affected_km2": _get(r, "built_hit", "area_km2"),
                   "cropland_affected_km2": _get(r, "crop_hit", "area_km2")}
        lines = [f"Between {p['pre_date']} and {p['post_date']}, {_km2(c['area_km2'])} changed "
                 f"({_pct(c['fraction'])} of the area; {c['method']}).",
                 f"Affected built-up land: {_km2(metrics['built_up_affected_km2'])}; "
                 f"affected cropland: {_km2(metrics['cropland_affected_km2'])}."]
        if "roads_hit" in r:
            lines.append(f"{r['roads_hit']['length_km_in_zone']:.1f} km of major road lie in changed areas.")
        figures = [_fig("Changed area", c["area_km2"], "km²", _pct(c["fraction"])),
                   _fig("Built-up affected", metrics["built_up_affected_km2"], "km²"),
                   _fig("Cropland affected", metrics["cropland_affected_km2"], "km²")]
        charts = []
        if r.get("change_lc", {}).get("class_fractions"):
            charts.append(_class_bar("Pre-event land cover of changed areas", r["change_lc"]["class_fractions"],
                                     c["area_km2"]))
        return lines, figures, metrics, charts, f"change_{p['pre_date']}_{p['post_date']}"


class Deforestation(Workflow):
    def __init__(self):
        super().__init__(
            "deforestation", "forest", "Deforestation monitoring",
            "Map forest lost between two dates, optionally near rivers, with terrain slope of cleared land.",
            [Param("start_date", "From", "date", "-3y"),
             Param("end_date", "To", "date", "-45d"),
             Param("near_water_m", "Only within this distance of rivers (m, 0 = whole area)", "number", 0, min=0, max=10000),
             Param("include_slope", "Report terrain slope of cleared areas", "boolean", True)],
            ["Forest loss map", "Forest cover change", "Slope of cleared land", "Land-cover comparison"],
            ["forest_loss_km2", "forest_loss_pct_of_forest", "forest_cover_pct_end"],
        )

    def build(self, p):
        a, b = p["start_date"], p["end_date"]
        plan = Plan(rationale=f"Workflow: deforestation {a} → {b}")
        plan.add("s1", "fetch_sentinel2_scene", date=a)
        plan.add("s2", "fetch_sentinel2_scene", date=b)
        plan.add("lc1", "classify_scene", scene_id=f"s2_{a}")
        plan.add("lc2", "classify_scene", scene_id=f"s2_{b}")
        plan.add("f1", "class_mask", layer_id=f"landcover_{a}", class_name="Forest")
        plan.add("f2", "class_mask", layer_id=f"landcover_{b}", class_name="Forest")
        plan.add("loss", "overlay_layers", layer_a=f"forest_{a}", layer_b=f"forest_{b}", operation="difference",
                 name="forest_loss")
        zone = "forest_loss"
        if p["near_water_m"] > 0:
            d = p["near_water_m"]
            plan.add("water", "fetch_osm_features", feature_type="water")
            plan.add("buffer", "buffer_features", layer_id="osm_water", distance_m=d)
            plan.add("near", "overlay_layers", layer_a="forest_loss", layer_b=f"buffer_osm_water_{d:g}m",
                     operation="intersect", name="forest_loss_near_water")
            zone = "forest_loss_near_water"
        if p["include_slope"]:
            plan.add("dem", "fetch_dem")
            plan.add("slope", "compute_slope")
            plan.add("slope_zone", "zonal_stats", value_layer="slope", zone_layer=zone)
        plan.zone = zone
        return plan

    def summarize(self, p, r, bbox):
        area = bbox_area_km2(bbox)
        loss, f1, f2 = r["loss"], r["f1"], r["f2"]
        metrics = {"forest_loss_km2": loss["area_km2"],
                   "forest_loss_pct_of_forest": round(100 * loss["fraction_of_layer_a"], 2),
                   "forest_cover_pct_end": round(100 * f2["fraction"], 2)}
        lines = [f"Forest lost {p['start_date']} → {p['end_date']}: {_km2(loss['area_km2'])} "
                 f"({_pct(loss['fraction_of_layer_a'])} of the starting forest).",
                 f"Forest cover went from {_pct(f1['fraction'])} to {_pct(f2['fraction'])} of the area."]
        figures = [_fig("Forest lost", loss["area_km2"], "km²", _pct(loss["fraction_of_layer_a"]) + " of forest"),
                   _fig("Forest cover (start)", round(100 * f1["fraction"], 1), "%"),
                   _fig("Forest cover (end)", round(100 * f2["fraction"], 1), "%")]
        if "near" in r:
            metrics["forest_loss_near_water_km2"] = r["near"]["area_km2"]
            lines.append(f"{_km2(r['near']['area_km2'])} of the loss is within {p['near_water_m']:g} m of rivers.")
            figures.append(_fig(f"Loss within {p['near_water_m']:g} m of rivers", r["near"]["area_km2"], "km²"))
        if "mean" in r.get("slope_zone", {}):
            z = r["slope_zone"]
            lines.append(f"Cleared land has a mean slope of {z['mean']:.1f}° (max {z['max']:.1f}°) — "
                         "steep clearings raise erosion and landslide risk.")
            figures.append(_fig("Mean slope of cleared land", z["mean"], "°"))
        charts = [_two_date_bar("Land cover comparison", r["lc1"]["class_fractions"], r["lc2"]["class_fractions"],
                                p["start_date"], p["end_date"], area)]
        return lines, figures, metrics, charts, r.get("near", loss)["layer_id"]


class VegetationTrend(Workflow):
    def __init__(self):
        super().__init__(
            "vegetation_trend", "forest", "Vegetation & forest trend",
            "Year-by-year vegetation health (NDVI) and forest cover, with a timelapse.",
            [Param("start_year", "First year", "integer", date.today().year - 5, min=2016, max=2100),
             Param("end_year", "Last year", "integer", date.today().year - 1, min=2016, max=2100),
             Param("month", "Month of each image (1-12)", "integer", 3, "Pick a dry-season month", min=1, max=12)],
            ["NDVI trend chart", "Forest cover trend", "Timelapse"],
            ["ndvi_change", "forest_cover_change_pct"],
        )

    def _dates(self, p):
        years = list(range(p["start_year"], p["end_year"] + 1))
        if len(years) < 2:
            raise ValueError("choose at least two years")
        if len(years) > 12:
            years = years[-12:]
        return [f"{y}-{p['month']:02d}-15" for y in years]

    def build(self, p):
        dates = self._dates(p)
        plan = Plan(rationale=f"Workflow: vegetation trend {dates[0][:4]}–{dates[-1][:4]}")
        plan.add("ndvi", "time_series", dates=dates, index="ndvi")
        plan.add("forest", "time_series", dates=dates, index="class_fraction", class_name="Forest")
        return plan

    def summarize(self, p, r, bbox):
        n, f = r["ndvi"], r["forest"]
        metrics = {"ndvi_change": n["change"], "forest_cover_change_pct": round(100 * f["change"], 2)}
        lines = [f"Mean NDVI changed by {n['change']:+.3f} ({n['relative_change_pct']:+.1f}%) from "
                 f"{n['series'][0]['date'][:4]} to {n['series'][-1]['date'][:4]}.",
                 f"Forest cover changed by {100 * f['change']:+.1f} percentage points."]
        if n["skipped"]:
            lines.append(f"{len(n['skipped'])} date(s) skipped for lack of clear imagery.")
        figures = [_fig("NDVI change", n["change"], "", f"{n['relative_change_pct']:+.1f}%"),
                   _fig("Forest cover change", round(100 * f["change"], 2), "pp")]
        x = [s["date"][:7] for s in n["series"]]
        charts = [{"type": "line", "title": "Vegetation health (mean NDVI)", "unit": "NDVI", "x": x,
                   "series": [{"name": "Mean NDVI", "values": [s["value"] for s in n["series"]]}]},
                  {"type": "line", "title": "Forest cover", "unit": "%", "x": [s["date"][:7] for s in f["series"]],
                   "series": [{"name": "Forest", "values": [round(100 * s["value"], 2) for s in f["series"]]}]}]
        return lines, figures, metrics, charts, f"ndvi_{n['series'][-1]['date']}"


class UrbanGrowth(Workflow):
    def __init__(self):
        super().__init__(
            "urban_growth", "urban", "Urban growth",
            "How much new built-up land appeared, and how much of it replaced green cover?",
            [Param("start_date", "From", "date", "-5y"), Param("end_date", "To", "date", "-45d")],
            ["New built-up map", "Built-up growth %", "Green cover lost to construction"],
            ["new_built_up_km2", "built_up_growth_pct", "green_lost_km2"],
        )

    def build(self, p):
        a, b = p["start_date"], p["end_date"]
        plan = Plan(rationale=f"Workflow: urban growth {a} → {b}")
        plan.add("s1", "fetch_sentinel2_scene", date=a)
        plan.add("s2", "fetch_sentinel2_scene", date=b)
        plan.add("lc1", "classify_scene", scene_id=f"s2_{a}")
        plan.add("lc2", "classify_scene", scene_id=f"s2_{b}")
        plan.add("b1", "class_mask", layer_id=f"landcover_{a}", group="built_up")
        plan.add("b2", "class_mask", layer_id=f"landcover_{b}", group="built_up")
        plan.add("new", "overlay_layers", layer_a=f"built_up_{b}", layer_b=f"built_up_{a}", operation="difference",
                 name="new_built_up")
        plan.add("veg", "class_mask", layer_id=f"landcover_{a}", group="vegetation")
        plan.add("green", "overlay_layers", layer_a="new_built_up", layer_b=f"vegetation_{a}", operation="intersect",
                 name="green_lost_to_construction")
        return plan

    def summarize(self, p, r, bbox):
        area = bbox_area_km2(bbox)
        b1, b2, new = r["b1"]["area_km2"], r["b2"]["area_km2"], r["new"]["area_km2"]
        growth = 100 * (b2 - b1) / b1 if b1 else None
        metrics = {"new_built_up_km2": new, "built_up_growth_pct": round(growth, 2) if growth is not None else None,
                   "green_lost_km2": r["green"]["area_km2"]}
        lines = [f"Built-up area went from {_km2(b1)} to {_km2(b2)}"
                 + (f" ({growth:+.1f}%)." if growth is not None else "."),
                 f"{_km2(new)} of land became built-up; {_km2(r['green']['area_km2'])} of that was "
                 "previously vegetation."]
        figures = [_fig("New built-up", new, "km²"),
                   _fig("Built-up growth", round(growth, 1) if growth is not None else "n/a", "%"),
                   _fig("Green cover lost", r["green"]["area_km2"], "km²")]
        charts = [_two_date_bar("Land cover comparison", r["lc1"]["class_fractions"], r["lc2"]["class_fractions"],
                                p["start_date"], p["end_date"], area)]
        return lines, figures, metrics, charts, "new_built_up"


class WaterEncroachment(Workflow):
    def __init__(self):
        super().__init__(
            "water_encroachment", "urban", "Lake & river encroachment",
            "Detect new construction on or beside water bodies, and water bodies that shrank.",
            [Param("start_date", "From", "date", "-5y"), Param("end_date", "To", "date", "-45d"),
             Param("buffer_m", "Protection zone around water (m)", "number", 100, min=0, max=2000)],
            ["Encroachment map", "Water body loss", "Construction in protection zone"],
            ["encroachment_km2", "water_loss_km2"],
        )

    def build(self, p):
        a, b, d = p["start_date"], p["end_date"], p["buffer_m"]
        plan = Plan(rationale=f"Workflow: water-body encroachment {a} → {b}, {d:g} m zone")
        plan.add("s1", "fetch_sentinel2_scene", date=a)
        plan.add("s2", "fetch_sentinel2_scene", date=b)
        plan.add("lc1", "classify_scene", scene_id=f"s2_{a}")
        plan.add("lc2", "classify_scene", scene_id=f"s2_{b}")
        plan.add("w1", "class_mask", layer_id=f"landcover_{a}", group="water")
        plan.add("w2", "class_mask", layer_id=f"landcover_{b}", group="water")
        plan.add("wloss", "overlay_layers", layer_a=f"water_{a}", layer_b=f"water_{b}", operation="difference",
                 name="water_body_loss")
        plan.add("zone", "buffer_features", layer_id=f"water_{a}", distance_m=d)
        plan.add("b1", "class_mask", layer_id=f"landcover_{a}", group="built_up")
        plan.add("b2", "class_mask", layer_id=f"landcover_{b}", group="built_up")
        plan.add("new", "overlay_layers", layer_a=f"built_up_{b}", layer_b=f"built_up_{a}", operation="difference",
                 name="new_built_up")
        plan.add("enc", "overlay_layers", layer_a="new_built_up", layer_b=f"buffer_water_{a}_{d:g}m",
                 operation="intersect", name="encroachment")
        return plan

    def summarize(self, p, r, bbox):
        metrics = {"encroachment_km2": r["enc"]["area_km2"], "water_loss_km2": r["wloss"]["area_km2"]}
        lines = [f"New construction within {p['buffer_m']:g} m of water bodies: {_km2(r['enc']['area_km2'])}.",
                 f"Water surface lost: {_km2(r['wloss']['area_km2'])} "
                 f"({_pct(r['wloss']['fraction_of_layer_a'])} of the original water area).",
                 "Verify flagged sites on the ground before enforcement action."]
        figures = [_fig("Encroachment", r["enc"]["area_km2"], "km²"),
                   _fig("Water surface lost", r["wloss"]["area_km2"], "km²"),
                   _fig("Water area (start)", r["w1"]["area_km2"], "km²"),
                   _fig("Water area (end)", r["w2"]["area_km2"], "km²")]
        charts = [{"type": "bar", "title": "Water and built-up area", "unit": "km²",
                   "categories": ["Water", "Built-up"],
                   "series": [{"name": p["start_date"], "values": [r["w1"]["area_km2"], r["b1"]["area_km2"]]},
                              {"name": p["end_date"], "values": [r["w2"]["area_km2"], r["b2"]["area_km2"]]}]}]
        return lines, figures, metrics, charts, "encroachment"


class CropHealth(Workflow):
    def __init__(self):
        super().__init__(
            "crop_health", "agriculture", "Crop health check",
            "Find stressed cropland (low NDVI) on a given date.",
            [Param("date", "Imagery date", "date", "-20d"),
             Param("stress_ndvi", "Stress threshold (NDVI below)", "number", 0.35, min=0, max=1)],
            ["Stressed cropland map", "Mean NDVI of cropland", "NDVI map"],
            ["stressed_cropland_km2", "stressed_cropland_pct", "cropland_mean_ndvi"],
        )

    def build(self, p):
        d, t = p["date"], p["stress_ndvi"]
        plan = Plan(rationale=f"Workflow: crop health on {d}, stress NDVI < {t:g}")
        plan.add("scene", "fetch_sentinel2_scene", date=d)
        plan.add("lc", "classify_scene", scene_id=f"s2_{d}")
        plan.add("crop", "class_mask", layer_id=f"landcover_{d}", group="cropland")
        plan.add("ndvi", "compute_ndvi", scene_id=f"s2_{d}")
        plan.add("crop_ndvi", "zonal_stats", value_layer=f"ndvi_{d}", zone_layer=f"cropland_{d}")
        plan.add("low", "threshold_layer", layer_id=f"ndvi_{d}", operator="lt", value=t)
        plan.add("stress", "overlay_layers", layer_a=f"cropland_{d}", layer_b=f"ndvi_{d}_lt_{t:g}",
                 operation="intersect", name="stressed_cropland")
        return plan

    def summarize(self, p, r, bbox):
        crop, stress = r["crop"], r["stress"]
        mean = r["crop_ndvi"].get("mean")
        metrics = {"stressed_cropland_km2": stress["area_km2"],
                   "stressed_cropland_pct": round(100 * stress["fraction_of_layer_a"], 2),
                   "cropland_mean_ndvi": mean}
        lines = [f"Cropland covers {_km2(crop['area_km2'])} ({_pct(crop['fraction'])} of the area)."]
        if mean is not None:
            lines.append(f"Mean cropland NDVI on {p['date']}: {mean:.2f}.")
        lines.append(f"Stressed cropland (NDVI < {p['stress_ndvi']:g}): {_km2(stress['area_km2'])} "
                     f"({_pct(stress['fraction_of_layer_a'])} of cropland). Fallow or freshly harvested fields "
                     "also show low NDVI — check the crop calendar.")
        figures = [_fig("Cropland", crop["area_km2"], "km²"),
                   _fig("Mean cropland NDVI", mean if mean is not None else "n/a"),
                   _fig("Stressed cropland", stress["area_km2"], "km²", _pct(stress["fraction_of_layer_a"]) + " of cropland")]
        charts = [_class_bar(f"Land cover {p['date']}", r["lc"]["class_fractions"], bbox_area_km2(bbox))]
        return lines, figures, metrics, charts, "stressed_cropland"


class CropSeason(Workflow):
    def __init__(self):
        super().__init__(
            "crop_season", "agriculture", "Crop season NDVI profile",
            "Month-by-month NDVI of cropland through a growing season (kharif / rabi), with a timelapse.",
            [Param("year", "Year", "integer", date.today().year, min=2016, max=2100),
             Param("start_month", "First month", "integer", 6, "June for kharif, November for rabi", min=1, max=12),
             Param("months", "Number of months", "integer", 5, min=2, max=12)],
            ["Seasonal NDVI curve for cropland", "Timelapse"],
            ["peak_ndvi", "season_ndvi_change"],
        )

    def _dates(self, p, today=None):
        today = today or date.today()
        out = []
        for i in range(p["months"]):
            y, m = divmod(p["start_month"] - 1 + i, 12)
            d = date(p["year"] + y, m + 1, 15)
            if d <= today:
                out.append(d.isoformat())
        if len(out) < 2:
            raise ValueError("choose a season with at least two past months")
        return out

    def build(self, p):
        dates = self._dates(p)
        first = dates[0]
        plan = Plan(rationale=f"Workflow: crop season NDVI {dates[0][:7]} → {dates[-1][:7]}")
        plan.add("scene", "fetch_sentinel2_scene", date=first)
        plan.add("lc", "classify_scene", scene_id=f"s2_{first}")
        plan.add("crop", "class_mask", layer_id=f"landcover_{first}", group="cropland")
        plan.add("series", "time_series", dates=dates, index="ndvi", zone_layer=f"cropland_{first}")
        return plan

    def summarize(self, p, r, bbox):
        s = r["series"]["series"]
        values = [x["value"] for x in s]
        peak = max(s, key=lambda x: x["value"])
        metrics = {"peak_ndvi": peak["value"], "season_ndvi_change": r["series"]["change"]}
        lines = [f"Cropland NDVI peaked at {peak['value']:.2f} in {peak['date'][:7]}.",
                 f"Change over the season: {r['series']['change']:+.3f}."]
        figures = [_fig("Peak NDVI", peak["value"], "", peak["date"][:7]),
                   _fig("Season change", r["series"]["change"])]
        charts = [{"type": "line", "title": "Cropland NDVI through the season", "unit": "NDVI",
                   "x": [x["date"][:7] for x in s], "series": [{"name": "Cropland mean NDVI", "values": values}]}]
        return lines, figures, metrics, charts, f"ndvi_{s[-1]['date']}"


WORKFLOWS = {w.id: w for w in [FloodRisk(), FloodScenarios(), DamageAssessment(), Deforestation(),
                                VegetationTrend(), UrbanGrowth(), WaterEncroachment(), CropHealth(), CropSeason()]}


def catalog() -> dict:
    return {"sectors": [{"id": k, **v, "workflows": [w.spec() for w in WORKFLOWS.values() if w.sector == k]}
                        for k, v in SECTORS.items()]}


def get(workflow_id: str) -> Workflow:
    if workflow_id not in WORKFLOWS:
        raise ValueError(f"unknown workflow '{workflow_id}'")
    return WORKFLOWS[workflow_id]


def instruction_text(wf: Workflow, params: dict) -> str:
    shown = ", ".join(f"{p.label}: {params[p.name]}" for p in wf.params)
    return f"{wf.title} ({shown})"
