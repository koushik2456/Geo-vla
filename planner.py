"""
planner.py — Offline, rule-based instruction decomposer.

Used when no ANTHROPIC_API_KEY is configured, and as a deterministic baseline
in the evaluation harness (eval/run_eval.py). It recognises a fixed set of
intents (change / deforestation, vegetation, land cover, proximity, terrain,
flooding) with keyword and regex rules and emits the same tool chain an LLM
would, so the UI, tools and trace can be exercised end to end.

It is deliberately simple: the point of Geo-VLA is that the LLM planner
generalises to compositions these rules cannot express.
"""
import re
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Step:
    key: str
    tool: str
    input: dict
    optional: bool = False   # failure is reported but does not stop the plan (e.g. OSM unavailable)


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    rationale: str = ""
    zone: str = None          # id of the final "focus" mask, if any
    dates: tuple = ()

    def add(self, key: str, tool: str, _optional: bool = False, **tool_input) -> None:
        self.steps.append(Step(key, tool, tool_input, _optional))


_FEATURES = {"river": "water", "stream": "water", "water": "water", "lake": "water",
             "road": "roads", "highway": "roads", "building": "buildings"}
_NEAR_RE = re.compile(
    r"\b(near|within|close to|along|around|next to|adjacent to|beside)\b[^.]*?\b"
    r"(rivers?|streams?|water|lakes?|roads?|highways?|buildings?)\b", re.I)
_DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(km|kilomet(?:er|re)s?|m|met(?:er|re)s?)\b", re.I)
_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YEAR_RE = re.compile(r"\b(20[0-3]\d|201\d)\b")
_LAST_N_RE = re.compile(r"(?:last|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?", re.I)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _has(text: str, *words) -> bool:
    return any(w in text for w in words)


class OfflinePlanner:
    def __init__(self, today: date = None):
        self.today = today or date.today()

    # -- parsing ------------------------------------------------------------

    def _recent(self) -> str:
        """Most recent complete summer (less cloud); falls back a year early in the season."""
        year = self.today.year if self.today.month >= 9 else self.today.year - 1
        return f"{year}-07-01"

    def _dates(self, text: str) -> tuple:
        iso = _ISO_RE.findall(text)
        if len(iso) >= 2:
            return tuple(sorted(iso[:2]))
        years = sorted(set(_YEAR_RE.findall(text)))
        if len(years) >= 2:
            return f"{years[0]}-07-01", f"{years[-1]}-07-01"
        t2 = iso[0] if iso else (f"{years[0]}-07-01" if years else self._recent())
        m = _LAST_N_RE.search(text)
        n = (_WORDS.get(m.group(1).lower()) or int(m.group(1))) if m else 3
        return f"{int(t2[:4]) - n}{t2[4:]}", t2

    @staticmethod
    def _distance_m(text: str, default: float = 1000.0) -> float:
        m = _DIST_RE.search(text)
        if not m:
            return default
        value, unit = float(m.group(1)), m.group(2).lower()
        return value * 1000 if unit.startswith("k") else value

    # -- planning -----------------------------------------------------------

    def plan(self, instruction: str) -> Plan:
        text = instruction.lower()
        plan = Plan()
        deforest = _has(text, "deforest", "forest loss", "lost forest", "tree loss", "clear-cut", "clearcut") or (
            "forest" in text and _has(text, "loss", "lost", "decreas", "change", "cleared"))
        change = deforest or _has(text, "change", "changed", "urban growth", "expansion", "compare", "difference", "increase")
        ndvi = _has(text, "ndvi", "vegetation", "greenness", "green cover")
        classify = _has(text, "land cover", "landcover", "land use", "land-use", "classif", "what kind", "what type")
        near = _NEAR_RE.search(text)
        flood = _has(text, "flood", "inundat", "submerge")
        slope = _has(text, "slope", "steep", "terrain", "gradient")
        elevation = _has(text, "elevation", "dem", "height", "altitude") and not slope

        if not any([change, ndvi, classify, near, flood, slope, elevation]):
            classify = ndvi = True
            plan.intents.append("overview")

        t1, t2 = self._dates(text)
        plan.dates = (t1, t2) if change else (t2,)

        # 1. Change / deforestation
        if change:
            plan.add("scene_1", "fetch_sentinel2_scene", date=t1)
            plan.add("scene_2", "fetch_sentinel2_scene", date=t2)
            if deforest:
                plan.intents.append("deforestation")
                plan.add("lc_1", "classify_scene", scene_id=f"s2_{t1}")
                plan.add("lc_2", "classify_scene", scene_id=f"s2_{t2}")
                plan.add("forest_1", "class_mask", layer_id=f"landcover_{t1}", class_name="Forest")
                plan.add("forest_2", "class_mask", layer_id=f"landcover_{t2}", class_name="Forest")
                plan.add("loss", "overlay_layers", layer_a=f"forest_{t1}", layer_b=f"forest_{t2}",
                         operation="difference", name="forest_loss")
                plan.zone = "forest_loss"
            else:
                plan.intents.append("change")
                plan.add("change", "detect_change", scene_id_1=f"s2_{t1}", scene_id_2=f"s2_{t2}")
                plan.zone = f"change_{t1}_{t2}"
        elif ndvi or classify:
            plan.add("scene_2", "fetch_sentinel2_scene", date=t2)

        # 2. Vegetation / land cover on the latest scene
        if ndvi:
            plan.intents.append("ndvi")
            plan.add("ndvi", "compute_ndvi", scene_id=f"s2_{t2}")
        if classify and not deforest:
            plan.intents.append("landcover")
            plan.add("lc_2", "classify_scene", scene_id=f"s2_{t2}")

        # 3. Proximity (buffer + overlay)
        if near:
            feature = next(v for k, v in _FEATURES.items() if k in near.group(2).lower())
            dist = self._distance_m(near.group(0))
            plan.intents.append(f"near_{feature}")
            plan.add("osm", "fetch_osm_features", feature_type=feature)
            plan.add("buffer", "buffer_features", layer_id=f"osm_{feature}", distance_m=dist)
            buffer_id = f"buffer_osm_{feature}_{dist:g}m"
            if plan.zone:
                zone = f"{plan.zone}_near_{feature}"
                plan.add("near", "overlay_layers", layer_a=plan.zone, layer_b=buffer_id,
                         operation="intersect", name=zone)
                plan.zone = zone
            else:
                plan.zone = buffer_id

        # 4. Terrain
        if slope or flood or elevation:
            plan.add("dem", "fetch_dem")
        if slope:
            plan.intents.append("slope")
            plan.add("slope", "compute_slope")
            if plan.zone:
                plan.add("slope_zone", "zonal_stats", value_layer="slope", zone_layer=plan.zone)
        if elevation:
            plan.intents.append("elevation")
            if plan.zone:
                plan.add("elev_zone", "zonal_stats", value_layer="dem", zone_layer=plan.zone)
        if flood:
            plan.intents.append("flood")
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", text)
            level = float(m.group(1)) if m else 5.0
            plan.add("flood", "flood_extent", water_level_m=level, mode="above_min")
            if any(s.key == "lc_2" for s in plan.steps):
                plan.add("flood_lc", "zonal_stats", value_layer=f"landcover_{t2}", zone_layer=f"flood_{level:g}m")

        if ndvi and plan.zone and plan.zone != f"ndvi_{t2}":
            plan.add("ndvi_zone", "zonal_stats", value_layer=f"ndvi_{t2}", zone_layer=plan.zone)

        plan.rationale = (
            "Offline planner (no ANTHROPIC_API_KEY): detected intents "
            f"[{', '.join(plan.intents)}] → {len(plan.steps)}-step tool chain: "
            + " → ".join(s.tool for s in plan.steps)
        )
        return plan

    # -- answer synthesis ---------------------------------------------------

    def summarize(self, plan: Plan, r: dict) -> str:
        pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
        inp = {step.key: step.input for step in plan.steps}
        lines = []
        if "loss" in r:
            t1, t2 = plan.dates
            lines.append(f"Forest loss {t1} → {t2}: {r['loss']['area_km2']} km² "
                         f"({pct(r['loss']['fraction_of_layer_a'])} of the {t1} forest; "
                         f"forest cover {pct(r['forest_1']['fraction'])} → {pct(r['forest_2']['fraction'])}).")
        if "change" in r:
            lines.append(f"Changed area {plan.dates[0]} → {plan.dates[1]}: {r['change']['area_km2']} km² "
                         f"({pct(r['change']['fraction'])} of the area; {r['change']['method']}).")
        if "near" in r:
            lines.append(f"Of that, {r['near']['area_km2']} km² ({pct(r['near']['fraction_of_layer_a'])}) "
                         f"lies within {inp['buffer']['distance_m']:g} m of {inp['osm']['feature_type']}.")
        elif "buffer" in r:
            lines.append(f"The proximity zone covers {r['buffer']['area_km2']} km² ({pct(r['buffer']['fraction'])}).")
        if "lc_2" in r and "loss" not in r:
            top = ", ".join(f"{k} {pct(v)}" for k, v in list(r["lc_2"]["class_fractions"].items())[:4])
            lines.append(f"Land cover: dominant class {r['lc_2']['dominant_class']} ({top}).")
        if "ndvi" in r:
            lines.append(f"Mean NDVI {r['ndvi']['mean']}; {pct(r['ndvi']['vegetated_fraction_ndvi_gt_0_4'])} "
                         "of pixels are vegetated (NDVI > 0.4).")
        if "ndvi_zone" in r and "mean" in r["ndvi_zone"]:
            lines.append(f"Mean NDVI inside the zone of interest: {r['ndvi_zone']['mean']}.")
        if "slope" in r:
            if "slope_zone" in r and "mean" in r["slope_zone"]:
                z = r["slope_zone"]
                lines.append(f"Terrain slope in those zones: mean {z['mean']}°, max {z['max']}° "
                             f"(area-wide mean {r['slope']['mean_slope_deg']}°).")
            else:
                lines.append(f"Terrain slope: mean {r['slope']['mean_slope_deg']}°, 90th percentile "
                             f"{r['slope']['p90_slope_deg']}°, {pct(r['slope']['steep_fraction_gt_15deg'])} steeper than 15°.")
        if "elev_zone" in r and "mean" in r["elev_zone"]:
            lines.append(f"Mean elevation in the zone: {r['elev_zone']['mean']} m.")
        elif "dem" in r and "elevation" in plan.intents:
            lines.append(f"Elevation ranges {r['dem']['min_m']}–{r['dem']['max_m']} m (mean {r['dem']['mean_m']} m).")
        if "flood" in r:
            f = r["flood"]
            lines.append(f"A {inp['flood']['water_level_m']:g} m rise "
                         f"(to {f['water_level_m_asl']} m a.s.l.) would flood {f['area_km2']} km² ({pct(f['fraction'])}).")
            if "flood_lc" in r and r["flood_lc"].get("class_fractions"):
                top = ", ".join(f"{k} {pct(v)}" for k, v in r["flood_lc"]["class_fractions"].items())
                lines.append(f"Flooded land cover: {top}.")
        sources = {o.get("source") for o in r.values() if isinstance(o, dict) and o.get("source")}
        if any("synthetic" in s for s in sources):
            lines.append("_Note: data is synthetic (offline demo mode)._")
        return "\n".join(lines) or "The plan ran but produced no summarisable results."
