"""
services/insights.py — Dashboard content (key figures, charts, tables) for a run.

Workflows supply their own figures and charts; for free-text agent runs we
derive charts generically from the tool outputs in the trace, so every run
gets a dashboard.
"""
from geo_utils import bbox_area_km2


def _class_chart(title, fractions, area):
    items = sorted(fractions.items(), key=lambda kv: -kv[1])
    return {"type": "bar", "title": title, "unit": "km²", "categories": [k for k, _ in items],
            "series": [{"name": "Area", "values": [round(v * area, 3) for _, v in items]}]}


def derive_charts(trace: list, bbox) -> list:
    area = bbox_area_km2(bbox)
    charts, classified = [], []
    for step in trace:
        if step.get("type") != "tool_call" or step.get("is_error"):
            continue
        out, tool = step["output"], step["tool"]
        if tool == "classify_scene":
            classified.append((out["layer_id"].removeprefix("landcover_"), out["class_fractions"]))
        elif tool == "time_series":
            charts.append({"type": "line", "title": f"Trend: {out['indicator']}", "unit": "",
                           "x": [s["date"][:7] for s in out["series"]],
                           "series": [{"name": out["indicator"], "values": [s["value"] for s in out["series"]]}]})
        elif tool == "zonal_stats" and out.get("class_fractions"):
            charts.append(_class_chart(f"Land cover inside {step['input'].get('zone_layer')}",
                                       out["class_fractions"], out.get("area_km2", area)))
    if len(classified) >= 2:
        (d1, a), (d2, b) = classified[0], classified[-1]
        classes = sorted(set(a) | set(b), key=lambda c: -(a.get(c, 0) + b.get(c, 0)))
        charts.insert(0, {"type": "bar", "title": "Land cover comparison", "unit": "km²", "categories": classes,
                          "series": [{"name": d1, "values": [round(a.get(c, 0) * area, 3) for c in classes]},
                                     {"name": d2, "values": [round(b.get(c, 0) * area, 3) for c in classes]}]})
    elif classified:
        d, f = classified[0]
        charts.insert(0, _class_chart(f"Land cover {d}", f, area))
    return charts


def area_table(layers: list) -> dict:
    rows = [[l["name"], l["stats"]["area_km2"], round(100 * l["stats"]["fraction"], 2)]
            for l in layers if l["kind"] == "mask" and "stats" in l]
    return {"type": "table", "title": "Mapped areas", "columns": ["Layer", "Area (km²)", "% of AOI"], "rows": rows}


def build(trace: list, layers: list, bbox, figures=None, charts=None, metrics=None, focus_layer=None) -> dict:
    charts = list(charts) if charts else derive_charts(trace, bbox)
    table = area_table(layers)
    if table["rows"]:
        charts.append(table)
    if focus_layer is None:
        masks = [l["id"] for l in layers if l["kind"] == "mask"]
        focus_layer = masks[-1] if masks else (layers[-1]["id"] if layers else None)
    tool_calls = [s for s in trace if s.get("type") == "tool_call"]
    return {
        "key_figures": figures or [],
        "metrics": metrics or {},
        "charts": charts,
        "focus_layer": focus_layer,
        "aoi_area_km2": round(bbox_area_km2(bbox), 3),
        "steps": len(tool_calls),
        "failed_steps": sum(1 for s in tool_calls if s["is_error"]),
        "sources": sorted({s["output"].get("source") for s in tool_calls
                           if isinstance(s.get("output"), dict) and s["output"].get("source")}),
        "methods": sorted({s["output"].get("method") for s in tool_calls
                           if isinstance(s.get("output"), dict) and s["output"].get("method")}),
    }
