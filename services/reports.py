"""
services/reports.py — Official PDF report for an analysis.

Contents: title block, area and date, a map (imagery + result layer with
legend, coordinates, scale bar and north arrow), key figures, findings,
charts, the full method (every tool step, data sources, models), limitations,
and a sign-off block for the preparing and verifying officers.
"""
import io
import math
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import rendering  # noqa: E402
from geo_utils import bbox_size_m  # noqa: E402
from services import storage  # noqa: E402

# Validated categorical slots 1-2 (blue, orange) for chart series; text stays neutral.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#1c2430", "#5d6b7e", "#dde2e9"
ACCENT = colors.HexColor("#1f4e8c")

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Title"], fontSize=18, leading=22, alignment=0, textColor=ACCENT)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4,
                    textColor=ACCENT)
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor(MUTED))
WARN = ParagraphStyle("warn", parent=BODY, backColor=colors.HexColor("#fff4d6"), borderPadding=6,
                      textColor=colors.HexColor("#7a5a00"))


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# -- map figure -----------------------------------------------------------------------------

def _nice_scale(width_m: float) -> float:
    target = width_m / 4
    exp = 10 ** math.floor(math.log10(target))
    return max(s * exp for s in (1, 2, 5) if s * exp <= target)


def map_figure(run: dict, layer_ids: list = None) -> io.BytesIO:
    layers = {l["id"]: l for l in run["layers"]}
    bbox = run["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox
    extent = (min_lon, max_lon, min_lat, max_lat)
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * (max_lat - min_lat) / (max_lon - min_lon)
                                    / max(math.cos(math.radians((min_lat + max_lat) / 2)), 0.2)))
    scenes = [l for l in run["layers"] if l["kind"] == "scene"]
    if scenes:
        ax.imshow(storage.display_array(run["id"], scenes[-1])[::2, ::2], extent=extent, interpolation="bilinear")
    else:
        ax.set_facecolor("#eef1f4")
    chosen = layer_ids or [run["insights"].get("focus_layer")]
    handles = []
    for lid in chosen:
        layer = layers.get(lid)
        if not layer or layer["kind"] in ("scene", "vector"):
            continue
        data = rendering.downsample(storage.display_array(run["id"], layer), 900)
        ax.imshow(rendering.colorize(layer["kind"], data, layer["style"]), extent=extent, interpolation="nearest")
        legend = layer["legend"]
        if legend["type"] == "mask":
            handles.append(Patch(color=np.array(layer["style"]["color"]) / 255, label=layer["name"]))
        elif legend["type"] == "categorical":
            for name, css in legend["classes"].items():
                rgb = [int(c) / 255 for c in css[4:-1].split(",")]
                handles.append(Patch(color=rgb, label=name))
    for layer in run["layers"]:
        if layer["kind"] == "vector" and layer["id"] in chosen:
            for f in storage.load_geojson(run["id"], layer["id"])["features"]:
                coords = np.array(f["geometry"]["coordinates"] if f["geometry"]["type"] == "LineString"
                                  else f["geometry"]["coordinates"][0])
                ax.plot(coords[:, 0], coords[:, 1], color="#00b8d4", lw=1.2)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(1 / max(math.cos(math.radians((min_lat + max_lat) / 2)), 0.2))
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xlabel("Longitude (°E)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Latitude (°N)", fontsize=7.5, color=MUTED)
    # Scale bar + north arrow
    width_m, _ = bbox_size_m(bbox)
    bar_m = _nice_scale(width_m)
    bar_deg = bar_m / width_m * (max_lon - min_lon)
    x0, y0 = min_lon + 0.04 * (max_lon - min_lon), min_lat + 0.05 * (max_lat - min_lat)
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="white", lw=5, solid_capstyle="butt")
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + bar_deg / 2, y0 + 0.025 * (max_lat - min_lat),
            f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m",
            ha="center", fontsize=7.5, color="black", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.84), xycoords="axes fraction", ha="center",
                fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    if handles:
        ax.legend(handles=handles[:12], loc="lower right", fontsize=7, framealpha=0.9)
    return _png(fig)


# -- charts ----------------------------------------------------------------------------------------

def chart_figure(chart: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    series = chart["series"][:3]
    if chart["type"] == "bar":
        cats = chart["categories"]
        n = len(series)
        width = 0.8 / n
        x = np.arange(len(cats))
        for i, s in enumerate(series):
            ax.bar(x + (i - (n - 1) / 2) * width, s["values"], width * 0.92, color=SERIES[i], label=s["name"])
        ax.set_xticks(x, cats, rotation=25 if len(cats) > 4 else 0, ha="right" if len(cats) > 4 else "center")
    else:
        x = chart["x"]
        for i, s in enumerate(series):
            ax.plot(x, s["values"], color=SERIES[i], lw=2, marker="o", ms=4, label=s["name"])
        if chart.get("x_label"):
            ax.set_xlabel(chart["x_label"], fontsize=7.5, color=MUTED)
    ax.set_ylabel(chart.get("unit", ""), fontsize=7.5, color=MUTED)
    if len(series) > 1:
        ax.legend(fontsize=7.5, frameon=False)
    return _png(fig)


def _img(buf, max_w, max_h) -> Image:
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(buf).getSize()
    buf.seek(0)
    scale = min(max_w / w, max_h / h)
    return Image(buf, width=w * scale, height=h * scale)


def _table(rows, col_widths, header=True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _footer(canvas, doc, run):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, f"Geo-VLA analysis {run['id'][:8]} · generated "
                      f"{datetime.now().strftime('%d %b %Y %H:%M')}")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# -- report ------------------------------------------------------------------------------------------

def report_pdf(run: dict, layer_ids: list = None, author=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"Geo-VLA report — {run['title']}", author="Geo-VLA")
    ins = run["insights"] or {}
    width = A4[0] - 36 * mm
    story = [Paragraph("Geo-VLA · Geospatial Analysis Report", SMALL),
             Paragraph(_esc(run["title"]), H1)]

    bbox = run["bbox"]
    meta_rows = [
        ["Area", _esc(run.get("place") or "Custom area of interest")],
        ["Bounding box", f"{bbox[0]:.4f}, {bbox[1]:.4f} → {bbox[2]:.4f}, {bbox[3]:.4f} (lon/lat, WGS84)"],
        ["Area of interest", f"{ins.get('aoi_area_km2', 0):,.2f} km²"],
        ["Request", _esc(run["instruction"])],
        ["Analysis date", (run.get("finished_at") or run["created_at"])[:19].replace("T", " ") + " UTC"],
        ["Method", "Fixed sector workflow" if run["planner"] == "workflow" else
         (f"AI planner ({run['model']})" if run["planner"] == "claude" else "Rule-based planner")],
    ]
    if author:
        meta_rows.append(["Prepared for", _esc(" · ".join(x for x in (author.get("full_name") or author["username"],
                                                                          author.get("organization")) if x))])
    story.append(_table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(v, BODY)] for k, v in meta_rows],
                        [38 * mm, width - 38 * mm], header=False))
    if run.get("data_mode") == "synthetic":
        story += [Spacer(1, 6), Paragraph(
            "<b>Demonstration data.</b> This analysis used synthetic imagery (offline demo mode). "
            "The figures are illustrative and must not be used for decisions.", WARN)]

    story += [Paragraph("Map", H2), _img(map_figure(run, layer_ids), width, 150 * mm)]
    story.append(Paragraph("Background: Sentinel-2 true colour. Coordinates in WGS84.", SMALL))

    if ins.get("key_figures"):
        rows = [["Indicator", "Value", "Note"]] + [
            [Paragraph(_esc(f["label"]), BODY), Paragraph(f"<b>{_esc(_fmt(f['value']))}</b> {_esc(f.get('unit', ''))}", BODY),
             Paragraph(_esc(f.get("note", "")), SMALL)] for f in ins["key_figures"]]
        story.append(KeepTogether([Paragraph("Key figures", H2), _table(rows, [70 * mm, 45 * mm, width - 115 * mm])]))

    story += [Paragraph("Findings", H2)]
    for line in (run.get("answer") or "").split("\n"):
        if line.strip():
            story.append(Paragraph("• " + _esc(line.strip("_ ")), BODY))

    for chart in ins.get("charts", []):
        if chart["type"] in ("bar", "line") and chart.get("series"):
            story.append(KeepTogether([Paragraph(_esc(chart["title"]), H2),
                                       _img(chart_figure(chart), width, 80 * mm)]))
        elif chart["type"] == "table" and chart["rows"]:
            rows = [chart["columns"]] + [[Paragraph(_esc(_fmt(c)), BODY) for c in r] for r in chart["rows"]]
            story += [Paragraph(_esc(chart["title"]), H2), _table(rows, [width * 0.6, width * 0.2, width * 0.2])]

    story += [PageBreak(), Paragraph("Method and provenance", H2),
              Paragraph("Every number in this report was produced by the tool chain below, in order. "
                        "Inputs and outputs are recorded so the analysis can be audited and reproduced.", BODY),
              Spacer(1, 4)]
    steps = [s for s in run["trace"] if s.get("type") == "tool_call"]
    rows = [["#", "Operation", "Inputs", "Status"]]
    for i, s in enumerate(steps, 1):
        inputs = ", ".join(f"{k}={v}" for k, v in s["input"].items())
        rows.append([str(i), Paragraph(f"<font name='Courier'>{_esc(s['tool'])}</font>", SMALL),
                     Paragraph(_esc(inputs[:240]), SMALL),
                     Paragraph("failed" if s["is_error"] else f"ok ({s['duration_ms']} ms)", SMALL)])
    story.append(_table(rows, [8 * mm, 42 * mm, width - 80 * mm, 30 * mm]))
    if ins.get("sources"):
        story += [Paragraph("Data sources", H2)] + [Paragraph("• " + _esc(s), BODY) for s in ins["sources"]]
    if ins.get("methods"):
        story += [Paragraph("Models and algorithms", H2)] + [Paragraph("• " + _esc(m), BODY) for m in ins["methods"]]
    story += [Paragraph("Limitations", H2)] + [Paragraph("• " + t, BODY) for t in (
        "Sentinel-2 has 10 m pixels: features smaller than about 20 m (narrow roads, single houses) are not resolved.",
        "Land cover uses the 10 EuroSAT classes; mixed pixels at class boundaries may be misassigned.",
        "The flood model is a static 'bathtub' model on a 30 m DEM. It ignores flow, drainage, embankments "
        "and timing, so it indicates exposure, not a forecast.",
        "Clouds and haze can reduce accuracy; imagery is the least-cloudy acquisition in the search window.",
        "Findings are decision support. Verify critical sites on the ground before action.")]
    story += [Spacer(1, 18), _table([["Prepared by", "Verified by", "Date"], ["\n\n", "\n\n", "\n\n"],
                                     ["Name / designation", "Name / designation", ""]],
                                    [width / 3] * 3)]
    doc.build(story, onFirstPage=lambda c, d: _footer(c, d, run), onLaterPages=lambda c, d: _footer(c, d, run))
    return buf.getvalue()
