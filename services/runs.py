"""
services/runs.py — Analyses as persistent, background jobs.

A run is created instantly and executed on a worker thread; the UI polls it
and shows the reasoning trace as it grows. When finished, every layer is saved
to disk (for full-resolution tiles and exports) and the dashboard insights are
stored with the run.

Access: a run belongs to the user who created it. Runs made without signing
in have no owner and are readable by anyone holding their unguessable id; they
are deleted after ANON_RUN_TTL_HOURS unless saved. Owners can publish a
read-only share link (share_token) and revoke it.
"""
import logging
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import config
import geotools
from agent import GeoVLAAgent
from services import db, insights, storage, workflows
from services.auth import has_role

log = logging.getLogger("geo-vla.runs")

_executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_RUNS, thread_name_prefix="run")
_JSON_FIELDS = ("params", "bbox", "trace", "layers", "insights", "usage")


class RunError(Exception):
    pass


# -- creation ------------------------------------------------------------------------

def create(bbox, instruction: str = None, workflow: str = None, params: dict = None, owner=None,
           place: str = None, project_id: int = None, monitor_id: int = None, title: str = None) -> dict:
    if not instruction and not workflow:
        raise RunError("give an instruction or a workflow")
    if workflow:
        wf = workflows.get(workflow)
        params = wf.coerce(params)
        wf.build(params)  # validate early (e.g. date ranges) so the user sees errors immediately
        instruction = instruction or workflows.instruction_text(wf, params)
        title = title or wf.title
    run_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO runs (id, owner_id, project_id, monitor_id, title, instruction, workflow, params, bbox, place, "
        "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
        (run_id, owner["id"] if owner else None, project_id, monitor_id, title or instruction[:80], instruction,
         workflow, db.dumps(params or {}), db.dumps(bbox), place, db.now()),
    )
    return get_row(run_id)


def submit(run_id: str) -> None:
    _executor.submit(execute, run_id)


def background(fn, *args) -> None:
    """Run any job (e.g. a monitor check) on the shared worker pool."""
    _executor.submit(fn, *args)


# -- execution -------------------------------------------------------------------------

def execute(run_id: str) -> dict:
    row = get_row(run_id)
    db.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))
    lock = threading.Lock()

    def on_event(trace):
        with lock:
            db.execute("UPDATE runs SET trace = ? WHERE id = ?", (db.dumps(trace), run_id))

    agent = GeoVLAAgent(bbox=row["bbox"], on_event=on_event,
                        use_llm=False if row["workflow"] else None)
    try:
        figures = charts = metrics = focus = None
        if row["workflow"]:
            wf = workflows.get(row["workflow"])
            plan = wf.build(row["params"])
            out = agent.run_plan(plan)
            lines, figures, metrics, charts, focus = wf.summarize(row["params"], out["results"], row["bbox"])
            if out["failed"]:
                lines.append(f"Optional steps unavailable: {', '.join(out['failed'])}.")
            meta = {"planner": "workflow", "model": None, "usage": None}
            answer = "\n".join(lines)
        else:
            result = agent.answer(row["instruction"])
            answer = result["answer"]
            meta = {k: result.get(k) for k in ("planner", "model", "usage")}
        if config.data_mode() == "synthetic":
            answer += "\n_Data is synthetic (offline demo mode) — not for decisions._"

        layers = []
        for layer in agent.workspace.layers.values():
            storage.save_layer(run_id, layer)
            summary = geotools.layer_summary(layer)
            summary["style"] = {k: v for k, v in geotools.layer_style(layer).items() if k != "legend"}
            layers.append(summary)
        ins = insights.build(agent.trace, layers, row["bbox"], figures, charts, metrics, focus)
        db.execute(
            "UPDATE runs SET status = 'done', answer = ?, planner = ?, model = ?, usage = ?, data_mode = ?, "
            "trace = ?, layers = ?, insights = ?, finished_at = ? WHERE id = ?",
            (answer, meta["planner"], meta["model"], db.dumps(meta["usage"]), config.data_mode(),
             db.dumps(agent.trace), db.dumps(layers), db.dumps(ins), db.now(), run_id),
        )
    except Exception as exc:
        log.exception("run %s failed", run_id)
        detail = str(exc) if isinstance(exc, (geotools.ToolError, ValueError)) else f"{type(exc).__name__}: {exc}"
        db.execute("UPDATE runs SET status = 'failed', error = ?, trace = ?, finished_at = ? WHERE id = ?",
                   (detail, db.dumps(agent.trace), db.now(), run_id))
    return get_row(run_id)


# -- reading ---------------------------------------------------------------------------------

def _decode(row: dict) -> dict:
    for k in _JSON_FIELDS:
        row[k] = db.loads(row.get(k), [] if k in ("trace", "layers") else None)
    return row


def get_row(run_id: str) -> dict:
    row = db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not row:
        raise KeyError(run_id)
    return _decode(row)


def by_share_token(token: str) -> dict:
    row = db.one("SELECT * FROM runs WHERE share_token = ?", (token,))
    if not row:
        raise KeyError(token)
    return _decode(row)


def can_read(run: dict, user=None, share: str = None) -> bool:
    if run["owner_id"] is None:
        return True
    if share and run["share_token"] and secrets.compare_digest(share, run["share_token"]):
        return True
    return bool(user) and (user["id"] == run["owner_id"] or has_role(user, "admin"))


def can_write(run: dict, user) -> bool:
    return bool(user) and (user["id"] == run["owner_id"] or has_role(user, "admin"))


def public_view(run: dict, user=None) -> dict:
    """API representation; tile URLs are relative to the API base."""
    out = {k: run[k] for k in ("id", "title", "instruction", "workflow", "params", "bbox", "place", "status",
                               "error", "answer", "planner", "model", "data_mode", "trace", "insights",
                               "usage", "saved", "project_id", "monitor_id", "created_at", "finished_at")}
    out["layers"] = []
    for layer in run["layers"]:
        item = {k: v for k, v in layer.items() if k != "style"}
        if layer["kind"] == "vector":
            item["geojson_url"] = f"/runs/{run['id']}/layers/{layer['id']}.geojson"
        else:
            item["tiles"] = f"/runs/{run['id']}/tiles/{layer['id']}/{{z}}/{{x}}/{{y}}.png"
            item["preview_url"] = f"/runs/{run['id']}/layers/{layer['id']}/preview.png"
        out["layers"].append(item)
    out["can_edit"] = can_write(run, user)
    out["share_token"] = run["share_token"] if out["can_edit"] else None
    return out


def layer_record(run: dict, layer_id: str) -> dict:
    for layer in run["layers"]:
        if layer["id"] == layer_id:
            return layer
    raise KeyError(layer_id)


def list_for(user, project_id: int = None, saved: bool = None, limit: int = 100) -> list:
    sql = ("SELECT id, title, instruction, workflow, bbox, place, status, saved, project_id, monitor_id, "
           "created_at, finished_at, share_token FROM runs WHERE owner_id = ?")
    args = [user["id"]]
    if project_id is not None:
        sql += " AND project_id = ?"
        args.append(project_id)
    if saved is not None:
        sql += " AND saved = ?"
        args.append(int(saved))
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.query(sql, args)
    for r in rows:
        r["bbox"] = db.loads(r["bbox"])
        r["shared"] = bool(r.pop("share_token"))
    return rows


# -- editing ----------------------------------------------------------------------------------------

def update(run_id: str, title: str = None, project_id=..., saved: bool = None) -> None:
    if title is not None:
        db.execute("UPDATE runs SET title = ? WHERE id = ?", (title[:200], run_id))
    if project_id is not ...:
        db.execute("UPDATE runs SET project_id = ? WHERE id = ?", (project_id, run_id))
    if saved is not None:
        db.execute("UPDATE runs SET saved = ? WHERE id = ?", (int(saved), run_id))


def share(run_id: str, enable: bool) -> str:
    token = secrets.token_urlsafe(16) if enable else None
    db.execute("UPDATE runs SET share_token = ? WHERE id = ?", (token, run_id))
    return token


def delete(run_id: str) -> None:
    db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    storage.delete_run(run_id)


def cleanup_anonymous() -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.ANON_RUN_TTL_HOURS)).isoformat(timespec="seconds")
    stale = db.query("SELECT id FROM runs WHERE owner_id IS NULL AND saved = 0 AND created_at < ?", (cutoff,))
    for r in stale:
        delete(r["id"])
    return len(stale)
