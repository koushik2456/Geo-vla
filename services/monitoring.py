"""
services/monitoring.py — Watched areas that re-run a workflow on a schedule and
raise alerts when a metric crosses a threshold.

Example: "Alert me if forest loss in Kaziranga over the last year exceeds
0.5 km², checked weekly." Workflow parameters keep their relative dates
("-1y", "-45d"), so every check analyses the latest period.

Alerts appear in the app (bell icon) and, when configured, by email (SMTP_*)
and webhook (ALERT_WEBHOOK_URL — e.g. a Slack/Teams incoming webhook).
"""
import logging
import smtplib
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import requests

import config
from services import db, runs, workflows

log = logging.getLogger("geo-vla.monitoring")

FREQUENCIES = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1), "monthly": timedelta(days=30)}
OPS = {"gt": (lambda a, b: a > b, ">"), "ge": (lambda a, b: a >= b, "≥"),
       "lt": (lambda a, b: a < b, "<"), "le": (lambda a, b: a <= b, "≤")}


def _decode(row: dict) -> dict:
    for k in ("params", "bbox", "rule"):
        row[k] = db.loads(row[k], {})
    row["active"] = bool(row["active"])
    return row


def validate(workflow: str, params: dict, frequency: str, rule: dict) -> None:
    wf = workflows.get(workflow)
    wf.build(wf.coerce(params))
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {list(FREQUENCIES)}")
    if rule.get("metric") not in wf.metrics:
        raise ValueError(f"metric must be one of {wf.metrics}")
    if rule.get("op") not in OPS:
        raise ValueError(f"op must be one of {list(OPS)}")
    float(rule.get("value"))


def create(owner: dict, name: str, workflow: str, params: dict, bbox: list, frequency: str, rule: dict,
           place: str = None, notify_email: str = None, run_now: bool = True) -> dict:
    validate(workflow, params, frequency, rule)
    next_run = datetime.now(timezone.utc) if run_now else datetime.now(timezone.utc) + FREQUENCIES[frequency]
    mid = db.execute(
        "INSERT INTO monitors (owner_id, name, workflow, params, bbox, place, frequency, rule, notify_email, "
        "next_run_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (owner["id"], name, workflow, db.dumps(params or {}), db.dumps(bbox), place, frequency,
         db.dumps({"metric": rule["metric"], "op": rule["op"], "value": float(rule["value"])}), notify_email,
         next_run.isoformat(timespec="seconds"), db.now()))
    return get(mid)


def get(monitor_id: int) -> dict:
    row = db.one("SELECT * FROM monitors WHERE id = ?", (monitor_id,))
    if not row:
        raise KeyError(monitor_id)
    return _decode(row)


def list_for(user: dict) -> list:
    rows = db.query("SELECT * FROM monitors WHERE owner_id = ? ORDER BY created_at DESC", (user["id"],))
    out = []
    for row in rows:
        row = _decode(row)
        last = db.one("SELECT status, insights, finished_at FROM runs WHERE id = ?", (row["last_run_id"],)) \
            if row["last_run_id"] else None
        row["last_value"] = db.loads(last["insights"], {}).get("metrics", {}).get(row["rule"]["metric"]) \
            if last and last["insights"] else None
        row["last_status"] = last["status"] if last else None
        row["alert_count"] = db.one("SELECT COUNT(*) AS n FROM alerts WHERE monitor_id = ?", (row["id"],))["n"]
        row["history"] = [
            {"run_id": r["id"], "date": r["finished_at"],
             "value": db.loads(r["insights"], {}).get("metrics", {}).get(row["rule"]["metric"])}
            for r in db.query("SELECT id, insights, finished_at FROM runs WHERE monitor_id = ? AND status = 'done' "
                              "ORDER BY created_at DESC LIMIT 20", (row["id"],))][::-1]
        out.append(row)
    return out


def rule_text(rule: dict) -> str:
    return f"{rule['metric']} {OPS[rule['op']][1]} {rule['value']:g}"


def check(monitor: dict) -> dict:
    """Run the monitor's workflow now, evaluate its rule, raise an alert if triggered."""
    owner = db.one("SELECT id, username, email FROM users WHERE id = ?", (monitor["owner_id"],))
    run = runs.create(monitor["bbox"], workflow=monitor["workflow"], params=monitor["params"], owner=owner,
                      place=monitor["place"], monitor_id=monitor["id"], title=f"{monitor['name']} (monitor)")
    run = runs.execute(run["id"])
    now = datetime.now(timezone.utc)
    db.execute("UPDATE monitors SET last_run_id = ?, last_checked_at = ?, next_run_at = ? WHERE id = ?",
               (run["id"], now.isoformat(timespec="seconds"),
                (now + FREQUENCIES[monitor["frequency"]]).isoformat(timespec="seconds"), monitor["id"]))
    rule = monitor["rule"]
    if run["status"] != "done":
        return _alert(monitor, run["id"], "error", f"{monitor['name']}: the check failed — {run['error']}", None)
    value = (run["insights"] or {}).get("metrics", {}).get(rule["metric"])
    if value is None:
        return _alert(monitor, run["id"], "error", f"{monitor['name']}: metric {rule['metric']} unavailable", None)
    if OPS[rule["op"]][0](value, rule["value"]):
        return _alert(monitor, run["id"], "alert",
                      f"{monitor['name']}: {rule['metric']} = {value:g} (rule: {rule_text(rule)})", value)
    return {"triggered": False, "value": value, "run_id": run["id"]}


def _alert(monitor, run_id, level, message, value) -> dict:
    aid = db.execute("INSERT INTO alerts (monitor_id, owner_id, run_id, level, message, value, created_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (monitor["id"], monitor["owner_id"], run_id, level, message, value, db.now()))
    notify(monitor, message, run_id)
    return {"triggered": level == "alert", "alert_id": aid, "value": value, "run_id": run_id, "message": message}


def notify(monitor: dict, message: str, run_id: str) -> None:
    if config.ALERT_WEBHOOK_URL:
        try:
            requests.post(config.ALERT_WEBHOOK_URL, timeout=10,
                          json={"text": f"[Geo-VLA] {message}", "monitor": monitor["name"], "run_id": run_id})
        except Exception as exc:
            log.warning("webhook failed: %s", exc)
    if config.SMTP_HOST and monitor.get("notify_email"):
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[Geo-VLA alert] {monitor['name']}"
            msg["From"], msg["To"] = config.SMTP_FROM, monitor["notify_email"]
            msg.set_content(f"{message}\n\nOpen the analysis in Geo-VLA (run {run_id}).")
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as smtp:
                smtp.starttls()
                if config.SMTP_USER:
                    smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
                smtp.send_message(msg)
        except Exception as exc:
            log.warning("alert email failed: %s", exc)


def due_monitors() -> list:
    return [_decode(r) for r in db.query("SELECT * FROM monitors WHERE active = 1 AND next_run_at <= ?", (db.now(),))]


# -- background scheduler ------------------------------------------------------------------------------

class Scheduler:
    def __init__(self, interval: int):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def tick(self) -> int:
        checked = 0
        for monitor in due_monitors():
            try:
                check(monitor)
                checked += 1
            except Exception:
                log.exception("monitor %s check crashed", monitor["id"])
        runs.cleanup_anonymous()
        return checked

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:
                log.exception("scheduler tick failed")

    def start(self):
        if self.interval > 0 and not self._thread:
            self._thread = threading.Thread(target=self._loop, name="monitor-scheduler", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
