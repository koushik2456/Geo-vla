"""Workflow catalog, place search, monitored areas and alerts."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from geo_utils import validate_bbox
from services import auth, db, geocode, monitoring, runs, workflows

router = APIRouter(tags=["workflows & monitoring"])


@router.get("/workflows")
def workflow_catalog():
    return workflows.catalog()


@router.get("/geocode")
def place_search(q: str):
    return geocode.search(q)


class Rule(BaseModel):
    metric: str
    op: Literal["gt", "ge", "lt", "le"]
    value: float


class MonitorRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    workflow: str
    params: dict = {}
    bbox: list[float]
    place: str | None = None
    frequency: Literal["daily", "weekly", "monthly"] = "weekly"
    rule: Rule
    notify_email: str | None = Field(None, max_length=200)
    run_now: bool = True

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v):
        return validate_bbox(v)


class MonitorPatch(BaseModel):
    name: str | None = Field(None, max_length=120)
    active: bool | None = None
    frequency: Literal["daily", "weekly", "monthly"] | None = None
    rule: Rule | None = None
    notify_email: str | None = None


official = auth.require_role("official")


def _mine(monitor_id: int, user) -> dict:
    try:
        m = monitoring.get(monitor_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="monitor not found")
    if m["owner_id"] != user["id"] and not auth.has_role(user, "admin"):
        raise HTTPException(status_code=404, detail="monitor not found")
    return m


@router.get("/monitors")
def list_monitors(user=Depends(official)):
    return monitoring.list_for(user)


@router.post("/monitors")
def create_monitor(req: MonitorRequest, user=Depends(official)):
    try:
        m = monitoring.create(user, req.name, req.workflow, req.params, req.bbox, req.frequency,
                              req.rule.model_dump(), req.place, req.notify_email, run_now=False)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.run_now:
        runs.background(monitoring.check, m)
    return m


@router.patch("/monitors/{monitor_id}")
def update_monitor(monitor_id: int, req: MonitorPatch, user=Depends(official)):
    m = _mine(monitor_id, user)
    if req.rule:
        try:
            monitoring.validate(m["workflow"], m["params"], req.frequency or m["frequency"], req.rule.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.execute("UPDATE monitors SET rule = ? WHERE id = ?", (db.dumps(req.rule.model_dump()), monitor_id))
    for key in ("name", "frequency", "notify_email"):
        value = getattr(req, key)
        if value is not None:
            db.execute(f"UPDATE monitors SET {key} = ? WHERE id = ?", (value, monitor_id))
    if req.active is not None:
        db.execute("UPDATE monitors SET active = ? WHERE id = ?", (int(req.active), monitor_id))
    return monitoring.get(monitor_id)


@router.delete("/monitors/{monitor_id}")
def delete_monitor(monitor_id: int, user=Depends(official)):
    _mine(monitor_id, user)
    db.execute("DELETE FROM monitors WHERE id = ?", (monitor_id,))
    return {"ok": True}


@router.post("/monitors/{monitor_id}/check", status_code=202)
def check_now(monitor_id: int, user=Depends(official)):
    runs.background(monitoring.check, _mine(monitor_id, user))
    return {"ok": True}


@router.get("/alerts")
def list_alerts(user=Depends(auth.current_user)):
    rows = db.query("SELECT a.*, m.name AS monitor_name FROM alerts a JOIN monitors m ON m.id = a.monitor_id "
                    "WHERE a.owner_id = ? ORDER BY a.created_at DESC LIMIT 100", (user["id"],))
    unread = db.one("SELECT COUNT(*) AS n FROM alerts WHERE owner_id = ? AND read = 0", (user["id"],))["n"]
    return {"unread": unread, "alerts": rows}


@router.post("/alerts/read")
def mark_read(user=Depends(auth.current_user)):
    db.execute("UPDATE alerts SET read = 1 WHERE owner_id = ?", (user["id"],))
    return {"ok": True}
