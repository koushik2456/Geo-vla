"""Analyses (runs), projects, share links, map tiles and downloads."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator

from geo_utils import validate_bbox
from services import auth, db, exports, runs, storage, tiles

router = APIRouter(tags=["analyses"])


class RunRequest(BaseModel):
    bbox: list[float]
    instruction: str | None = Field(None, max_length=2000)
    workflow: str | None = None
    params: dict | None = None
    place: str | None = Field(None, max_length=200)
    project_id: int | None = None
    title: str | None = Field(None, max_length=200)

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v):
        return validate_bbox(v)


class RunPatch(BaseModel):
    title: str | None = Field(None, max_length=200)
    project_id: int | None = None
    saved: bool | None = None
    clear_project: bool = False


class ProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(None, max_length=1000)


def _load(run_id: str, user, share: str = None) -> dict:
    try:
        run = runs.get_row(storage.check_id(run_id))
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="analysis not found")
    if not runs.can_read(run, user, share):
        raise HTTPException(status_code=404, detail="analysis not found")
    return run


def _owned(run_id: str, user) -> dict:
    run = _load(run_id, user)
    if not runs.can_write(run, user):
        raise HTTPException(status_code=403, detail="only the owner can change this analysis")
    return run


def _check_project(project_id, user):
    if project_id is not None and not db.one("SELECT id FROM projects WHERE id = ? AND owner_id = ?",
                                             (project_id, user["id"])):
        raise HTTPException(status_code=404, detail="project not found")


# -- runs -------------------------------------------------------------------------------------

@router.post("/runs", status_code=202)
def start_run(req: RunRequest, user=Depends(auth.optional_user)):
    if req.project_id is not None:
        if not user:
            raise HTTPException(status_code=401, detail="sign in to add analyses to a project")
        _check_project(req.project_id, user)
    if req.instruction is not None and not req.instruction.strip() and not req.workflow:
        raise HTTPException(status_code=400, detail="instruction must not be empty")
    try:
        run = runs.create(req.bbox, instruction=req.instruction, workflow=req.workflow, params=req.params,
                          owner=user, place=req.place, project_id=req.project_id, title=req.title)
    except (runs.RunError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runs.submit(run["id"])
    return {"id": run["id"], "status": run["status"]}


@router.get("/runs")
def list_runs(project_id: int | None = None, saved: bool | None = None, user=Depends(auth.current_user)):
    return runs.list_for(user, project_id, saved)


@router.get("/runs/{run_id}")
def get_run(run_id: str, share: str | None = None, user=Depends(auth.optional_user)):
    return runs.public_view(_load(run_id, user, share), user)


@router.get("/share/{token}")
def shared_run(token: str, user=Depends(auth.optional_user)):
    try:
        run = runs.by_share_token(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="this link is invalid or was revoked")
    return {**runs.public_view(run, user), "share": token}


@router.patch("/runs/{run_id}")
def update_run(run_id: str, req: RunPatch, user=Depends(auth.current_user)):
    run = _owned(run_id, user)
    if req.project_id is not None:
        _check_project(req.project_id, user)
    project = None if req.clear_project else (req.project_id if req.project_id is not None else ...)
    runs.update(run["id"], title=req.title, project_id=project, saved=req.saved)
    return runs.public_view(runs.get_row(run["id"]), user)


@router.post("/runs/{run_id}/share")
def share_run(run_id: str, user=Depends(auth.current_user)):
    run = _owned(run_id, user)
    runs.update(run["id"], saved=True)  # shared analyses must not expire
    return {"share_token": runs.share(run["id"], True)}


@router.delete("/runs/{run_id}/share")
def unshare_run(run_id: str, user=Depends(auth.current_user)):
    runs.share(_owned(run_id, user)["id"], False)
    return {"ok": True}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, user=Depends(auth.current_user)):
    runs.delete(_owned(run_id, user)["id"])
    return {"ok": True}


# -- tiles and downloads ------------------------------------------------------------------------

def _layer(run: dict, layer_id: str) -> dict:
    try:
        return runs.layer_record(run, layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="layer not found")


_TILE_HEADERS = {"Cache-Control": "private, max-age=86400"}


@router.get("/runs/{run_id}/tiles/{layer_id}/{z}/{x}/{y}.png")
def tile(run_id: str, layer_id: str, z: int, x: int, y: int, share: str | None = None,
         user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    layer = _layer(run, layer_id)
    if layer["kind"] == "vector":
        raise HTTPException(status_code=400, detail="vector layers are served as GeoJSON")
    try:
        png = tiles.render_tile(storage.display_array(run["id"], layer), layer["kind"], layer["style"],
                                layer["bbox"], z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(png, media_type="image/png", headers=_TILE_HEADERS)


@router.get("/runs/{run_id}/layers/{layer_id}/preview.png")
def preview(run_id: str, layer_id: str, share: str | None = None, user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    layer = _layer(run, layer_id)
    if layer["kind"] == "vector":
        raise HTTPException(status_code=400, detail="vector layers have no preview")
    return Response(exports.preview_png(run, layer), media_type="image/png", headers=_TILE_HEADERS)


@router.get("/runs/{run_id}/layers/{layer_id}.geojson")
def layer_geojson(run_id: str, layer_id: str, share: str | None = None, user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    layer = _layer(run, layer_id)
    try:
        body = exports.geojson_bytes(run, layer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(body, media_type="application/geo+json",
                    headers={"Content-Disposition": f'attachment; filename="{layer_id}.geojson"'})


@router.get("/runs/{run_id}/layers/{layer_id}.tif")
def layer_geotiff(run_id: str, layer_id: str, share: str | None = None, user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    layer = _layer(run, layer_id)
    try:
        body = exports.geotiff_bytes(run, layer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(body, media_type="image/tiff",
                    headers={"Content-Disposition": f'attachment; filename="{layer_id}.tif"'})


@router.get("/runs/{run_id}/report.pdf")
def report(run_id: str, share: str | None = None, layers: str | None = Query(None, description="comma-separated layer ids to map"),
           user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    if run["status"] != "done":
        raise HTTPException(status_code=409, detail="the analysis has not finished")
    chosen = [s for s in (layers or "").split(",") if s] or None
    body = exports.report_pdf(run, chosen, author=user)
    return Response(body, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="geo-vla-report-{run["id"][:8]}.pdf"'})


@router.get("/runs/{run_id}/export.zip")
def export_zip(run_id: str, share: str | None = None, user=Depends(auth.optional_user)):
    run = _load(run_id, user, share)
    if run["status"] != "done":
        raise HTTPException(status_code=409, detail="the analysis has not finished")
    return Response(exports.bundle_zip(run, author=user), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="geo-vla-{run["id"][:8]}.zip"'})


# -- projects -----------------------------------------------------------------------------------------

@router.get("/projects")
def list_projects(user=Depends(auth.current_user)):
    return db.query(
        "SELECT p.*, (SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS run_count "
        "FROM projects p WHERE owner_id = ? ORDER BY created_at DESC", (user["id"],))


@router.post("/projects")
def create_project(req: ProjectRequest, user=Depends(auth.current_user)):
    pid = db.execute("INSERT INTO projects (owner_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                     (user["id"], req.name, req.description, db.now()))
    return db.one("SELECT * FROM projects WHERE id = ?", (pid,))


@router.patch("/projects/{project_id}")
def rename_project(project_id: int, req: ProjectRequest, user=Depends(auth.current_user)):
    _check_project(project_id, user)
    db.execute("UPDATE projects SET name = ?, description = ? WHERE id = ?", (req.name, req.description, project_id))
    return db.one("SELECT * FROM projects WHERE id = ?", (project_id,))


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, user=Depends(auth.current_user)):
    _check_project(project_id, user)
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))  # runs keep existing, unassigned
    return {"ok": True}
