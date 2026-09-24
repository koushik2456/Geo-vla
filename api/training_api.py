"""Training studio + model registry (admin only)."""
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services import auth, training

router = APIRouter(tags=["training studio"])
admin = auth.require_role("admin")
MAX_UPLOAD_MB = 500


class JobRequest(BaseModel):
    model: str
    dataset: str
    params: dict = {}


@router.get("/training/datasets")
def datasets(_=Depends(admin)):
    return training.datasets()


@router.get("/training/jobs")
def jobs(_=Depends(admin)):
    return training.list_jobs()


@router.post("/training/jobs")
def start(req: JobRequest, user=Depends(admin)):
    try:
        return training.start_job(req.model, req.dataset, req.params, user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training/jobs/{job_id}")
def job(job_id: int, _=Depends(admin)):
    try:
        return training.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@router.post("/training/jobs/{job_id}/cancel")
def cancel(job_id: int, _=Depends(admin)):
    training.cancel_job(job_id)
    return {"ok": True}


@router.get("/models")
def model_versions(_=Depends(admin)):
    return training.versions()


@router.post("/models/{model}/versions/{version}/promote")
def promote(model: str, version: str, _=Depends(admin)):
    try:
        training.promote(model, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="version not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"checkpoint failed to load: {exc}") from exc
    return training.versions()[model]


@router.post("/models/{model}/deactivate")
def deactivate(model: str, _=Depends(admin)):
    if model not in training.MODELS:
        raise HTTPException(status_code=404, detail="unknown model")
    training.deactivate(model)
    return training.versions()[model]


@router.delete("/models/{model}/versions/{version}")
def delete_version(model: str, version: str, _=Depends(admin)):
    try:
        training.delete_version(model, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="version not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/models/{model}/upload")
async def upload(model: str, file: UploadFile = File(...), notes: str = Form(""),
                 metrics: UploadFile | None = File(None), _=Depends(admin)):
    """Register a checkpoint trained elsewhere (e.g. the Colab notebook), with its optional
    .metrics.json. The checkpoint must load into the model architecture."""
    import json
    if model not in training.MODELS:
        raise HTTPException(status_code=404, detail="unknown model")
    if not file.filename.endswith((".pth", ".pt")):
        raise HTTPException(status_code=400, detail="upload a .pth checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.pth")
        size = 0
        with open(path, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    raise HTTPException(status_code=413, detail=f"checkpoint larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        parsed = None
        if metrics is not None:
            try:
                parsed = json.loads(await metrics.read())
            except ValueError:
                raise HTTPException(status_code=400, detail="metrics file is not valid JSON")
        try:
            row = training.register_version(model, path, source="upload", metrics=parsed, notes=notes or None,
                                            validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"checkpoint does not match the {model} architecture: {exc}")
    return {"version": row["version"]}
