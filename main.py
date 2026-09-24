"""
main.py — FastAPI backend for Geo-VLA.

Run with:
    uvicorn main:app --reload --port 8000

Routers:
  api/auth_api.py        sign-in, sign-up, admin user management
  api/runs_api.py        analyses (background runs), projects, sharing, tiles, PDF/GeoTIFF/GeoJSON
  api/monitoring_api.py  sector workflows, place search, monitored areas, alerts
  api/training_api.py    training studio and model registry (admin)

If frontend/dist exists (after `npm run build`), it is served at / so a
single container can host the whole product.
"""
import logging
import os
from contextlib import asynccontextmanager

import config  # loads .env before anything reads the environment

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import geotools
from agent import GeoVLAAgent
from api import auth_api, monitoring_api, runs_api, training_api
from geo_utils import validate_bbox
from services import auth, db, monitoring, runs, training

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("geo-vla.api")

scheduler = monitoring.Scheduler(config.MONITOR_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(_app):
    db.init()
    auth.ensure_admin()
    training.recover_after_restart()
    db.execute("UPDATE runs SET status = 'failed', error = 'server restarted during the analysis' "
               "WHERE status IN ('queued', 'running')")
    runs.cleanup_anonymous()
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="Geo-VLA API", version="0.2.0", lifespan=lifespan,
              description="Tool-augmented vision-language-action platform for geospatial decision support.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # set CORS_ORIGINS to your frontend origin before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (auth_api.router, runs_api.router, monitoring_api.router, training_api.router):
    app.include_router(router)


# -- synchronous agent endpoint (scripts, evaluation, simple API clients) ----------------------------

class QueryRequest(BaseModel):
    instruction: str = Field(..., max_length=2000)
    bbox: list[float] | None = Field(None, description="[min_lon, min_lat, max_lon, max_lat]")

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v):
        return validate_bbox(v) if v is not None else v


class QueryResponse(BaseModel):
    answer: str
    trace: list
    layers: list
    planner: str
    model: str | None = None
    data_mode: str
    usage: dict | None = None


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run the agent and wait for the answer, with layers inlined as preview images.
    The web app uses POST /runs instead (background, live trace, full-resolution tiles)."""
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty")
    agent = GeoVLAAgent(bbox=req.bbox)
    try:
        result = await run_in_threadpool(agent.run, req.instruction)
    except Exception as exc:
        if type(exc).__module__.startswith("anthropic"):
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
        raise
    return QueryResponse(**result)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "planner": "claude" if config.llm_enabled() else "offline",
        "model": config.MODEL_NAME if config.llm_enabled() else None,
        "data_mode": config.data_mode(),
        "checkpoints": {
            "classifier": geotools.model_version("classifier"),
            "change_detector": geotools.model_version("change"),
        },
        "monitoring": config.MONITOR_INTERVAL_SEC > 0,
        "signup_open": config.ALLOW_PUBLIC_SIGNUP,
    }


@app.get("/tools")
def tools():
    """The tool schemas exactly as they are sent to the LLM."""
    return geotools.tool_schemas()


_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
