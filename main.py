"""
main.py — FastAPI backend exposing the Geo-VLA agent to the web frontend.

Run with:
    uvicorn main:app --reload --port 8000

If frontend/dist exists (after `npm run build`), it is served at / so a
single container can host the whole demo.
"""
import logging
import os

import config  # loads .env before anything reads the environment

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import geotools
from agent import GeoVLAAgent
from geo_utils import validate_bbox

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("geo-vla.api")

app = FastAPI(title="Geo-VLA API", version="0.1.0",
              description="Tool-augmented vision-language-action agent for geospatial reasoning.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # set CORS_ORIGINS to your frontend origin before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty")
    agent = GeoVLAAgent(bbox=req.bbox)
    try:
        # Tools are CPU/network bound and synchronous; keep the event loop free.
        result = await run_in_threadpool(agent.run, req.instruction)
    except Exception as exc:
        if type(exc).__module__.startswith("anthropic"):
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
        raise
    return QueryResponse(**result)


@app.get("/health")
def health():
    ckpt = lambda name: os.path.exists(os.path.join(config.MODEL_CHECKPOINT_DIR, name))  # noqa: E731
    return {
        "status": "ok",
        "planner": "claude" if config.llm_enabled() else "offline",
        "model": config.MODEL_NAME if config.llm_enabled() else None,
        "data_mode": config.data_mode(),
        "checkpoints": {
            "classifier": ckpt("resnet50_eurosat.pth"),
            "change_detector": ckpt("siamese_unet_levircd.pth"),
        },
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
