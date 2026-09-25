"""Globe explorer: client map configuration, place details at a point, click-to-classify."""
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
import geotools
from services import explorer

router = APIRouter(tags=["explore"])
_classify_slots = threading.Semaphore(2)  # CPU-heavy: at most two ResNet explanations at once


@router.get("/client-config")
def client_config():
    """Browser-side map keys. GOOGLE_MAPS_API_KEY is a browser key by design: restrict it by
    HTTP referrer and to the Map Tiles API in the Google Cloud console."""
    return {
        "google_maps_key": config.GOOGLE_MAPS_API_KEY or None,
        "cesium_ion_token": config.CESIUM_ION_TOKEN or None,
        "geocoder": "google" if config.GOOGLE_GEOCODING_KEY and not config.OFFLINE
        else ("nominatim" if not config.OFFLINE else "gazetteer"),
        "data_mode": config.data_mode(),
        "imagery_source": config.imagery_source() if config.data_live() else "synthetic",
    }


@router.get("/explore/point")
def explore_point(lon: float, lat: float):
    try:
        return explorer.point_info(lon, lat)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class ClassifyRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-85, le=85)
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


@router.post("/explore/classify")
def explore_classify(req: ClassifyRequest):
    if not _classify_slots.acquire(timeout=60):
        raise HTTPException(503, "the explorer is busy; try again in a moment")
    try:
        return explorer.classify_point(req.lon, req.lat, req.date)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except geotools.ToolError as exc:
        raise HTTPException(422, str(exc))
    finally:
        _classify_slots.release()
