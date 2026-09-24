"""
config.py — Central runtime configuration, read from environment variables.

Geo-VLA degrades gracefully so the full pipeline can be demoed with no
credentials and no trained checkpoints:

  * No ANTHROPIC_API_KEY        -> offline rule-based planner instead of Claude
  * No COPERNICUS_CLIENT_ID/SECRET -> synthetic Sentinel-2, DEM and OSM data
  * No .pth checkpoints          -> classical fallbacks (spectral rules, CVA)

Every fallback is labelled in the tool output, so the reasoning trace always
says which path produced a number.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEO_VLA_MODEL", "claude-sonnet-5")
# "summarized" surfaces Claude's reasoning summaries in the trace panel; "off" omits them.
THINKING_DISPLAY = os.getenv("GEO_VLA_THINKING", "summarized")
MAX_AGENT_TURNS = int(os.getenv("GEO_VLA_MAX_TURNS", "12"))

COPERNICUS_CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
COPERNICUS_CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()

# Force every data source to synthetic (no network at all) — used by tests/CI.
OFFLINE = _flag("GEO_VLA_OFFLINE")

MODEL_CHECKPOINT_DIR = os.getenv("MODEL_CHECKPOINT_DIR", "./models/checkpoints")
DEM_TILE_CACHE_DIR = os.getenv("DEM_TILE_CACHE_DIR", "./data/dem_cache")
IMAGERY_CACHE_DIR = os.getenv("IMAGERY_CACHE_DIR", "./data/imagery_cache")

PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]


def llm_enabled() -> bool:
    return bool(ANTHROPIC_API_KEY)


# Data mode: "live" (Sentinel-2 + Copernicus DEM + OpenStreetMap over the network)
# or "synthetic" (deterministic offline world from synthetic.py). All three
# sources switch together so layers from different tools stay spatially consistent.
# "auto" = live when Copernicus credentials are present.
_DATA_MODE = os.getenv("GEO_VLA_DATA_MODE", "auto").strip().lower()


def data_mode() -> str:
    if OFFLINE:
        return "synthetic"
    if _DATA_MODE in {"live", "synthetic"}:
        return _DATA_MODE
    return "live" if (COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET) else "synthetic"


def data_live() -> bool:
    return data_mode() == "live"


# ---------------------------------------------------------------------------
# Product features: storage, accounts, monitoring, notifications
# ---------------------------------------------------------------------------

DATA_DIR = os.getenv("GEO_VLA_DATA_DIR", "./data")
DB_PATH = os.getenv("GEO_VLA_DB", os.path.join(DATA_DIR, "geovla.db"))
RUNS_DIR = os.path.join(DATA_DIR, "runs")
MODEL_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", "./models/registry")
TRAINING_JOBS_DIR = os.path.join(DATA_DIR, "training_jobs")

# First admin account, created on startup if no admin exists yet.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ALLOW_PUBLIC_SIGNUP = _flag("ALLOW_PUBLIC_SIGNUP", "1")
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "14"))
ANON_RUN_TTL_HOURS = int(os.getenv("ANON_RUN_TTL_HOURS", "24"))
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "4"))

# Monitoring scheduler (seconds between checks for due monitors; 0 disables).
MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "300"))

# Optional alert delivery.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "geo-vla@localhost")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# Place search (OpenStreetMap Nominatim; bundled gazetteer when offline).
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
