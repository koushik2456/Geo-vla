"""
config.py — Central runtime configuration, read from environment variables.

Geo-VLA degrades gracefully so the full pipeline can be demoed with no
credentials and no trained checkpoints:

  * No LLM key (Groq, OpenRouter, Together, Ollama, Anthropic, …) -> offline rule-based planner
  * No COPERNICUS_CLIENT_ID/SECRET or EE_PROJECT -> synthetic Sentinel-2, DEM and OSM data
  * No .pth checkpoints          -> classical fallbacks (spectral rules, CVA)

Every fallback is labelled in the tool output, so the reasoning trace always
says which path produced a number.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Reasoning layer (LLM planner)
# ---------------------------------------------------------------------------
# LLM_PROVIDER: auto | groq | openrouter | together | ollama | openai | anthropic | none
#   groq / openrouter / together / ollama / openai all speak the OpenAI chat-completions
#   API with function calling, so any open model they host (Llama, Qwen, Mistral,
#   DeepSeek, gpt-oss, …) can drive the tools. "openai" = any compatible server:
#   set LLM_BASE_URL (+ LLM_API_KEY). "auto" picks the first provider with a key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "").strip()   # e.g. http://localhost:11434/v1

PROVIDER_PRESETS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": lambda: GROQ_API_KEY,
             "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key": lambda: OPENROUTER_API_KEY,
                   "model": "meta-llama/llama-3.3-70b-instruct"},
    "together": {"base_url": "https://api.together.xyz/v1", "key": lambda: TOGETHER_API_KEY,
                 "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "ollama": {"base_url": OLLAMA_URL or "http://localhost:11434/v1", "key": lambda: "ollama" if OLLAMA_URL else "",
               "model": "qwen2.5:7b"},
    "openai": {"base_url": LLM_BASE_URL, "key": lambda: LLM_API_KEY or ("none" if LLM_BASE_URL else ""),
               "model": ""},
    "anthropic": {"base_url": None, "key": lambda: ANTHROPIC_API_KEY, "model": "claude-sonnet-5"},
}
MODEL_NAME = os.getenv("GEO_VLA_MODEL", "claude-sonnet-5")  # Anthropic model (kept for compatibility)
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


def llm_settings():
    """Resolved LLM connection: {provider, base_url, api_key, model} or None (offline planner)."""
    if OFFLINE or LLM_PROVIDER == "none":
        return None
    order = [LLM_PROVIDER] if LLM_PROVIDER != "auto" else ["groq", "openrouter", "together", "openai", "ollama", "anthropic"]
    for name in order:
        preset = PROVIDER_PRESETS.get(name)
        if not preset:
            continue
        key = preset["key"]()
        if not key:
            continue
        model = LLM_MODEL or (MODEL_NAME if name == "anthropic" else preset["model"])
        if not model:
            continue
        return {"provider": name, "base_url": LLM_BASE_URL or preset["base_url"], "api_key": key, "model": model}
    return None


def llm_enabled() -> bool:
    return llm_settings() is not None


# Google Earth Engine (optional imagery source). Authenticate once with
# `earthengine authenticate` (personal account) or set a service account + JSON key.
EE_PROJECT = os.getenv("EE_PROJECT", "").strip()
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "").strip()
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
# IMAGERY_SOURCE: auto | copernicus | earthengine ("auto" prefers Copernicus when both are set)
_IMAGERY_SOURCE = os.getenv("IMAGERY_SOURCE", "auto").strip().lower()

# Google Maps Platform (optional): Photorealistic 3D Tiles on the globe (browser,
# Map Tiles API) and accurate geocoding / reverse geocoding (server, Geocoding API).
# GOOGLE_GEOCODING_KEY defaults to the same key; use a separate unrestricted-by-referrer
# key there if the browser key is restricted to your domain.
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
GOOGLE_GEOCODING_KEY = os.getenv("GOOGLE_GEOCODING_KEY", "").strip() or GOOGLE_MAPS_API_KEY
CESIUM_ION_TOKEN = os.getenv("CESIUM_ION_TOKEN", "").strip()


def copernicus_configured() -> bool:
    return bool(COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET)


def earthengine_configured() -> bool:
    return bool(EE_PROJECT)


def imagery_source() -> str:
    """Where live Sentinel-2 comes from: "copernicus" or "earthengine"."""
    if _IMAGERY_SOURCE == "earthengine" and earthengine_configured():
        return "earthengine"
    if _IMAGERY_SOURCE == "copernicus" or copernicus_configured():
        return "copernicus"
    return "earthengine" if earthengine_configured() else "copernicus"


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
    return "live" if (copernicus_configured() or earthengine_configured()) else "synthetic"


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
NOMINATIM_REVERSE_URL = os.getenv("NOMINATIM_REVERSE_URL", "https://nominatim.openstreetmap.org/reverse")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
