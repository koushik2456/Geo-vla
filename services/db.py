"""
services/db.py — SQLite storage (stdlib only).

One connection per call keeps it safe across FastAPI worker threads, the
background run executor and the monitoring scheduler. WAL mode lets readers
(the UI polling a run) proceed while a run writes progress.
"""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT,
    organization TEXT,
    email TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'official', 'public')),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    monitor_id INTEGER REFERENCES monitors(id) ON DELETE SET NULL,
    title TEXT,
    instruction TEXT NOT NULL,
    workflow TEXT,
    params TEXT,
    bbox TEXT NOT NULL,
    place TEXT,
    status TEXT NOT NULL,
    error TEXT,
    answer TEXT,
    planner TEXT,
    model TEXT,
    data_mode TEXT,
    trace TEXT NOT NULL DEFAULT '[]',
    layers TEXT NOT NULL DEFAULT '[]',
    insights TEXT,
    usage TEXT,
    saved INTEGER NOT NULL DEFAULT 0,
    share_token TEXT UNIQUE,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS runs_owner ON runs(owner_id, created_at);
CREATE TABLE IF NOT EXISTS monitors (
    id INTEGER PRIMARY KEY,
    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    workflow TEXT NOT NULL,
    params TEXT NOT NULL,
    bbox TEXT NOT NULL,
    place TEXT,
    frequency TEXT NOT NULL,
    rule TEXT NOT NULL,
    notify_email TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    last_run_id TEXT,
    last_checked_at TEXT,
    next_run_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    monitor_id INTEGER NOT NULL REFERENCES monitors(id) ON DELETE CASCADE,
    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id TEXT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    value REAL,
    read INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    dataset TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    pid INTEGER,
    version TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    dataset TEXT,
    metrics TEXT,
    source TEXT NOT NULL,
    job_id INTEGER REFERENCES training_jobs(id) ON DELETE SET NULL,
    notes TEXT,
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (model, version)
);
"""

_init_lock = threading.Lock()
_initialized_path = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init() -> None:
    global _initialized_path
    with _init_lock:
        if _initialized_path == config.DB_PATH:
            return
        os.makedirs(os.path.dirname(os.path.abspath(config.DB_PATH)), exist_ok=True)
        conn = _connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _initialized_path = config.DB_PATH


@contextmanager
def connection():
    init()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params=()) -> list:
    with connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(sql: str, params=()):
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=()) -> int:
    """Run a write; returns lastrowid (inserts) or rowcount (updates)."""
    with connection() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount


def dumps(value) -> str:
    return json.dumps(value, default=str)


def loads(value, default=None):
    return json.loads(value) if value else default
