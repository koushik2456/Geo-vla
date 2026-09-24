"""
services/auth.py — Accounts, roles and sessions (stdlib crypto only).

Roles:
  public   — run analyses, save their own projects, share links
  official — everything public can, plus area monitoring and alerts
  admin    — everything, plus user management and the training studio

Passwords are hashed with scrypt (random salt). Sessions are random bearer
tokens; only their SHA-256 is stored, so a leaked database cannot be used to
log in.
"""
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException

import config
from services import db

log = logging.getLogger("geo-vla.auth")

ROLES = ("public", "official", "admin")
_RANK = {r: i for i, r in enumerate(ROLES)}
_PUBLIC_FIELDS = ("id", "username", "full_name", "organization", "email", "role", "active", "created_at")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2 ** 14, r=8, p=1)
    return hmac.compare_digest(digest.hex(), digest_hex)


def public_user(row: dict) -> dict:
    return {k: row[k] for k in _PUBLIC_FIELDS if k in row}


def validate_new_user(username: str, password: str, role: str) -> None:
    if not (3 <= len(username) <= 40) or not username.replace("_", "").replace(".", "").replace("-", "").isalnum():
        raise ValueError("username must be 3-40 letters, digits, '.', '_' or '-'")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")


def create_user(username: str, password: str, role: str = "public", full_name: str = None,
                organization: str = None, email: str = None) -> dict:
    username = username.strip().lower()
    validate_new_user(username, password, role)
    if db.one("SELECT id FROM users WHERE username = ?", (username,)):
        raise ValueError("username already taken")
    uid = db.execute(
        "INSERT INTO users (username, full_name, organization, email, password_hash, role, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, full_name, organization, email, hash_password(password), role, db.now()),
    )
    return public_user(db.one("SELECT * FROM users WHERE id = ?", (uid,)))


def authenticate(username: str, password: str):
    row = db.one("SELECT * FROM users WHERE username = ? AND active = 1", (username.strip().lower(),))
    if row and verify_password(password, row["password_hash"]):
        return public_user(row)
    return None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=config.SESSION_DAYS)
    db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
               (_token_hash(token), user_id, expires.isoformat(timespec="seconds")))
    return token


def end_session(token: str) -> None:
    db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def user_for_token(token: str):
    row = db.one(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1",
        (_token_hash(token), db.now()),
    )
    return public_user(row) if row else None


def ensure_admin() -> None:
    """Create the first admin from ADMIN_USERNAME / ADMIN_PASSWORD if none exists."""
    if db.one("SELECT id FROM users WHERE role = 'admin'"):
        return
    if not config.ADMIN_PASSWORD:
        log.warning("no admin account exists — set ADMIN_PASSWORD to create one on startup")
        return
    create_user(config.ADMIN_USERNAME, config.ADMIN_PASSWORD, role="admin", full_name="Administrator")
    log.info("created admin account '%s'", config.ADMIN_USERNAME)


def has_role(user, role: str) -> bool:
    return bool(user) and _RANK[user["role"]] >= _RANK[role]


# -- FastAPI dependencies ------------------------------------------------------

def _bearer(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def optional_user(token: str = Depends(_bearer)):
    return user_for_token(token) if token else None


def current_user(user=Depends(optional_user)):
    if not user:
        raise HTTPException(status_code=401, detail="sign in required")
    return user


def require_role(role: str):
    def dependency(user=Depends(current_user)):
        if not has_role(user, role):
            raise HTTPException(status_code=403, detail=f"requires the '{role}' role")
        return user
    return dependency
