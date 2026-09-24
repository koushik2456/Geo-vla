"""Sign-in, sign-up, profile and (admin) user management."""
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import config
from services import auth, db

router = APIRouter(tags=["accounts"])


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str = Field(..., max_length=40)
    password: str = Field(..., max_length=200)
    full_name: str | None = Field(None, max_length=120)
    organization: str | None = Field(None, max_length=160)
    email: str | None = Field(None, max_length=200)


class NewUser(SignupRequest):
    role: Literal["public", "official", "admin"] = "official"


class UserPatch(BaseModel):
    role: Literal["public", "official", "admin"] | None = None
    active: bool | None = None
    password: str | None = Field(None, max_length=200)
    full_name: str | None = None
    organization: str | None = None
    email: str | None = None


def _session(user: dict) -> dict:
    return {"token": auth.create_session(user["id"]), "user": user}


@router.post("/auth/login")
def login(req: LoginRequest):
    user = auth.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="wrong username or password")
    return _session(user)


@router.post("/auth/signup")
def signup(req: SignupRequest):
    if not config.ALLOW_PUBLIC_SIGNUP:
        raise HTTPException(status_code=403, detail="self sign-up is disabled; ask an administrator")
    try:
        user = auth.create_user(req.username, req.password, "public", req.full_name, req.organization, req.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session(user)


@router.post("/auth/logout")
def logout(authorization: str = Header(default="")):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        auth.end_session(token.strip())
    return {"ok": True}


@router.get("/auth/me")
def me(user=Depends(auth.optional_user)):
    return {"user": user, "signup_open": config.ALLOW_PUBLIC_SIGNUP}


# -- admin ----------------------------------------------------------------------------

@router.get("/admin/users")
def list_users(_=Depends(auth.require_role("admin"))):
    return [auth.public_user(u) for u in db.query("SELECT * FROM users ORDER BY created_at")]


@router.post("/admin/users")
def create_user(req: NewUser, _=Depends(auth.require_role("admin"))):
    try:
        return auth.create_user(req.username, req.password, req.role, req.full_name, req.organization, req.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/admin/users/{user_id}")
def update_user(user_id: int, req: UserPatch, admin=Depends(auth.require_role("admin"))):
    if not db.one("SELECT id FROM users WHERE id = ?", (user_id,)):
        raise HTTPException(status_code=404, detail="no such user")
    if user_id == admin["id"] and (req.role not in (None, "admin") or req.active is False):
        raise HTTPException(status_code=400, detail="you cannot demote or deactivate yourself")
    fields = {k: v for k, v in req.model_dump(exclude_none=True).items() if k != "password"}
    if "active" in fields:
        fields["active"] = int(fields["active"])
    if req.password:
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="password must be at least 8 characters")
        fields["password_hash"] = auth.hash_password(req.password)
    for key, value in fields.items():
        db.execute(f"UPDATE users SET {key} = ? WHERE id = ?", (value, user_id))  # keys come from the model
    if req.active is False or req.password:
        db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    return auth.public_user(db.one("SELECT * FROM users WHERE id = ?", (user_id,)))
