from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user, oauth, role_for_groups, upsert_user
from ..config import settings
from ..db import get_db
from ..models import Role, Team, User
from ..schemas import TeamOut, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)):
    return user


@router.get("/login")
async def login(request: Request):
    if settings.auth_mode == "dev":
        return RedirectResponse("/")
    redirect_uri = f"{settings.public_url.rstrip('/')}/api/auth/callback"
    return await oauth.idp.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def callback(request: Request, db: AsyncSession = Depends(get_db)):
    if settings.auth_mode == "dev":
        return RedirectResponse("/")
    token = await oauth.idp.authorize_access_token(request)
    claims = token.get("userinfo") or {}
    email = claims.get("email")
    if not email:
        raise HTTPException(400, "identity provider did not return an email claim")
    groups = claims.get("groups") or claims.get("roles") or []
    if isinstance(groups, str):
        groups = [groups]
    user = await upsert_user(
        db, email=email, name=claims.get("name", ""), role=role_for_groups(groups),
        team_name=(claims.get("team") or ""),
    )
    request.session["user"] = {"email": user.email}
    return RedirectResponse("/")


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/teams", response_model=list[TeamOut])
async def teams(db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    rows = await db.execute(select(Team).order_by(Team.name))
    return list(rows.scalars())


@router.get("/users", response_model=list[UserOut])
async def users(db: AsyncSession = Depends(get_db), user: User = Depends(current_user)):
    if user.role != Role.admin:
        raise HTTPException(403, "admin only")
    rows = await db.execute(select(User).order_by(User.email))
    return list(rows.scalars())
