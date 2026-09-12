"""Authentication and roles.

Two modes. `dev` signs everyone in as an admin so you can run the whole thing
locally with no identity provider. `oidc` does a standard authorization-code
flow against your IdP and maps group claims onto roles.
"""
from __future__ import annotations

import logging

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_db
from .models import Role, Team, User

log = logging.getLogger(__name__)

DEV_EMAIL = "dev@localhost"
ROLE_RANK = {Role.viewer: 0, Role.deployer: 1, Role.admin: 2}

oauth = OAuth()
if settings.auth_mode == "oidc" and settings.oidc_issuer:
    oauth.register(
        name="idp",
        server_metadata_url=f"{settings.oidc_issuer.rstrip('/')}/.well-known/openid-configuration",
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        client_kwargs={"scope": "openid email profile groups"},
    )


def role_for_groups(groups: list[str]) -> Role:
    admin = {g.strip() for g in settings.oidc_admin_groups.split(",") if g.strip()}
    deployer = {g.strip() for g in settings.oidc_deployer_groups.split(",") if g.strip()}
    gset = set(groups or [])
    if gset & admin:
        return Role.admin
    if gset & deployer:
        return Role.deployer
    return Role.viewer


async def upsert_user(db: AsyncSession, *, email: str, name: str, role: Role, team_name: str = "") -> User:
    """Provision or refresh a user, safely against a concurrent first sign-in.

    Get-or-create is a race: several requests arriving together on a database
    that has never seen this person all find nothing and all insert. The unique
    index on email is what settles it — one wins, the losers look again. A
    browser opening the dashboard fires half a dozen parallel polls, so this is
    the ordinary case on a fresh deployment, not a corner case.
    """
    row = await db.execute(select(User).where(User.email == email))
    user = row.scalar_one_or_none()
    if user is None:
        db.add(User(email=email, name=name, role=role))
        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
        row = await db.execute(select(User).where(User.email == email))
        user = row.scalar_one()
    user.name = name or user.name
    user.role = role
    if team_name:
        trow = await db.execute(select(Team).where(Team.name == team_name))
        team = trow.scalar_one_or_none()
        if team is None:
            team = Team(name=team_name)
            db.add(team)
            await db.flush()
        user.team_id = team.id
    await db.commit()
    return user


async def current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    if settings.auth_mode == "dev":
        row = await db.execute(select(User).where(User.email == DEV_EMAIL))
        user = row.scalar_one_or_none()
        if user is None:
            user = await upsert_user(db, email=DEV_EMAIL, name="Local Admin", role=Role.admin)
        return user

    sess = request.session.get("user")
    if not sess:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not signed in")
    row = await db.execute(select(User).where(User.email == sess["email"]))
    user = row.scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer provisioned")
    return user


def require(min_role: Role):
    """Dependency factory: `user = Depends(require(Role.deployer))`."""

    async def _check(user: User = Depends(current_user)) -> User:
        if ROLE_RANK[user.role] < ROLE_RANK[min_role]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"requires {min_role.value}; you are {user.role.value}",
            )
        return user

    return _check


require_viewer = require(Role.viewer)
require_deployer = require(Role.deployer)
require_admin = require(Role.admin)
