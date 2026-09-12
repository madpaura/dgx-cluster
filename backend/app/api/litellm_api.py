from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import current_user, require_deployer
from ..config import settings
from ..db import get_db
from ..models import ACTIVE_STATUSES, Deployment, DeployStatus, User
from ..services.litellm import LiteLLMClient, LiteLLMError, sync_deployment

router = APIRouter(prefix="/api/litellm", tags=["litellm"])


class TestRequest(BaseModel):
    model_name: str


@router.get("/status")
async def status(_: User = Depends(current_user)):
    client = LiteLLMClient()
    try:
        health = await client.health()
        groups = await client.groups() if health["reachable"] else []
        return {
            "base_url": settings.litellm_base_url,
            "auto_register": settings.litellm_auto_register,
            "reachable": health["reachable"],
            "detail": health["detail"],
            "groups": groups,
        }
    finally:
        await client.close()


@router.post("/test")
async def test_model(body: TestRequest, _: User = Depends(current_user)):
    """Send a real one-token request through the proxy. The fastest way to tell
    'the container is up' from 'the model actually answers'."""
    client = LiteLLMClient()
    try:
        return await client.test_model(body.model_name)
    finally:
        await client.close()


@router.post("/resync")
async def resync(db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    """Make LiteLLM match reality: register every healthy deployment that is
    missing, drop entries whose deployment is gone. Safe to run any time."""
    client = LiteLLMClient()
    added, removed, errors = [], [], []
    try:
        try:
            existing = await client.groups()
        except LiteLLMError as exc:
            raise HTTPException(502, str(exc)) from exc

        registered_ids = {
            m["deployment_id"] for g in existing for m in g["members"] if m.get("deployment_id")
        }
        rows = await db.execute(select(Deployment).where(Deployment.status == DeployStatus.healthy))
        healthy = list(rows.scalars().unique())
        healthy_ids = {d.id for d in healthy}

        for dep in healthy:
            if dep.id in registered_ids:
                dep.litellm_registered = True
                continue
            ok, msg = await sync_deployment(dep, register=True)
            dep.litellm_registered = ok
            (added if ok else errors).append(f"{dep.served_model_name}@{dep.node.name}" if ok else msg)

        for stale in registered_ids - healthy_ids:
            try:
                await client.deregister(stale)
                removed.append(stale)
            except LiteLLMError as exc:
                errors.append(str(exc))

        rows = await db.execute(select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES)))
        for dep in rows.scalars().unique():
            if dep.status != DeployStatus.healthy:
                dep.litellm_registered = False

        await audit.record(
            db, actor=user.email, action="litellm.resync", target_type="litellm", target_id="proxy",
            summary=f"resync: +{len(added)} -{len(removed)}",
            detail={"added": added, "removed": removed, "errors": errors}, ok=not errors,
        )
        await db.commit()
        return {"added": added, "removed": removed, "errors": errors}
    finally:
        await client.close()
