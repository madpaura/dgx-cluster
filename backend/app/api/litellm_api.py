from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import current_user, require_deployer
from ..config import settings
from ..db import get_db
from ..models import User
from ..services.litellm import LiteLLMClient, LiteLLMError, reconcile

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
            "console_url": _console_url(),
            "auto_register": settings.litellm_auto_register,
            "reachable": health["reachable"],
            "detail": health["detail"],
            "groups": groups,
        }
    finally:
        await client.close()


@router.get("/metrics")
async def metrics(_: User = Depends(current_user)):
    """What the proxy itself reports: which backends it can reach, and the
    traffic that has gone through it.

    Kept apart from /status because probing every backend takes as long as the
    slowest one, and the status card should not wait on that.
    """
    client = LiteLLMClient()
    try:
        try:
            backends = await client.backend_health()
        except LiteLLMError as exc:
            return {"reachable": False, "error": str(exc), "backends": None, "traffic": None}
        return {
            "reachable": True,
            "console_url": _console_url(),
            "backends": backends,
            "traffic": await client.recent_traffic(),
        }
    finally:
        await client.close()


def _console_url() -> str:
    """Where to send a browser. Falls back to the internal address, which is
    right when dgxctl and the proxy are not in separate containers."""
    base = (settings.litellm_public_url or settings.litellm_base_url).rstrip("/")
    return f"{base}/ui/"


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
    missing, drop entries whose deployment is gone. Safe to run any time.

    The worker does this on a timer; this is the same code path, on demand.
    """
    result = await reconcile(db)
    await audit.record(
        db, actor=user.email, action="litellm.resync", target_type="litellm", target_id="proxy",
        summary=f"resync: +{len(result['added'])} -{len(result['removed'])}",
        detail=result, ok=not result["errors"],
    )
    await db.commit()
    return result
