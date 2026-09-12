from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import events
from ..auth import DEV_EMAIL, current_user, upsert_user
from ..config import settings
from ..db import get_db
from ..models import ACTIVE_STATUSES, AuditLog, Deployment, DeployStatus, Event, Node, NodeStatus, Role, User
from ..schemas import AuditOut, EventOut, FleetSummary
from ..services.litellm import LiteLLMClient
from ..services.placement import busy_gpu_indices

router = APIRouter(prefix="/api", tags=["fleet"])

# A deployment that died days ago and was never cleaned up is history, not a
# problem. The summary is a "right now" indicator, so it counts only failures
# recent enough to still be worth acting on. The full record stays available
# through /api/deployments?active_only=false and the activity feed.
RECENT_FAILURE_WINDOW = timedelta(hours=1)

# The dashboard polls /api/summary every few seconds per open tab; asking
# LiteLLM fresh on each of those would mean a slow or hung proxy stalls the
# whole dashboard for everyone watching it. A stale-by-seconds reachability
# answer is fine, a blocked poll loop is not.
_litellm_cache: dict[str, object] = {"checked_at": 0.0, "reachable": False}


async def _litellm_reachable() -> bool:
    now = asyncio.get_running_loop().time()
    if now - _litellm_cache["checked_at"] < settings.summary_cache_seconds:
        return bool(_litellm_cache["reachable"])
    client = LiteLLMClient()
    try:
        reachable = (await client.health())["reachable"]
    finally:
        await client.close()
    _litellm_cache["checked_at"] = now
    _litellm_cache["reachable"] = reachable
    return reachable


@router.get("/summary", response_model=FleetSummary)
async def summary(db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    """Everything the top strip of the dashboard shows, in one query round."""
    nodes = list((await db.execute(select(Node))).scalars().unique())
    deps = list(
        (await db.execute(select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES + (DeployStatus.failed,)))))
        .scalars().unique()
    )

    gpus_total = sum(len(n.gpus) for n in nodes)
    gpus_busy = sum(len(busy_gpu_indices(n)) for n in nodes)
    vram_total = sum(g.memory_total_mb for n in nodes for g in n.gpus) / 1024
    vram_used = sum(g.memory_used_mb for n in nodes for g in n.gpus) / 1024

    cutoff = datetime.now(timezone.utc) - RECENT_FAILURE_WINDOW
    recent_failures = [
        d for d in deps
        if d.status == DeployStatus.failed and _as_utc(d.created_at) >= cutoff
    ]
    healthy = [d for d in deps if d.status == DeployStatus.healthy]
    tps = sum(float(d.last_metrics.get("gen_tps", 0) or 0) for d in healthy)
    running = sum(int(d.last_metrics.get("running", 0) or 0) for d in healthy)
    waiting = sum(int(d.last_metrics.get("waiting", 0) or 0) for d in healthy)

    litellm_ok = await _litellm_reachable()

    return FleetSummary(
        nodes_total=len(nodes),
        nodes_online=sum(1 for n in nodes if n.status == NodeStatus.online),
        nodes_unreachable=sum(1 for n in nodes if n.status == NodeStatus.unreachable),
        gpus_total=gpus_total,
        gpus_busy=gpus_busy,
        gpus_free=gpus_total - gpus_busy,
        vram_total_gb=round(vram_total, 1),
        vram_used_gb=round(vram_used, 1),
        deployments_healthy=len(healthy),
        deployments_degraded=sum(1 for d in deps if d.status == DeployStatus.degraded),
        deployments_failed=len(recent_failures),
        models_served=len({d.served_model_name for d in healthy}),
        tokens_per_second=round(tps, 1),
        requests_running=running,
        requests_waiting=waiting,
        litellm_reachable=litellm_ok,
    )


def _as_utc(value: datetime | None) -> datetime:
    """created_at is stored as UTC, but not every backend returns tzinfo."""
    if value is None:
        return datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@router.get("/events", response_model=list[EventOut])
async def list_events(
    limit: int = Query(100, ge=1, le=500),
    severity: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(current_user),
):
    q = select(Event).order_by(Event.ts.desc()).limit(limit)
    if severity:
        q = q.where(Event.severity == severity)
    return list((await db.execute(q)).scalars())


@router.get("/audit", response_model=list[AuditOut])
async def list_audit(
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(current_user),
):
    rows = await db.execute(select(AuditLog).order_by(AuditLog.ts.desc()).limit(limit))
    return list(rows.scalars())


@router.get("/config")
async def runtime_config(_: User = Depends(current_user)):
    return {
        "driver": settings.driver,
        "simulated": settings.driver == "sim",
        "auth_mode": settings.auth_mode,
        "vllm_image": settings.vllm_image,
        "litellm_base_url": settings.litellm_base_url,
        "poll_seconds": settings.gpu_poll_seconds,
    }


async def _authenticate_socket(socket: WebSocket, db: AsyncSession) -> User | None:
    """Same identity check as `current_user`, adapted for a connection that
    cannot carry an Authorization header — a browser attaches the session
    cookie to a WebSocket upgrade automatically, so that is what stands in
    for it here."""
    if settings.auth_mode == "dev":
        row = await db.execute(select(User).where(User.email == DEV_EMAIL))
        user = row.scalar_one_or_none()
        if user is None:
            user = await upsert_user(db, email=DEV_EMAIL, name="Local Admin", role=Role.admin)
        return user

    sess = socket.session.get("user")
    if not sess:
        return None
    row = await db.execute(select(User).where(User.email == sess["email"]))
    return row.scalar_one_or_none()


@router.websocket("/ws")
async def ws(socket: WebSocket, db: AsyncSession = Depends(get_db)):
    """Live fleet feed: node/gpu updates, deployment transitions, metrics, events."""
    user = await _authenticate_socket(socket, db)
    if user is None:
        # Reject before accept: a signed-out caller never gets the stream.
        await socket.close(code=4401)
        return
    await socket.accept()
    with events.subscription() as q:
        try:
            await socket.send_text(json.dumps({"topic": "hello", "data": {"driver": settings.driver}}))
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    await socket.send_text(json.dumps({"topic": "ping", "data": {}}))
                    continue
                await socket.send_text(msg)
        except (WebSocketDisconnect, RuntimeError):
            return
