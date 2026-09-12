"""Background reconciliation.

Three loops, all idempotent, all safe to restart:
  inventory  - probe nodes, refresh GPU table
  reconcile  - observe containers, move deployments to their true status,
               register newly-healthy models with LiteLLM
  metrics    - scrape vLLM /metrics for anything healthy
Plus a retention sweep so the samples table cannot grow without bound.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import audit, events
from .config import settings
from .db import SessionLocal
from .drivers import get_driver
from .models import ACTIVE_STATUSES, Deployment, DeployStatus, MetricSample, Node, NodeStatus
from .services import litellm as litellm_svc
from .services import nodes as node_svc
from .services.deployments import LABEL_KEY, startup_expired
from .services.vllm_metrics import parse_prometheus, summarize

log = logging.getLogger(__name__)


async def _loop(name: str, interval: int, fn) -> None:
    """Run fn forever; one failure must never kill the loop."""
    while True:
        started = asyncio.get_running_loop().time()
        try:
            async with SessionLocal() as db:
                await fn(db)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker loop %s failed", name)
        elapsed = asyncio.get_running_loop().time() - started
        await asyncio.sleep(max(1.0, interval - elapsed))


# ------------------------------------------------------------------ inventory

async def inventory_pass(db: AsyncSession) -> None:
    rows = await db.execute(select(Node))
    for node in rows.scalars().unique():
        try:
            await node_svc.refresh(db, node)
        except Exception:
            log.exception("probe failed for %s", node.name)
            await db.rollback()  # one bad node must not poison the rest of the pass


# ------------------------------------------------------------------ reconcile

async def reconcile_pass(db: AsyncSession) -> None:
    driver = get_driver()
    rows = await db.execute(select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES)))
    deployments = list(rows.scalars().unique())
    if not deployments:
        return

    by_node: dict[str, list[Deployment]] = {}
    for d in deployments:
        by_node.setdefault(d.node_id, []).append(d)

    for node_id, deps in by_node.items():
        node = deps[0].node
        if node.status == NodeStatus.unreachable:
            for d in deps:
                if d.status == DeployStatus.healthy:
                    await _transition(db, d, DeployStatus.degraded, "node unreachable")
            continue

        try:
            containers = await driver.list_containers(node, label_filter=LABEL_KEY)
        except Exception as exc:
            log.warning("docker ps failed on %s: %s", node.name, exc)
            continue
        by_id = {c.labels.get(LABEL_KEY): c for c in containers}

        for dep in deps:
            await _reconcile_one(db, dep, by_id.get(dep.id))

    await db.commit()


async def _reconcile_one(db: AsyncSession, dep: Deployment, container) -> None:
    driver = get_driver()

    if container is None:
        if dep.status == DeployStatus.pending:
            return  # launch still in flight
        await _transition(db, dep, DeployStatus.failed, "container not present on node")
        return

    if container.state in ("exited", "dead"):
        reason = f"container exited (code {container.exit_code})"
        if dep.status != DeployStatus.failed:
            tail = await driver.logs(dep.node, dep.container_name, tail=60)
            reason = f"{reason}: {_last_error_line(tail)}"
            await _transition(db, dep, DeployStatus.failed, reason)
        return

    if container.state == "restarting":
        await _transition(db, dep, DeployStatus.degraded, "container is restart-looping")
        return

    code, _ = await driver.http_get(dep.node, dep.port, "/health", timeout=4.0)
    if code == 200:
        if dep.status != DeployStatus.healthy:
            await _transition(db, dep, DeployStatus.healthy, "serving")
            dep.healthy_since = datetime.now(timezone.utc)
            await _register_litellm(db, dep, announce=True)
        elif not dep.litellm_registered:
            # Keep trying. A model that went healthy while the proxy was still
            # booting — the normal case after restarting the control plane —
            # would otherwise stay unrouted until a human noticed. Quietly,
            # because the first failure was already reported.
            await _register_litellm(db, dep, announce=False)
        return

    # Running but not answering: fine while loading, a problem once the grace period ends.
    if startup_expired(dep):
        await _transition(db, dep, DeployStatus.degraded, "container is up but /health does not answer")
    elif dep.status not in (DeployStatus.starting, DeployStatus.pulling):
        await _transition(db, dep, DeployStatus.starting, "loading weights")


async def _register_litellm(db: AsyncSession, dep: Deployment, *, announce: bool) -> None:
    """Register a healthy deployment with the proxy.

    `announce=False` is a retry: report success, but stay quiet about failure so
    an unreachable proxy does not fill the event feed once per poll.
    """
    if dep.litellm_registered or not settings.litellm_auto_register:
        return
    ok, msg = await litellm_svc.sync_deployment(dep, register=True)
    dep.litellm_registered = ok
    if ok:
        dep.litellm_model_id = dep.id
    if ok or announce:
        await audit.emit(
            db,
            severity="info" if ok else "warning",
            source="litellm", source_id=dep.id,
            message=msg if ok else f"LiteLLM registration failed for {dep.served_model_name}: {msg}",
        )


async def _transition(db: AsyncSession, dep: Deployment, status: DeployStatus, reason: str) -> None:
    if dep.status == status and dep.status_reason == reason:
        return
    previous = dep.status
    dep.status = status
    dep.status_reason = reason[:1000]
    dep.updated_at = datetime.now(timezone.utc)
    severity = {"failed": "error", "degraded": "warning"}.get(status.value, "info")
    if previous != status:
        await audit.emit(
            db, severity=severity, source="deployment", source_id=dep.id,
            message=f"{dep.served_model_name} on {dep.node.name}: {previous.value} -> {status.value} ({reason[:160]})",
        )
    events.publish("deployment", {"id": dep.id, "status": status.value, "reason": reason[:200]})


def _last_error_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if any(tok in line for tok in ("Error", "ERROR", "error:", "Exception", "Traceback")):
            return line.strip()[:300]
    return (text.splitlines() or [""])[-1][:300]


# -------------------------------------------------------------------- metrics

async def metrics_pass(db: AsyncSession) -> None:
    driver = get_driver()
    rows = await db.execute(
        select(Deployment).where(Deployment.status.in_((DeployStatus.healthy, DeployStatus.degraded)))
    )
    now = datetime.now(timezone.utc)
    for dep in rows.scalars().unique():
        code, body = await driver.http_get(dep.node, dep.port, "/metrics", timeout=6.0)
        if code != 200:
            continue
        summary = summarize(parse_prometheus(body), dep.last_metrics or None)
        dep.last_metrics = summary
        db.add(
            MetricSample(
                scope="deployment", scope_id=dep.id, ts=now,
                values={
                    "running": summary["running"],
                    "waiting": summary["waiting"],
                    "kv_cache_pct": summary["kv_cache_pct"],
                    "gen_tps": summary["gen_tps"],
                    "prompt_tps": summary["prompt_tps"],
                    "ttft_ms": summary["ttft_avg_ms"],
                    "e2e_ms": summary["e2e_avg_ms"],
                },
            )
        )
        events.publish("metrics", {"deployment_id": dep.id, **summary})
    await db.commit()


# ------------------------------------------------------------------ retention

async def retention_pass(db: AsyncSession) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.metric_retention_hours)
    await db.execute(delete(MetricSample).where(MetricSample.ts < cutoff))
    await db.commit()


def start(loop_tasks: list[asyncio.Task]) -> None:
    loop_tasks.append(asyncio.create_task(_loop("inventory", settings.gpu_poll_seconds, inventory_pass)))
    loop_tasks.append(asyncio.create_task(_loop("reconcile", settings.health_poll_seconds, reconcile_pass)))
    loop_tasks.append(asyncio.create_task(_loop("metrics", settings.vllm_poll_seconds, metrics_pass)))
    loop_tasks.append(asyncio.create_task(_loop("retention", 3600, retention_pass)))
