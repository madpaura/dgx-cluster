"""Node inventory: probe hardware, keep the GPU table current, adopt strays."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit, events
from ..drivers import get_driver
from ..models import ACTIVE_STATUSES, Deployment, DeployStatus, Gpu, MetricSample, Node, NodeStatus
from .deployments import LABEL_KEY

log = logging.getLogger(__name__)


async def refresh(db: AsyncSession, node: Node, *, record_samples: bool = True) -> Node:
    """Probe one node and write what we learned. Never raises."""
    facts = await get_driver().probe(node)
    previous = node.status

    if not facts.reachable or facts.error:
        node.status = NodeStatus.unreachable if not facts.reachable else node.status
        node.last_error = facts.error
        if previous != NodeStatus.unreachable and node.status == NodeStatus.unreachable:
            await audit.emit(
                db, severity="error", source="node", source_id=node.id,
                message=f"{node.name} became unreachable: {facts.error[:160]}",
            )
        await db.commit()
        events.publish("node", {"id": node.id, "status": node.status.value})
        return node

    node.last_error = ""
    node.last_seen = datetime.now(timezone.utc)
    node.driver_version = facts.driver_version or node.driver_version
    node.cuda_version = facts.cuda_version or node.cuda_version
    node.docker_version = facts.docker_version or node.docker_version
    node.cpu_count = facts.cpu_count or node.cpu_count
    node.memory_gb = facts.memory_gb or node.memory_gb
    # Don't override an operator's deliberate drain/maintenance flag.
    if node.status not in (NodeStatus.draining, NodeStatus.maintenance):
        node.status = NodeStatus.online
        if previous == NodeStatus.unreachable:
            await audit.emit(
                db, severity="info", source="node", source_id=node.id,
                message=f"{node.name} is back online",
            )

    existing = {g.index: g for g in node.gpus}
    now = datetime.now(timezone.utc)
    touched: list[tuple[Gpu, object]] = []
    for probe in facts.gpus:
        gpu = existing.get(probe.index)
        if gpu is None:
            gpu = Gpu(node_id=node.id, index=probe.index)
            db.add(gpu)
            node.gpus.append(gpu)
        gpu.uuid = probe.uuid
        gpu.name = probe.name
        gpu.memory_total_mb = probe.memory_total_mb
        gpu.memory_used_mb = probe.memory_used_mb
        gpu.utilization = probe.utilization
        gpu.temperature_c = probe.temperature_c
        gpu.power_draw_w = probe.power_draw_w
        gpu.power_limit_w = probe.power_limit_w
        gpu.ecc_errors = probe.ecc_errors
        gpu.updated_at = now
        touched.append((gpu, probe))

    if record_samples and touched:
        # New Gpu rows only get their primary key at flush time, and the sample
        # references it — so flush before writing the time series.
        await db.flush()
        for gpu, probe in touched:
            db.add(
                MetricSample(
                    scope="gpu", scope_id=gpu.id, ts=now,
                    values={
                        "util": probe.utilization,
                        "mem_used_mb": probe.memory_used_mb,
                        "mem_total_mb": probe.memory_total_mb,
                        "temp_c": probe.temperature_c,
                        "power_w": probe.power_draw_w,
                    },
                )
            )

    await db.commit()
    events.publish(
        "node",
        {
            "id": node.id,
            "status": node.status.value,
            "gpus": [
                {"index": g.index, "util": g.utilization,
                 "mem_used_mb": g.memory_used_mb, "mem_total_mb": g.memory_total_mb,
                 "temp_c": g.temperature_c, "power_w": g.power_draw_w}
                for g in node.gpus
            ],
        },
    )
    return node


async def adopt_and_prune(db: AsyncSession, node: Node) -> dict:
    """Compare what the node is actually running against what we think.

    Two mismatches matter: containers we launched that have vanished (mark the
    deployment failed) and dgxctl-labelled containers with no deployment row
    (an orphan from a control-server restore — offer to clean it up).
    """
    driver = get_driver()
    try:
        containers = await driver.list_containers(node, label_filter=LABEL_KEY)
    except Exception as exc:
        return {"error": str(exc)[:300], "orphans": [], "missing": []}

    by_dep_id = {c.labels.get(LABEL_KEY): c for c in containers if c.labels.get(LABEL_KEY)}
    rows = await db.execute(
        select(Deployment).where(Deployment.node_id == node.id, Deployment.status.in_(ACTIVE_STATUSES))
    )
    active = list(rows.scalars().unique())
    known_ids = {d.id for d in active}

    missing = []
    for dep in active:
        if dep.id not in by_dep_id and dep.status != DeployStatus.pending:
            dep.status = DeployStatus.failed
            dep.status_reason = "container is gone from the node (removed externally or node rebooted)"
            missing.append(dep.id)
            await audit.emit(
                db, severity="error", source="deployment", source_id=dep.id,
                message=f"{dep.served_model_name} vanished from {node.name}",
            )

    orphans = [
        {"name": c.name, "id": c.id, "image": c.image, "state": c.state,
         "model": c.labels.get("dgxctl.model", ""), "deployment_id": c.labels.get(LABEL_KEY, "")}
        for c in containers
        if c.labels.get(LABEL_KEY) not in known_ids
    ]
    await db.commit()
    return {"error": "", "orphans": orphans, "missing": missing}


async def gpu_allocation(node: Node) -> dict[int, str]:
    """index -> deployment id, for GPUs currently claimed."""
    out: dict[int, str] = {}
    for d in node.deployments:
        if d.status in ACTIVE_STATUSES:
            for i in d.gpu_indices:
                out[int(i)] = d.id
    return out
