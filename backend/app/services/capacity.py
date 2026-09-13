"""What is actually left on each GPU.

A GPU is a pool rather than a slot. Two things can consume it: reservations
dgxctl has made, and whatever nvidia-smi reports as in use. Neither alone is
trustworthy for scheduling — a reservation counts memory a model has not
finished loading yet, and observed usage counts processes dgxctl never started
— so the larger of the two is what remains available.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ACTIVE_STATUSES, Deployment, Gpu, Node

# Headroom left on every GPU beyond what anyone reserved: CUDA context,
# fragmentation, and the cuBLAS workspace each process maps.
RESERVE_MB = 1500


def reserved_mb(node: Node) -> dict[int, int]:
    """index -> VRAM claimed on it by deployments that are alive or coming up.

    A deployment with no recorded reservation is read as holding its whole GPU.
    That is the only safe reading: rows written before dgxctl tracked shares
    took the card exclusively, and treating an unknown as zero would schedule
    a second model on top of a running one.
    """
    total_mb = {g.index: g.memory_total_mb for g in node.gpus}
    out: dict[int, int] = {}
    for dep in node.deployments:
        if dep.status not in ACTIVE_STATUSES:
            continue
        for index in dep.gpu_indices:
            index = int(index)
            share = int(dep.reserved_mb_per_gpu or 0) or total_mb.get(index, 0)
            out[index] = out.get(index, 0) + share
    return out


def tenants(node: Node) -> dict[int, list]:
    """index -> the deployments sharing it, in the order they arrived."""
    out: dict[int, list] = {}
    for dep in node.deployments:
        if dep.status not in ACTIVE_STATUSES:
            continue
        for index in dep.gpu_indices:
            out.setdefault(int(index), []).append(dep)
    return out


def available_mb(gpu: Gpu, claimed_mb: int) -> int:
    """What a new model could still ask for on this GPU.

    Observed usage is consulted as well as the ledger so that a process dgxctl
    did not start — someone's notebook on a workstation — cannot be scheduled
    over.
    """
    in_use = max(claimed_mb, gpu.memory_used_mb)
    return max(0, gpu.memory_total_mb - in_use - RESERVE_MB)


def node_capacity(node: Node) -> dict[int, int]:
    """index -> free VRAM in MB, for every GPU on the node."""
    claimed = reserved_mb(node)
    return {g.index: available_mb(g, claimed.get(g.index, 0)) for g in node.gpus}


async def node_capacity_now(db: AsyncSession, node: Node) -> dict[int, int]:
    """index -> free VRAM, read from committed state rather than from whatever
    this session loaded earlier.

    The in-memory reading is fine for planning, which is advisory. Granting a
    claim is not: a session that loaded the node before a rival committed would
    otherwise be told the GPU is still empty, and hand out memory that is
    already spoken for. Querying the reservations directly also avoids expiring
    the session, which would turn every later attribute access into IO.
    """
    rows = await db.execute(
        select(Deployment.gpu_indices, Deployment.reserved_mb_per_gpu).where(
            Deployment.node_id == node.id,
            Deployment.status.in_(ACTIVE_STATUSES),
        )
    )
    total_mb = {g.index: g.memory_total_mb for g in node.gpus}
    claimed: dict[int, int] = {}
    for indices, share in rows.all():
        for index in indices or []:
            index = int(index)
            claimed[index] = claimed.get(index, 0) + (int(share or 0) or total_mb.get(index, 0))
    return {g.index: available_mb(g, claimed.get(g.index, 0)) for g in node.gpus}
