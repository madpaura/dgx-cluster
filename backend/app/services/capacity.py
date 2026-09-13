"""What is actually left on each GPU.

A GPU is a pool rather than a slot. Two things can consume it: reservations
dgxctl has made, and whatever nvidia-smi reports as in use. Neither alone is
trustworthy for scheduling — a reservation counts memory a model has not
finished loading yet, and observed usage counts processes dgxctl never started
— so the larger of the two is what remains available.
"""
from __future__ import annotations

from ..models import ACTIVE_STATUSES, Gpu, Node

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
