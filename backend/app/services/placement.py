"""Where should this model run?

Answers the question the operator would otherwise answer with a spreadsheet:
given a model's per-GPU memory need and tensor-parallel size, which GPUs on
which nodes are free and big enough — and which choice wastes the least.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import ACTIVE_STATUSES, Node

# Headroom left on a GPU beyond the model's own need (CUDA context, fragmentation).
RESERVE_MB = 1500


@dataclass
class Placement:
    node_id: str
    node_name: str
    gpu_indices: list[int]
    gpu_model: str
    free_mb_per_gpu: int
    score: float
    note: str = ""


@dataclass
class Rejection:
    node_name: str
    reason: str


def busy_gpu_indices(node: Node) -> set[int]:
    """GPUs claimed by a deployment that is alive or coming up."""
    busy: set[int] = set()
    for d in node.deployments:
        if d.status in ACTIVE_STATUSES:
            busy.update(int(i) for i in d.gpu_indices)
    return busy


def plan(
    nodes: list[Node],
    per_gpu_gb: float,
    tp: int,
    required_labels: dict[str, str] | None = None,
    exclude_node_ids: set[str] | None = None,
) -> tuple[list[Placement], list[Rejection]]:
    """Return viable placements best-first, plus why each other node was skipped.

    The rejection list matters as much as the placements — it is what turns
    "no capacity" from a dead end into something the operator can act on.
    """
    need_mb = int(per_gpu_gb * 1024) + RESERVE_MB
    options: list[Placement] = []
    rejections: list[Rejection] = []
    exclude_node_ids = exclude_node_ids or set()

    for node in nodes:
        if node.id in exclude_node_ids:
            continue
        if not node.schedulable:
            rejections.append(Rejection(node.name, f"node is {node.status.value}"))
            continue
        if required_labels and any(node.labels.get(k) != v for k, v in required_labels.items()):
            rejections.append(Rejection(node.name, "does not match required labels"))
            continue
        if len(node.gpus) < tp:
            rejections.append(Rejection(node.name, f"has {len(node.gpus)} GPUs, needs {tp}"))
            continue

        busy = busy_gpu_indices(node)
        free = [g for g in node.gpus if g.index not in busy]
        if len(free) < tp:
            rejections.append(
                Rejection(node.name, f"only {len(free)} of {len(node.gpus)} GPUs free, needs {tp}")
            )
            continue

        fits = [g for g in free if (g.memory_total_mb - g.memory_used_mb) >= need_mb]
        if len(fits) < tp:
            biggest = max((g.memory_total_mb - g.memory_used_mb for g in free), default=0)
            rejections.append(
                Rejection(
                    node.name,
                    f"needs {need_mb / 1024:.0f} GiB per GPU, largest free GPU has {biggest / 1024:.0f} GiB",
                )
            )
            continue

        group = _pick_group(fits, tp)
        if group is None:
            rejections.append(Rejection(node.name, f"no homogeneous group of {tp} GPUs available"))
            continue

        free_mb = min(g.memory_total_mb - g.memory_used_mb for g in group)
        # Best-fit: prefer the node with the fewest spare GPUs left over, so big
        # contiguous blocks stay available for models that actually need them.
        leftover = len(free) - tp
        score = leftover * 100 + (free_mb - need_mb) / 1024
        note = ""
        if tp > 1 and group[-1].index - group[0].index == tp - 1 and group[0].index % tp == 0:
            note = "aligned NVLink group"
            score -= 25
        options.append(
            Placement(
                node_id=node.id,
                node_name=node.name,
                gpu_indices=[g.index for g in group],
                gpu_model=group[0].name,
                free_mb_per_gpu=free_mb,
                score=score,
                note=note,
            )
        )

    options.sort(key=lambda p: p.score)
    return options, rejections


def _pick_group(candidates: list, tp: int):
    """Prefer an aligned contiguous run of identical GPUs; fall back to any
    identical set. Tensor parallel across mixed GPU models is never right."""
    by_model: dict[str, list] = {}
    for g in candidates:
        by_model.setdefault(g.name, []).append(g)

    for group in by_model.values():
        group.sort(key=lambda g: g.index)
        if len(group) < tp:
            continue
        for start in range(len(group) - tp + 1):
            window = group[start : start + tp]
            contiguous = window[-1].index - window[0].index == tp - 1
            if contiguous and window[0].index % tp == 0:
                return window
        for start in range(len(group) - tp + 1):
            window = group[start : start + tp]
            if window[-1].index - window[0].index == tp - 1:
                return window
        return group[:tp]
    return None
