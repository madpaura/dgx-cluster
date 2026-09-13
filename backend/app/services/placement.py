"""Where should this model run?

Answers the question the operator would otherwise answer with a spreadsheet:
given a model's per-GPU memory need and tensor-parallel size, which GPUs on
which nodes are free and big enough — and which choice wastes the least.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import ACTIVE_STATUSES, Node
from .capacity import RESERVE_MB, node_capacity, tenants  # noqa: F401


@dataclass
class Placement:
    node_id: str
    node_name: str
    gpu_indices: list[int]
    gpu_model: str
    free_mb_per_gpu: int
    reserve_mb_per_gpu: int
    score: float
    note: str = ""
    shares_with: int = 0        # models already on the chosen GPUs


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
    need_mb = int(per_gpu_gb * 1024)   # headroom is already held back by node_capacity
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

        capacity = node_capacity(node)
        occupants = tenants(node)

        fits = [g for g in node.gpus if capacity.get(g.index, 0) >= need_mb]
        if len(fits) < tp:
            biggest = max(capacity.values(), default=0)
            roomy = len([v for v in capacity.values() if v >= need_mb])
            rejections.append(
                Rejection(
                    node.name,
                    f"needs {need_mb / 1024:.0f} GiB free per GPU; "
                    f"{roomy} of {len(node.gpus)} GPUs have that "
                    f"(most free: {biggest / 1024:.0f} GiB)",
                )
            )
            continue

        group = _pick_group(fits, tp, capacity)
        if group is None:
            rejections.append(Rejection(node.name, f"no homogeneous group of {tp} GPUs has room"))
            continue

        free_mb = min(capacity[g.index] for g in group)
        sharing = max(len(occupants.get(g.index, [])) for g in group)

        # Best-fit on the leftover, so a small model lands on a card that is
        # already partly used rather than opening a fresh one — whole GPUs stay
        # available for the models that genuinely need them.
        leftover_gb = (free_mb - need_mb) / 1024
        score = leftover_gb + (len(node.gpus) - len(group)) * 0.5
        note = ""
        if sharing:
            note = f"shares with {sharing} model{'s' if sharing > 1 else ''}"
            score -= 20     # prefer packing over opening another card
        if tp > 1 and group[-1].index - group[0].index == tp - 1 and group[0].index % tp == 0:
            note = ("aligned NVLink group" + (f", {note}" if note else ""))
            score -= 25
        options.append(
            Placement(
                node_id=node.id,
                node_name=node.name,
                gpu_indices=[g.index for g in group],
                gpu_model=group[0].name,
                free_mb_per_gpu=free_mb,
                reserve_mb_per_gpu=need_mb,
                score=score,
                note=note,
                shares_with=sharing,
            )
        )

    options.sort(key=lambda p: p.score)
    return options, rejections


def _pick_group(candidates: list, tp: int, capacity: dict[int, int]):
    """Prefer an aligned contiguous run of identical GPUs; fall back to any
    identical set. Tensor parallel across mixed GPU models is never right, and
    a group whose members have unequal room is limited by its smallest."""
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
        # Nothing contiguous: take the fullest cards that still fit, so the
        # emptiest ones stay whole.
        return sorted(group, key=lambda g: capacity[g.index])[:tp]
    return None
