"""Deployment lifecycle: turn 'serve this model' into running vLLM containers.

The control server never holds long locks against the fleet. It writes the
intended state to Postgres, fires the docker command, then lets the reconciler
(worker.py) converge status by observing the node. That is what keeps the UI
honest when a node reboots behind your back.
"""
from __future__ import annotations

import asyncio
import logging
import weakref
import re
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit, events
from ..config import settings
from ..drivers import LaunchSpec, get_driver
from ..models import ACTIVE_STATUSES, Deployment, DeployStatus, ModelSpec, Node, NodeStatus
from . import litellm as litellm_svc
from .capacity import node_capacity_now
from .placement import Placement, plan

log = logging.getLogger(__name__)

LABEL_KEY = "dgxctl.deployment"
STARTUP_GRACE_SECONDS = 900  # big models legitimately take many minutes to load

# Choosing GPUs and a port is read-then-act: two callers that look at the same
# moment both see the same GPU free and both claim it. That is not hypothetical
# here — the MCP endpoint exists so an agent can deploy while a human is at the
# dashboard. gpu_indices is a JSON list, so no unique constraint can catch the
# collision either; serialising the claim is what prevents it.
#
# This holds for one control-plane process, which is what dgxctl is: the event
# bus and the worker loops already assume a single instance. Running the API
# replicated would need a database-level lock instead.
#
# Keyed by event loop rather than created once at import: an asyncio.Lock binds
# to the loop that first awaits it and raises on any other, which a module-level
# singleton makes impossible to reuse across loops.
_allocation_locks: "weakref.WeakKeyDictionary[object, asyncio.Lock]" = weakref.WeakKeyDictionary()


def allocation_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _allocation_locks.get(loop)
    if lock is None:
        lock = _allocation_locks[loop] = asyncio.Lock()
    return lock


@dataclass
class Target:
    node_id: str
    gpu_indices: list[int]


class DeployError(RuntimeError):
    pass


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()[:48] or "model"


def build_vllm_args(
    *,
    hf_repo: str,
    served_model_name: str,
    tensor_parallel_size: int,
    max_model_len: int = 0,
    quantization: str = "",
    gpu_memory_utilization: float = 0.90,
    revision: str = "",
    extra_args: dict | None = None,
) -> list[str]:
    args = [
        "--model", hf_repo,
        "--served-model-name", served_model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", f"{gpu_memory_utilization:g}",
    ]
    if max_model_len:
        args += ["--max-model-len", str(max_model_len)]
    if quantization:
        args += ["--quantization", quantization]
    if revision:
        args += ["--revision", revision]
    for k, v in (extra_args or {}).items():
        flag = k if k.startswith("--") else f"--{k}"
        if v is True:
            args.append(flag)
        elif v is False or v is None:
            continue
        else:
            args += [flag, str(v)]
    return args


async def allocate_port(db: AsyncSession, node_id: str) -> int:
    rows = await db.execute(
        select(Deployment.port).where(
            Deployment.node_id == node_id, Deployment.status.in_(ACTIVE_STATUSES)
        )
    )
    taken = set(rows.scalars().all())
    for port in range(settings.vllm_port_range_start, settings.vllm_port_range_end + 1):
        if port not in taken:
            return port
    raise DeployError(f"no free port in {settings.vllm_port_range_start}-{settings.vllm_port_range_end}")


async def resolve_targets(
    db: AsyncSession,
    *,
    replicas: int,
    tensor_parallel_size: int,
    per_gpu_gb: float,
    explicit: list[Target] | None,
    node_filter: list[str] | None = None,
) -> list[Target]:
    """Explicit targets win. Otherwise auto-place across the fleet, one replica
    per node so a single node failure never takes the whole model offline."""
    if explicit:
        return explicit

    rows = await db.execute(select(Node))
    nodes = list(rows.scalars().unique())
    if node_filter:
        nodes = [n for n in nodes if n.id in set(node_filter)]

    chosen: list[Target] = []
    used_nodes: set[str] = set()
    for _ in range(replicas):
        options, rejections = plan(
            nodes, per_gpu_gb=per_gpu_gb, tp=tensor_parallel_size, exclude_node_ids=used_nodes
        )
        if not options:
            if chosen:
                break  # placed what we could; caller reports the shortfall
            detail = "; ".join(f"{r.node_name}: {r.reason}" for r in rejections[:6]) or "no nodes registered"
            raise DeployError(f"no node can host this model ({detail})")
        best: Placement = options[0]
        chosen.append(Target(node_id=best.node_id, gpu_indices=best.gpu_indices))
        used_nodes.add(best.node_id)
    return chosen


async def create(
    db: AsyncSession,
    *,
    actor: str,
    served_model_name: str,
    hf_repo: str,
    targets: list[Target],
    tensor_parallel_size: int,
    per_gpu_gb: float,
    spec: ModelSpec | None = None,
    max_model_len: int = 0,
    quantization: str = "",
    gpu_memory_utilization: float | None = None,
    extra_args: dict | None = None,
    image: str = "",
    team_id: str | None = None,
) -> list[Deployment]:
    """Claim capacity, then launch. Failures are per-target: deploying to six
    nodes where one is down still gives you five."""
    created = await _reserve(
        db, actor=actor, served_model_name=served_model_name, hf_repo=hf_repo,
        targets=targets, tensor_parallel_size=tensor_parallel_size,
        per_gpu_gb=per_gpu_gb, spec=spec,
        max_model_len=max_model_len, quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization, extra_args=extra_args,
        image=image, team_id=team_id,
    )
    await _launch(db, created, actor=actor)
    return created


async def _reserve(
    db: AsyncSession,
    *,
    actor: str,
    served_model_name: str,
    hf_repo: str,
    targets: list[Target],
    tensor_parallel_size: int,
    per_gpu_gb: float,
    spec: ModelSpec | None,
    max_model_len: int,
    quantization: str,
    gpu_memory_utilization: float | None,
    extra_args: dict | None,
    image: str,
    team_id: str | None,
) -> list[Deployment]:
    """Write the claim down and commit it, before anything slow happens.

    The commit is the point: until these rows are visible with an active
    status, another caller still sees these GPUs and ports as free. Launching
    inside the lock instead would serialise every `docker run` in the fleet
    behind one mutex, so the reservation is what is protected, not the work.
    """
    image = image or (spec.vllm_image if spec and spec.vllm_image else settings.vllm_image)
    created: list[Deployment] = []

    async with allocation_lock():
        for target in targets:
            node = await db.get(Node, target.node_id)
            if node is None:
                raise DeployError(f"unknown node {target.node_id}")
            if node.status in (NodeStatus.draining, NodeStatus.maintenance):
                raise DeployError(f"{node.name} is {node.status.value}; not accepting deployments")

            reserve_mb = int(round(per_gpu_gb * 1024))
            capacity = await node_capacity_now(db, node)
            short = {
                i: capacity.get(i, 0) for i in target.gpu_indices
                if capacity.get(i, 0) < reserve_mb
            }
            if short:
                detail = ", ".join(
                    f"GPU {i} has {mb / 1024:.0f} GiB free" for i, mb in sorted(short.items())
                )
                raise DeployError(
                    f"{node.name} cannot fit {per_gpu_gb:.0f} GiB per GPU: {detail}; "
                    f"refresh and pick again"
                )

            port = await allocate_port(db, node.id)
            dep = Deployment(
                id=str(uuid.uuid4()),
                served_model_name=served_model_name,
                spec_id=spec.id if spec else None,
                hf_repo=hf_repo,
                node_id=node.id,
                node=node,
                gpu_indices=list(target.gpu_indices),
                reserved_mb_per_gpu=reserve_mb,
                port=port,
                status=DeployStatus.pending,
                status_reason="claimed; starting container",
                image=image,
                tensor_parallel_size=tensor_parallel_size,
                team_id=team_id,
                created_by=actor,
            )
            dep.container_name = f"{settings.container_prefix}-{slugify(served_model_name)}-{dep.id[:8]}"
            # The fraction is derived from the reservation rather than fixed,
            # because --gpu-memory-utilization is a share of the WHOLE card: a
            # default of 0.9 would have the first model claim almost all of it
            # however little it needs, and nothing could ever share the GPU.
            largest = max((g.memory_total_mb for g in node.gpus if g.index in target.gpu_indices),
                          default=0)
            share = gpu_memory_utilization
            if share is None:
                share = round(reserve_mb / largest, 3) if largest else 0.90

            dep.vllm_args = {"argv": build_vllm_args(
                hf_repo=hf_repo,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                quantization=quantization or (spec.quantization if spec else ""),
                gpu_memory_utilization=share,
                revision=spec.revision if spec else "",
                extra_args={**(spec.extra_args if spec else {}), **(extra_args or {})},
            )}
            db.add(dep)
            node.deployments.append(dep)
            created.append(dep)

        await db.commit()
    return created


async def _launch(db: AsyncSession, deployments: list[Deployment], *, actor: str) -> None:
    """Start the containers for claims already written down.

    A claim whose container will not start is marked failed, which releases its
    GPUs — the reservation only outlives the attempt if the attempt succeeds.
    """
    driver = get_driver()
    env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    if settings.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

    for dep in deployments:
        args = list(dep.vllm_args.get("argv", []))
        launch = LaunchSpec(
            name=dep.container_name,
            image=dep.image,
            gpu_indices=[int(i) for i in dep.gpu_indices],
            host_port=dep.port,
            args=args,
            env=env,
            volumes={settings.hf_cache_dir: "/root/.cache/huggingface"},
            labels={
                LABEL_KEY: dep.id,
                "dgxctl.model": dep.served_model_name,
                "dgxctl.owner": actor,
            },
        )
        try:
            dep.status = DeployStatus.pulling
            dep.container_id = await driver.launch(dep.node, launch)
            dep.status = DeployStatus.starting
            dep.status_reason = "container started, loading weights"
            await audit.record(
                db, actor=actor, action="deployment.create", target_type="deployment",
                target_id=dep.id,
                summary=f"deploy {dep.served_model_name} to {dep.node.name} GPUs {dep.gpu_indices}",
                detail={"argv": args, "image": dep.image, "port": dep.port},
            )
            await audit.emit(
                db, severity="info", source="deployment", source_id=dep.id,
                message=(f"{dep.served_model_name} starting on {dep.node.name} "
                         f"GPU {','.join(str(i) for i in dep.gpu_indices)}"),
            )
        except Exception as exc:
            dep.status = DeployStatus.failed
            dep.status_reason = str(exc)[:1000]
            await audit.record(
                db, actor=actor, action="deployment.create", target_type="deployment",
                target_id=dep.id,
                summary=f"failed to start {dep.served_model_name} on {dep.node.name}",
                detail={"error": str(exc)}, ok=False,
            )
            await audit.emit(
                db, severity="error", source="deployment", source_id=dep.id,
                message=(f"failed to start {dep.served_model_name} on {dep.node.name}: "
                         f"{str(exc)[:200]}"),
            )

    await db.commit()
    for dep in deployments:
        events.publish("deployment", {"id": dep.id, "status": dep.status.value})


async def stop(db: AsyncSession, dep: Deployment, *, actor: str, remove: bool = True) -> None:
    driver = get_driver()
    dep.status = DeployStatus.stopping
    await db.flush()

    if dep.litellm_registered:
        ok, msg = await litellm_svc.sync_deployment(dep, register=False)
        dep.litellm_registered = not ok
        if not ok:
            await audit.emit(
                db, severity="warning", source="litellm", source_id=dep.id,
                message=f"could not deregister {dep.served_model_name} from LiteLLM: {msg}",
            )

    try:
        if dep.container_name:
            await driver.stop(dep.node, dep.container_name, remove=remove)
        dep.status = DeployStatus.stopped
        dep.status_reason = f"stopped by {actor}"
    except Exception as exc:
        # Not `failed`: that status releases the GPUs, and the container may
        # well still be running and still holding them. Leaving the claim
        # standing keeps the next deploy off hardware that is still busy;
        # the reconciler clears it once the node says the container is gone.
        dep.status = DeployStatus.degraded
        dep.status_reason = (
            f"stop failed, container may still be running: {exc}"
        )[:1000]

    await audit.record(
        db, actor=actor, action="deployment.stop", target_type="deployment", target_id=dep.id,
        summary=f"stop {dep.served_model_name} on {dep.node.name}",
        ok=dep.status == DeployStatus.stopped,
    )
    if dep.status == DeployStatus.stopped:
        await audit.emit(
            db, severity="info", source="deployment", source_id=dep.id,
            message=f"{dep.served_model_name} stopped on {dep.node.name}",
        )
    else:
        await audit.emit(
            db, severity="error", source="deployment", source_id=dep.id,
            message=(f"could not stop {dep.served_model_name} on {dep.node.name}; "
                     f"its GPUs stay reserved until the container is confirmed gone"),
        )
    await db.commit()
    events.publish("deployment", {"id": dep.id, "status": dep.status.value})


async def restart(db: AsyncSession, dep: Deployment, *, actor: str) -> Deployment:
    """Stop and recreate with identical args — the common fix after a hung engine."""
    argv = list(dep.vllm_args.get("argv", []))
    node_id, gpus, spec_id = dep.node_id, list(dep.gpu_indices), dep.spec_id
    name, repo, tp, image = dep.served_model_name, dep.hf_repo, dep.tensor_parallel_size, dep.image
    team = dep.team_id

    await stop(db, dep, actor=actor, remove=True)

    node = await db.get(Node, node_id)
    async with allocation_lock():
        port = await allocate_port(db, node_id)
    driver = get_driver()
    new = Deployment(
        id=str(uuid.uuid4()),
        served_model_name=name, spec_id=spec_id, hf_repo=repo, node_id=node_id, node=node,
        gpu_indices=gpus, port=port, status=DeployStatus.pending, image=image,
        tensor_parallel_size=tp, vllm_args={"argv": argv}, team_id=team, created_by=actor,
    )
    new.container_name = f"{settings.container_prefix}-{slugify(name)}-{new.id[:8]}"
    db.add(new)
    await db.flush()

    env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    if settings.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    try:
        new.container_id = await driver.launch(
            node,
            LaunchSpec(
                name=new.container_name, image=image, gpu_indices=gpus, host_port=port, args=argv,
                env=env, volumes={settings.hf_cache_dir: "/root/.cache/huggingface"},
                labels={LABEL_KEY: new.id, "dgxctl.model": name, "dgxctl.owner": actor},
            ),
        )
        new.status = DeployStatus.starting
        new.status_reason = "restarted"
    except Exception as exc:
        new.status = DeployStatus.failed
        new.status_reason = str(exc)[:1000]

    await audit.record(
        db, actor=actor, action="deployment.restart", target_type="deployment", target_id=new.id,
        summary=f"restart {name} on {node.name}", detail={"previous": dep.id},
        ok=new.status != DeployStatus.failed,
    )
    await db.commit()
    events.publish("deployment", {"id": new.id, "status": new.status.value})
    return new


async def logs(dep: Deployment, tail: int = 300) -> str:
    if not dep.container_name:
        return "(no container yet)"
    try:
        return await get_driver().logs(dep.node, dep.container_name, tail=tail)
    except Exception as exc:
        return f"(could not read logs: {exc})"


def startup_expired(dep: Deployment) -> bool:
    """Has this deployment had long enough to load its weights?

    created_at is stored as UTC, but not every backend hands back tzinfo
    (SQLite does not). A naive value must be read as UTC — reading it as local
    time makes every deployment look hours old and flips it straight to
    degraded before it ever finishes loading.
    """
    if dep.created_at is None:
        return False
    created = dep.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds() > STARTUP_GRACE_SECONDS
