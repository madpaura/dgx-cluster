"""Deployment lifecycle: turn 'serve this model' into running vLLM containers.

The control server never holds long locks against the fleet. It writes the
intended state to Postgres, fires the docker command, then lets the reconciler
(worker.py) converge status by observing the node. That is what keeps the UI
honest when a node reboots behind your back.
"""
from __future__ import annotations

import logging
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
from .placement import Placement, plan

log = logging.getLogger(__name__)

LABEL_KEY = "dgxctl.deployment"
STARTUP_GRACE_SECONDS = 900  # big models legitimately take many minutes to load


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
    spec: ModelSpec | None = None,
    max_model_len: int = 0,
    quantization: str = "",
    gpu_memory_utilization: float = 0.90,
    extra_args: dict | None = None,
    image: str = "",
    team_id: str | None = None,
) -> list[Deployment]:
    """Create and launch one deployment per target. Failures are per-target:
    deploying to six nodes where one is down still gives you five."""
    driver = get_driver()
    image = image or (spec.vllm_image if spec and spec.vllm_image else settings.vllm_image)
    created: list[Deployment] = []

    for target in targets:
        node = await db.get(Node, target.node_id)
        if node is None:
            raise DeployError(f"unknown node {target.node_id}")
        if node.status in (NodeStatus.draining, NodeStatus.maintenance):
            raise DeployError(f"{node.name} is {node.status.value}; not accepting deployments")

        port = await allocate_port(db, node.id)
        # Assign the id up front: the container name embeds it, and a Python-side
        # column default is not materialised until flush.
        dep = Deployment(
            id=str(uuid.uuid4()),
            served_model_name=served_model_name,
            spec_id=spec.id if spec else None,
            hf_repo=hf_repo,
            node_id=node.id,
            node=node,  # populate the relationship so callers can serialise without a lazy load
            gpu_indices=list(target.gpu_indices),
            port=port,
            status=DeployStatus.pending,
            image=image,
            tensor_parallel_size=tensor_parallel_size,
            team_id=team_id,
            created_by=actor,
        )
        dep.container_name = f"{settings.container_prefix}-{slugify(served_model_name)}-{dep.id[:8]}"
        args = build_vllm_args(
            hf_repo=hf_repo,
            served_model_name=served_model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            quantization=quantization or (spec.quantization if spec else ""),
            gpu_memory_utilization=gpu_memory_utilization,
            revision=spec.revision if spec else "",
            extra_args={**(spec.extra_args if spec else {}), **(extra_args or {})},
        )
        dep.vllm_args = {"argv": args}
        db.add(dep)
        await db.flush()

        env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        if settings.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        launch = LaunchSpec(
            name=dep.container_name,
            image=image,
            gpu_indices=list(target.gpu_indices),
            host_port=port,
            args=args,
            env=env,
            volumes={settings.hf_cache_dir: "/root/.cache/huggingface"},
            labels={
                LABEL_KEY: dep.id,
                "dgxctl.model": served_model_name,
                "dgxctl.owner": actor,
            },
        )
        try:
            dep.status = DeployStatus.pulling
            dep.container_id = await driver.launch(node, launch)
            dep.status = DeployStatus.starting
            dep.status_reason = "container started, loading weights"
            await audit.record(
                db, actor=actor, action="deployment.create", target_type="deployment", target_id=dep.id,
                summary=f"deploy {served_model_name} to {node.name} GPUs {target.gpu_indices}",
                detail={"argv": args, "image": image, "port": port},
            )
            await audit.emit(
                db, severity="info", source="deployment", source_id=dep.id,
                message=f"{served_model_name} starting on {node.name} GPU {','.join(map(str, target.gpu_indices))}",
            )
        except Exception as exc:
            dep.status = DeployStatus.failed
            dep.status_reason = str(exc)[:1000]
            await audit.record(
                db, actor=actor, action="deployment.create", target_type="deployment", target_id=dep.id,
                summary=f"failed to start {served_model_name} on {node.name}", detail={"error": str(exc)}, ok=False,
            )
            await audit.emit(
                db, severity="error", source="deployment", source_id=dep.id,
                message=f"failed to start {served_model_name} on {node.name}: {str(exc)[:200]}",
            )
        created.append(dep)

    await db.commit()
    for dep in created:
        events.publish("deployment", {"id": dep.id, "status": dep.status.value})
    return created


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
        dep.status = DeployStatus.failed
        dep.status_reason = f"stop failed: {exc}"[:1000]

    await audit.record(
        db, actor=actor, action="deployment.stop", target_type="deployment", target_id=dep.id,
        summary=f"stop {dep.served_model_name} on {dep.node.name}", ok=dep.status == DeployStatus.stopped,
    )
    await audit.emit(
        db, severity="info", source="deployment", source_id=dep.id,
        message=f"{dep.served_model_name} stopped on {dep.node.name}",
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

    driver = get_driver()
    node = await db.get(Node, node_id)
    port = await allocate_port(db, node_id)
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
