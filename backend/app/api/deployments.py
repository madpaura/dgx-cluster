from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user, require_deployer
from ..db import get_db
from ..models import ACTIVE_STATUSES, Deployment, DeployStatus, MetricSample, ModelSpec, Node, Role, User
from ..schemas import (
    BulkAction, DeploymentOut, DeployRequest, FindingOut, LogsOut, PlacementOut, PlanOut,
    RejectionOut, SeriesOut, SeriesPoint,
)
from ..services import deployments as deploy_svc
from ..services import diagnostics
from ..services.placement import plan as plan_placements

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


def to_out(dep: Deployment) -> DeploymentOut:
    out = DeploymentOut.model_validate(dep)
    out.node_name = dep.node.name if dep.node else ""
    out.endpoint = dep.endpoint if dep.node else ""
    return out


async def _resolve_model(db: AsyncSession, body: DeployRequest):
    spec = None
    if body.spec_key:
        row = await db.execute(select(ModelSpec).where(ModelSpec.key == body.spec_key))
        spec = row.scalar_one_or_none()
        if spec is None:
            raise HTTPException(404, f"no catalog entry '{body.spec_key}'")
    hf_repo = body.hf_repo or (spec.hf_repo if spec else None)
    if not hf_repo:
        raise HTTPException(400, "give either spec_key or hf_repo")
    name = body.served_model_name or (spec.key if spec else hf_repo.split("/")[-1].lower())
    tp = body.tensor_parallel_size or (spec.recommended_tp if spec else 1)
    per_gpu_gb = (spec.min_gpu_memory_gb if spec else 0) or _estimate_per_gpu_gb(hf_repo, tp)
    max_len = body.max_model_len or (spec.max_model_len if spec else 0)
    return spec, hf_repo, name, tp, per_gpu_gb, max_len


def _estimate_per_gpu_gb(hf_repo: str, tp: int) -> float:
    """Fallback when the model is not in the catalog: size from the repo name."""
    r = hf_repo.lower()
    params = 7.0
    for token, b in (("405b", 405), ("235b", 235), ("123b", 123), ("72b", 72), ("70b", 70),
                     ("34b", 34), ("32b", 32), ("30b", 30), ("14b", 14), ("13b", 13),
                     ("8b", 8), ("7b", 7), ("3b", 3), ("1.5b", 1.5), ("0.6b", 0.6)):
        if token in r:
            params = b
            break
    bytes_per_param = 1.0 if any(q in r for q in ("fp8", "awq", "gptq", "int4", "w4a16")) else 2.0
    weights_gb = params * bytes_per_param
    return round(weights_gb / max(tp, 1) + 8.0, 1)  # + KV cache working room


@router.get("", response_model=list[DeploymentOut])
async def list_deployments(
    active_only: bool = True,
    node_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(current_user),
):
    q = select(Deployment).order_by(Deployment.created_at.desc())
    if active_only:
        q = q.where(Deployment.status.in_(ACTIVE_STATUSES))
    if node_id:
        q = q.where(Deployment.node_id == node_id)
    rows = await db.execute(q.limit(500))
    return [to_out(d) for d in rows.scalars().unique()]


@router.post("/plan", response_model=PlanOut)
async def plan_deployment(
    body: DeployRequest, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)
):
    """What would happen if I deployed this? Placements, rejections, exact argv.

    The deploy dialog calls this on every change, so nobody has to launch a
    container to find out it will not fit.
    """
    spec, hf_repo, name, tp, per_gpu_gb, max_len = await _resolve_model(db, body)
    rows = await db.execute(select(Node).order_by(Node.name))
    nodes = list(rows.scalars().unique())
    if body.node_ids:
        nodes = [n for n in nodes if n.id in set(body.node_ids)]

    placements, rejections = plan_placements(nodes, per_gpu_gb=per_gpu_gb, tp=tp)
    argv = deploy_svc.build_vllm_args(
        hf_repo=hf_repo, served_model_name=name, tensor_parallel_size=tp,
        max_model_len=max_len, quantization=body.quantization or (spec.quantization if spec else ""),
        gpu_memory_utilization=body.gpu_memory_utilization,
        revision=spec.revision if spec else "",
        extra_args={**(spec.extra_args if spec else {}), **body.extra_args},
    )
    return PlanOut(
        placements=[
            PlacementOut(
                node_id=p.node_id, node_name=p.node_name, gpu_indices=p.gpu_indices,
                gpu_model=p.gpu_model, free_gb_per_gpu=round(p.free_mb_per_gpu / 1024, 1), note=p.note,
            )
            for p in placements
        ],
        rejections=[RejectionOut(node_name=r.node_name, reason=r.reason) for r in rejections],
        per_gpu_gb=per_gpu_gb,
        tensor_parallel_size=tp,
        argv=argv,
    )


@router.post("", response_model=list[DeploymentOut], status_code=201)
async def create_deployment(
    body: DeployRequest, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)
):
    spec, hf_repo, name, tp, per_gpu_gb, max_len = await _resolve_model(db, body)

    if body.replicas < 1 or body.replicas > 32:
        raise HTTPException(400, "replicas must be between 1 and 32")

    explicit = [deploy_svc.Target(node_id=t.node_id, gpu_indices=t.gpu_indices) for t in body.targets] or None
    if explicit:
        await _check_free(db, explicit)

    try:
        targets = await deploy_svc.resolve_targets(
            db, replicas=body.replicas, tensor_parallel_size=tp, per_gpu_gb=per_gpu_gb,
            explicit=explicit, node_filter=body.node_ids or None,
        )
    except deploy_svc.DeployError as exc:
        raise HTTPException(409, str(exc)) from exc

    await _check_quota(db, user, targets, tp)

    try:
        deps = await deploy_svc.create(
            db, actor=user.email, served_model_name=name, hf_repo=hf_repo, targets=targets,
            tensor_parallel_size=tp, spec=spec, max_model_len=max_len,
            quantization=body.quantization or "", gpu_memory_utilization=body.gpu_memory_utilization,
            extra_args=body.extra_args, image=body.image,
            team_id=body.team_id or user.team_id,
        )
    except deploy_svc.DeployError as exc:
        # Drained node, no free port, unknown target: all the caller's problem
        # to fix, none of them a server fault.
        raise HTTPException(409, str(exc)) from exc
    return [to_out(d) for d in deps]


async def _check_free(db: AsyncSession, targets: list[deploy_svc.Target]) -> None:
    """Manual GPU picks still have to be free — the UI can be stale."""
    for t in targets:
        node = await db.get(Node, t.node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {t.node_id}")
        busy = {int(i) for d in node.deployments if d.status in ACTIVE_STATUSES for i in d.gpu_indices}
        clash = sorted(set(t.gpu_indices) & busy)
        if clash:
            raise HTTPException(409, f"{node.name} GPU {clash} already in use; refresh and pick again")
        known = {g.index for g in node.gpus}
        unknown = sorted(set(t.gpu_indices) - known)
        if unknown:
            raise HTTPException(400, f"{node.name} has no GPU {unknown}")


async def _check_quota(db: AsyncSession, user: User, targets, tp: int) -> None:
    if user.role == Role.admin or not user.team_id or not user.team or not user.team.max_gpus:
        return
    rows = await db.execute(
        select(Deployment).where(Deployment.team_id == user.team_id, Deployment.status.in_(ACTIVE_STATUSES))
    )
    in_use = sum(len(d.gpu_indices) for d in rows.scalars().unique())
    asking = sum(len(t.gpu_indices) for t in targets)
    if in_use + asking > user.team.max_gpus:
        raise HTTPException(
            409,
            f"team '{user.team.name}' quota is {user.team.max_gpus} GPUs; "
            f"{in_use} in use, this request needs {asking}",
        )


@router.get("/{dep_id}", response_model=DeploymentOut)
async def get_deployment(dep_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    dep = await db.get(Deployment, dep_id)
    if not dep:
        raise HTTPException(404, "deployment not found")
    return to_out(dep)


@router.get("/{dep_id}/logs", response_model=LogsOut)
async def deployment_logs(
    dep_id: str, tail: int = Query(300, ge=10, le=5000),
    db: AsyncSession = Depends(get_db), _: User = Depends(current_user),
):
    dep = await db.get(Deployment, dep_id)
    if not dep:
        raise HTTPException(404, "deployment not found")
    text = await deploy_svc.logs(dep, tail=tail)
    findings = diagnostics.analyze_logs(text + "\n" + (dep.status_reason or ""))
    if dep.status == DeployStatus.healthy:
        # A model that is serving does not need to be told it once loaded weights.
        # Keep only what still warrants attention.
        findings = [f for f in findings if f.severity != "info"]
    return LogsOut(
        deployment_id=dep_id, text=text, findings=[FindingOut(**f.__dict__) for f in findings]
    )


@router.get("/{dep_id}/series", response_model=SeriesOut)
async def deployment_series(
    dep_id: str, minutes: int = Query(60, ge=5, le=2880),
    db: AsyncSession = Depends(get_db), _: User = Depends(current_user),
):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rows = await db.execute(
        select(MetricSample)
        .where(MetricSample.scope == "deployment", MetricSample.scope_id == dep_id, MetricSample.ts >= since)
        .order_by(MetricSample.ts)
        .limit(2000)
    )
    return SeriesOut(
        scope="deployment", scope_id=dep_id,
        points=[SeriesPoint(ts=s.ts, values=s.values) for s in rows.scalars()],
    )


# NOTE: the /bulk/* routes must be declared before /{dep_id}/*. FastAPI
# matches in registration order, so a parameterised path declared first
# swallows "bulk" as a deployment id and every bulk action 404s.
@router.post("/bulk/stop", response_model=list[DeploymentOut])
async def bulk_stop(body: BulkAction, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    out = []
    for dep_id in body.deployment_ids[:100]:
        dep = await db.get(Deployment, dep_id)
        if dep and dep.status in ACTIVE_STATUSES:
            _assert_owner(user, dep)
            await deploy_svc.stop(db, dep, actor=user.email)
            out.append(to_out(dep))
    return out


@router.post("/bulk/restart", response_model=list[DeploymentOut])
async def bulk_restart(body: BulkAction, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    out = []
    for dep_id in body.deployment_ids[:100]:
        dep = await db.get(Deployment, dep_id)
        if dep:
            _assert_owner(user, dep)
            out.append(to_out(await deploy_svc.restart(db, dep, actor=user.email)))
    return out


@router.post("/{dep_id}/stop", response_model=DeploymentOut)
async def stop_deployment(dep_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    dep = await db.get(Deployment, dep_id)
    if not dep:
        raise HTTPException(404, "deployment not found")
    _assert_owner(user, dep)
    await deploy_svc.stop(db, dep, actor=user.email)
    return to_out(dep)


@router.post("/{dep_id}/restart", response_model=DeploymentOut)
async def restart_deployment(dep_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    dep = await db.get(Deployment, dep_id)
    if not dep:
        raise HTTPException(404, "deployment not found")
    _assert_owner(user, dep)
    new = await deploy_svc.restart(db, dep, actor=user.email)
    return to_out(new)


def _assert_owner(user: User, dep: Deployment) -> None:
    """Deployers act within their team; admins act anywhere."""
    if user.role == Role.admin:
        return
    if dep.team_id and user.team_id and dep.team_id == user.team_id:
        return
    if dep.created_by == user.email:
        return
    raise HTTPException(403, f"{dep.served_model_name} belongs to another team")
