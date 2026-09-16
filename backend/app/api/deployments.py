from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user, require_deployer
from ..config import settings
from ..db import get_db
from ..models import (
    ACTIVE_STATUSES, Deployment, DeployStatus, MetricSample, ModelSpec, Node, Role, Team, User,
)
from ..schemas import (
    BulkAction, CheckOut, OptionOut, DeploymentOut, DeployRequest, EstimateOut, FindingOut, LogsOut,
    PlacementOut, PlanOut, RejectionOut, SeriesOut, SeriesPoint,
)
from ..services import deployments as deploy_svc
from ..services import diagnostics
from ..services import sizing, vllm_args
from ..services.capacity import node_capacity, node_capacity_now
from ..services.placement import plan as plan_placements

router = APIRouter(prefix="/api/deployments", tags=["deployments"])

# How many times an automatically placed deploy will look again after losing a
# race for the GPUs it chose. Every caller in a burst tends to pick the same
# best node, so each one needs roughly as many looks as there are rivals ahead
# of it. The bound exists to stop a livelock on a genuinely full fleet, not to
# ration attempts; exhausting it returns a 409 that is safe to retry.
PLACEMENT_RETRIES = 10


def _best_available(nodes, tp: int) -> tuple[float, int]:
    """The roomiest GPU anywhere the caller could use, and how many GPUs sit
    beside it — what any recommendation has to fit into."""
    best_gb, gpus_there = 0.0, 0
    for node in nodes:
        if not node.schedulable or len(node.gpus) < tp:
            continue
        free = max(node_capacity(node).values(), default=0) / 1024
        if free > best_gb:
            best_gb, gpus_there = free, len(node.gpus)
    return best_gb, gpus_there


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


async def _requirement(db: AsyncSession, body: DeployRequest, spec, hf_repo: str,
                       tp: int, max_len: int, nodes) -> tuple[float, sizing.Assessment]:
    """How much VRAM per GPU this actually needs, and why.

    The catalog number is an operator's measurement and a useful floor, but on
    its own it is a hand-typed figure that nothing checks: a 70B model with 10
    written against it would be placed on a card that cannot hold it, and vLLM
    would take the node down with it. The larger of the two wins, so a wrong
    catalog entry can be conservative but never dangerous.
    """
    assessment = await _assess(db, body, spec, hf_repo, tp, max_len, nodes, [])
    catalog_gb = (spec.min_gpu_memory_gb if spec else 0.0) or 0.0

    # The floor, not the comfortable size. vLLM sizes its KV cache to whatever
    # memory it is given, so the question that decides yes or no is whether it
    # can start at all: weights, one full-length sequence, and overhead. Gating
    # on the roomier figure would refuse deployments that run perfectly well,
    # just with less concurrency.
    floor = assessment.estimate.minimum_gb_per_gpu
    return max(catalog_gb, floor), assessment


async def _assess(db: AsyncSession, body: DeployRequest, spec, hf_repo: str, tp: int,
                  max_len: int, nodes, placements) -> sizing.Assessment:
    """Size the model and check the settings against what it declares.

    Done against the node it would actually land on, because two of the checks —
    FP8 needing Hopper, and having enough GPUs for the tensor-parallel size —
    depend on the hardware rather than the model.
    """
    gpu_model, gpus_available = "", 0
    if placements:
        node = next((n for n in nodes if n.id == placements[0].node_id), None)
        if node and node.gpus:
            gpu_model = node.gpus[0].name
            gpus_available = len(node.gpus)

    extra = {**(spec.extra_args if spec else {}), **body.extra_args}
    return await sizing.assess(
        hf_repo=hf_repo,
        tensor_parallel=tp,
        max_model_len=max_len,
        max_num_seqs=int(extra.get("--max-num-seqs", extra.get("max-num-seqs", 256))),
        quantization=body.quantization or (spec.quantization if spec else ""),
        revision=spec.revision if spec else "",
        catalog_gb=(spec.min_gpu_memory_gb if spec else 0.0),
        hf_token=settings.hf_token,
        gpu_model=gpu_model,
        gpus_available=gpus_available,
    )


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

    per_gpu_gb, assessment = await _requirement(db, body, spec, hf_repo, tp, max_len, nodes)
    placements, rejections = plan_placements(nodes, per_gpu_gb=per_gpu_gb, tp=tp)

    # Show the command that would actually run, which means deriving the memory
    # share the same way the deploy will: from this model's slice of the card it
    # would land on, not from a fixed fraction.
    share = body.gpu_memory_utilization
    if share is None:
        card_mb = 0
        if placements:
            node = next(n for n in nodes if n.id == placements[0].node_id)
            card_mb = max((g.memory_total_mb for g in node.gpus
                           if g.index in placements[0].gpu_indices), default=0)
        share = round(per_gpu_gb * 1024 / card_mb, 3) if card_mb else 0.90

    argv = deploy_svc.build_vllm_args(
        hf_repo=hf_repo, served_model_name=name, tensor_parallel_size=tp,
        max_model_len=max_len, quantization=body.quantization or (spec.quantization if spec else ""),
        gpu_memory_utilization=share,
        revision=spec.revision if spec else "",
        extra_args={**(spec.extra_args if spec else {}), **body.extra_args},
    )
    checks = list(assessment.checks)
    checks += [
        sizing.Check(i.severity, i.title, i.detail, i.fix)
        for i in vllm_args.validate({**(spec.extra_args if spec else {}), **body.extra_args})
    ]
    blocked = any(c.severity == "error" for c in checks)

    options: list[sizing.Option] = []
    if not placements:
        free_gb, gpus_there = _best_available(nodes, tp)
        options = sizing.recommend(
            assessment.estimate,
            config=await sizing.fetch_config(hf_repo, spec.revision if spec else "",
                                             settings.hf_token),
            largest_free_gb=free_gb,
            gpus_on_best_node=gpus_there,
            quantization=body.quantization or (spec.quantization if spec else ""),
        )

    return PlanOut(
        estimate=EstimateOut(**assessment.estimate.__dict__),
        checks=[CheckOut(**c.__dict__) for c in checks],
        options=[OptionOut(**o.__dict__) for o in options],
        blocked=blocked,
        placements=[
            PlacementOut(
                node_id=p.node_id, node_name=p.node_name, gpu_indices=p.gpu_indices,
                gpu_model=p.gpu_model, free_gb_per_gpu=round(p.free_mb_per_gpu / 1024, 1),
                reserve_gb_per_gpu=round(p.reserve_mb_per_gpu / 1024, 1),
                shares_with=p.shares_with, note=p.note,
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
        await _check_free(db, explicit, per_gpu_gb)

    # Placement reads the fleet without holding the allocation lock, so another
    # caller can claim the chosen GPUs in between. The claim itself is checked
    # under the lock and refuses, which is what prevents double-booking; here we
    # simply look again. Someone who named exact GPUs gets the refusal instead —
    # re-placing would put their model somewhere they did not ask for.
    # Judge the settings before anything is claimed or downloaded. A context
    # length the model does not have, or a tensor-parallel size that does not
    # divide its heads, is a container that dies several minutes into pulling
    # weights — knowable now, from what the model declares about itself.
    rows = await db.execute(select(Node).order_by(Node.name))
    candidates = list(rows.scalars().unique())
    if body.node_ids:
        candidates = [n for n in candidates if n.id in set(body.node_ids)]
    per_gpu_gb, assessment = await _requirement(db, body, spec, hf_repo, tp, max_len, candidates)

    bad_args = [i for i in vllm_args.validate({**(spec.extra_args if spec else {}), **body.extra_args})
                if i.severity == "error"]
    if bad_args:
        raise HTTPException(422, f"{bad_args[0].title}: {bad_args[0].detail} {bad_args[0].fix}")
    if assessment.blocking:
        first = assessment.blocking[0]
        raise HTTPException(422, f"{first.title}: {first.detail} {first.fix}")

    preview, why_not = plan_placements(candidates, per_gpu_gb=per_gpu_gb, tp=tp)
    if not preview and why_not and not any("GiB" in r.reason for r in why_not):
        # Drained, unreachable, too few GPUs: the node said no for a reason of
        # its own, and repeating a memory figure would bury it.
        raise HTTPException(409, "; ".join(f"{r.node_name}: {r.reason}" for r in why_not[:4]))
    if not preview:
        # Refusing is the point. vLLM asked for more than the card has does not
        # fail politely — it takes the node down with it.
        free_gb, gpus_there = _best_available(candidates, tp)
        options = sizing.recommend(
            assessment.estimate,
            config=await sizing.fetch_config(hf_repo, spec.revision if spec else "",
                                             settings.hf_token),
            largest_free_gb=free_gb, gpus_on_best_node=gpus_there,
            quantization=body.quantization or (spec.quantization if spec else ""),
        )
        advice = ("; ".join(f"{o.change} would need {o.needs_gb_per_gpu} GiB" for o in options)
                  or "free capacity, or use a smaller model")
        raise HTTPException(
            409,
            f"{hf_repo} needs at least {per_gpu_gb:.0f} GiB per GPU to start at these "
            f"settings, and the largest free GPU has {free_gb:.0f} GiB. Try: {advice}.",
        )

    # Read off the ORM object once. Rolling back a lost race expires every
    # instance in the session, and touching an expired attribute afterwards is
    # a refresh — IO the async session cannot perform mid-request.
    actor, actor_role, actor_team = user.email, user.role, user.team_id

    attempts = 1 if explicit else PLACEMENT_RETRIES
    for attempt in range(attempts):
        try:
            targets = await deploy_svc.resolve_targets(
                db, replicas=body.replicas, tensor_parallel_size=tp, per_gpu_gb=per_gpu_gb,
                explicit=explicit, node_filter=body.node_ids or None,
            )
            await _check_quota(db, actor_role, actor_team, targets)
            deps = await deploy_svc.create(
                db, actor=actor, served_model_name=name, hf_repo=hf_repo, targets=targets,
                tensor_parallel_size=tp, per_gpu_gb=per_gpu_gb, spec=spec, max_model_len=max_len,
                quantization=body.quantization or "", gpu_memory_utilization=body.gpu_memory_utilization,
                extra_args=body.extra_args, image=body.image,
                team_id=body.team_id or actor_team,
            )
            break
        except deploy_svc.DeployError as exc:
            lost_the_race = "refresh and pick again" in str(exc) and attempt < attempts - 1
            if lost_the_race:
                await db.rollback()
                continue
            # Drained node, no capacity, unknown target: all the caller's
            # problem to fix, none of them a server fault.
            raise HTTPException(409, str(exc)) from exc

    return [to_out(d) for d in deps]


async def _check_free(db: AsyncSession, targets: list[deploy_svc.Target], per_gpu_gb: float) -> None:
    """Manual GPU picks still have to have room — the UI can be stale.

    Room, not emptiness: several models share a card when the reservations fit,
    so what disqualifies a GPU is a shortfall, not an existing tenant.
    """
    need_mb = int(round(per_gpu_gb * 1024))
    for t in targets:
        node = await db.get(Node, t.node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {t.node_id}")
        known = {g.index for g in node.gpus}
        unknown = sorted(set(t.gpu_indices) - known)
        if unknown:
            raise HTTPException(400, f"{node.name} has no GPU {unknown}")

        capacity = await node_capacity_now(db, node)
        short = {i: capacity.get(i, 0) for i in t.gpu_indices if capacity.get(i, 0) < need_mb}
        if short:
            detail = ", ".join(
                f"GPU {i} has {mb / 1024:.0f} GiB free" for i, mb in sorted(short.items())
            )
            raise HTTPException(
                409,
                f"{node.name} cannot fit {per_gpu_gb:.0f} GiB per GPU: {detail}; "
                f"refresh and pick again",
            )


async def _check_quota(db: AsyncSession, role: Role, team_id: str | None, targets) -> None:
    """Takes the caller's role and team rather than the User row: this runs
    inside a retry loop whose rollback expires every ORM instance."""
    if role == Role.admin or not team_id:
        return
    # Fetched rather than reached through user.team: a User that was just
    # inserted has never been through a relationship loader, and touching one
    # then is a lazy load, which async cannot perform.
    team = await db.get(Team, team_id)
    if team is None or not team.max_gpus:
        return

    rows = await db.execute(
        select(Deployment).where(Deployment.team_id == team_id, Deployment.status.in_(ACTIVE_STATUSES))
    )
    in_use = sum(len(d.gpu_indices) for d in rows.scalars().unique())
    asking = sum(len(t.gpu_indices) for t in targets)
    if in_use + asking > team.max_gpus:
        raise HTTPException(
            409,
            f"team '{team.name}' quota is {team.max_gpus} GPUs; "
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
