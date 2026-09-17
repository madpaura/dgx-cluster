"""MCP endpoint: the dashboard's capabilities as agent tools.

Every tool calls the same endpoint function the browser calls, with an explicit
session and an identity of its own. That matters for three reasons: the agent
gets exactly the validation a human gets, the audit log shows agent actions as
`agent@mcp` rather than impersonating someone, and there is no second
implementation to drift.

Responses are shaped for an agent, not a UI: compact, decision-oriented, and
carrying the *reason* for anything that failed — a refusal an agent can act on
beats a stack trace it has to parse.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select

from .api import catalog as catalog_api
from .api import clusters as clusters_api
from .api import deployments as deployments_api
from .api import fleet as fleet_api
from .api import litellm_api
from .api import nodes as nodes_api
from .config import settings
from .db import SessionLocal
from .auth import upsert_user
from .models import Role
from .schemas import (
    CatalogImportIn, ClusterCreate, DeployRequest, ModelSpecIn, NodeCreate, NodeMove,
)

log = logging.getLogger(__name__)

AGENT_EMAIL = "agent@mcp"

mcp = MCPServer(
    name="dgxctl",
    version="0.4.0",
    instructions=(
        "Control plane for a GPU fleet running vLLM models.\n\n"
        "Start with fleet_summary to see whether anything is wrong, then "
        "list_nodes or list_models to narrow down. When a deployment is "
        "unhealthy, call deployment_logs — it returns a diagnosed cause and a "
        "concrete fix, which is faster and more reliable than reading the raw "
        "log yourself.\n\n"
        "Before deploying, call plan_deployment: it reports which nodes can "
        "host the model and, more usefully, why each other node cannot. "
        "deploy_model refuses rather than half-placing, and tells you why.\n\n"
        "Deploying the same served_model_name twice creates a second replica; "
        "LiteLLM load-balances across replicas automatically, so that is how "
        "you scale a model out or drain a node without downtime."
    ),
)


# --------------------------------------------------------------- plumbing

@asynccontextmanager
async def _ctx():
    """A session plus the agent's own identity, so actions are attributable."""
    async with SessionLocal() as db:
        # Agents call tools in parallel, so provisioning the identity has to
        # tolerate losing the insert race the same way a first sign-in does.
        user = await upsert_user(
            db, email=AGENT_EMAIL, name="MCP agent", role=_agent_role()
        )
        yield db, user


def _agent_role() -> Role:
    """What an agent may do. Read per request rather than cached, so lowering it
    and restarting actually demotes the identity instead of leaving an admin row
    behind from before."""
    try:
        return Role(settings.mcp_role)
    except ValueError:
        log.warning("DGXCTL_MCP_ROLE=%r is not a role; falling back to viewer",
                    settings.mcp_role)
        return Role.viewer


def _fail(exc: HTTPException) -> ToolError:
    """Turn an API refusal into something an agent can read and act on.

    ToolError specifically: the SDK treats anything else as a crash and replaces
    the message with "Error executing tool <name>", which would hide the very
    thing the agent needs — "only 2 of 8 GPUs free, needs 4" is actionable,
    "an error occurred" is not.
    """
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return ToolError(f"{detail} (HTTP {exc.status_code})")


def _node_brief(n: Any) -> dict:
    busy = [g.index for g in n.gpus if g.deployment_id]
    free = [g.index for g in n.gpus if not g.deployment_id]
    return {
        "id": n.id,
        "name": n.name,
        "status": n.status.value if hasattr(n.status, "value") else n.status,
        "cluster": n.cluster_name or None,
        "gpu_model": (n.gpus[0].name if n.gpus else None),
        "gpus_total": len(n.gpus),
        "gpus_free": free,
        "gpus_busy": busy,
        "models": sorted({g.model_name for g in n.gpus if g.model_name}),
        "error": n.last_error or None,
    }


def _dep_brief(d: Any) -> dict:
    m = d.last_metrics or {}
    out = {
        "id": d.id,
        "model": d.served_model_name,
        "hf_repo": d.hf_repo,
        "node": d.node_name,
        "gpus": d.gpu_indices,
        "tensor_parallel": d.tensor_parallel_size,
        "status": d.status.value if hasattr(d.status, "value") else d.status,
        "endpoint": d.endpoint,
        "in_litellm": d.litellm_registered,
    }
    if d.status_reason:
        out["reason"] = d.status_reason
    if m:
        out["metrics"] = {
            "tokens_per_second": m.get("gen_tps"),
            "requests_running": m.get("running"),
            "requests_queued": m.get("waiting"),
            "kv_cache_pct": m.get("kv_cache_pct"),
            "ttft_ms": m.get("ttft_avg_ms"),
        }
    return out


# ------------------------------------------------------------------- read

@mcp.tool(description="Fleet health at a glance: nodes up, GPUs in use, models "
                      "serving, throughput, and whether the LiteLLM proxy is reachable. "
                      "Call this first.")
async def fleet_summary() -> dict:
    async with _ctx() as (db, user):
        s = await fleet_api.summary(db=db, _=user)
        d = s.model_dump()
    d["needs_attention"] = [
        reason for reason, bad in [
            (f"{d['nodes_unreachable']} node(s) unreachable", d["nodes_unreachable"] > 0),
            (f"{d['deployments_failed']} deployment(s) failed", d["deployments_failed"] > 0),
            (f"{d['deployments_degraded']} deployment(s) degraded", d["deployments_degraded"] > 0),
            (f"{d['requests_waiting']} request(s) queued", d["requests_waiting"] > 0),
            ("LiteLLM proxy unreachable", not d["litellm_reachable"]),
        ] if bad
    ]
    return d


@mcp.tool(description="Every node with its GPU occupancy, cluster and any error. "
                      "Set detail=true for per-GPU utilisation, memory and temperature.")
async def list_nodes(detail: bool = False) -> dict:
    async with _ctx() as (db, user):
        nodes = await nodes_api.list_nodes(db=db, _=user)
    if not detail:
        return {"nodes": [_node_brief(n) for n in nodes]}
    return {"nodes": [
        {**_node_brief(n), "gpu_detail": [
            {"index": g.index, "serving": g.model_name, "util_pct": g.utilization,
             "vram_used_mb": g.memory_used_mb, "vram_total_mb": g.memory_total_mb,
             "temp_c": g.temperature_c, "ecc_errors": g.ecc_errors}
            for g in n.gpus]}
        for n in nodes]}


@mcp.tool(description="Deployments grouped by the model name clients call. Replicas of "
                      "one name are one load-balanced model group.")
async def list_models(include_stopped: bool = False) -> dict:
    async with _ctx() as (db, user):
        deps = await deployments_api.list_deployments(
            active_only=not include_stopped, node_id=None, db=db, _=user)
    groups: dict[str, list] = {}
    for d in deps:
        groups.setdefault(d.served_model_name, []).append(_dep_brief(d))
    return {"models": [
        {
            "model": name,
            "replicas": len(members),
            "healthy": sum(1 for m in members if m["status"] == "healthy"),
            "tokens_per_second": round(
                sum((m.get("metrics") or {}).get("tokens_per_second") or 0 for m in members), 1),
            "deployments": members,
        }
        for name, members in sorted(groups.items())]}


@mcp.tool(description="One deployment in full, including live metrics.")
async def get_deployment(deployment_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            d = await deployments_api.get_deployment(dep_id=deployment_id, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _dep_brief(d)


@mcp.tool(description="Container logs for a deployment, plus a diagnosis: each finding "
                      "names the cause and the fix. Use this before reading raw logs.")
async def deployment_logs(deployment_id: str, tail: int = 200) -> dict:
    async with _ctx() as (db, user):
        try:
            body = await deployments_api.deployment_logs(
                dep_id=deployment_id, tail=tail, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return {
        "deployment_id": body.deployment_id,
        "findings": [
            {"severity": f.severity, "problem": f.title, "detail": f.detail,
             "fix": f.fix, "evidence": f.evidence}
            for f in body.findings],
        "log_tail": body.text,
    }


@mcp.tool(description="Hardware-level problems on a node: unreachable, ECC errors, "
                      "thermal throttling, mixed GPU models.")
async def diagnose_node(node_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            findings = await nodes_api.node_diagnostics(node_id=node_id, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return {"findings": [
        {"severity": f.severity, "problem": f.title, "detail": f.detail, "fix": f.fix}
        for f in findings]}


@mcp.tool(description="Models available to deploy, with the per-GPU memory and "
                      "tensor-parallel size that placement uses.")
async def list_catalog() -> dict:
    async with _ctx() as (db, user):
        specs = await catalog_api.list_specs(db=db, _=user)
    return {"models": [
        {"key": s.key, "name": s.display_name, "hf_repo": s.hf_repo,
         "params_b": s.params_b, "quantization": s.quantization or None,
         "vram_gb_per_gpu": s.min_gpu_memory_gb, "tensor_parallel": s.recommended_tp,
         "tags": s.tags, "notes": s.notes or None}
        for s in specs]}


@mcp.tool(description="Read a Hugging Face or GitHub model page and draft a catalog "
                      "entry from it. Returns the draft WITHOUT saving: show it to "
                      "the operator, then call add_catalog_entry to keep it. Needs a "
                      "drafting model configured under Settings.")
async def draft_catalog_entry(url: str) -> dict:
    async with _ctx() as (db, user):
        try:
            draft = await catalog_api.import_from_url(
                body=CatalogImportIn(url=url), db=db, user=user
            )
        except HTTPException as exc:
            raise _fail(exc) from exc
    out = draft.model_dump()
    out["saved"] = False
    out["next_step"] = (
        "Nothing has been saved. Show these fields to the operator, apply any "
        "corrections they ask for, then call add_catalog_entry."
    )
    return out


@mcp.tool(description="Save a catalog entry. Use it to keep a draft from "
                      "draft_catalog_entry after the operator has checked it, or to "
                      "add one from scratch. `key` must be unique.")
async def add_catalog_entry(
    key: str,
    display_name: str,
    hf_repo: str,
    min_gpu_memory_gb: float,
    recommended_tp: int = 1,
    params_b: float = 0.0,
    quantization: str = "",
    max_model_len: int = 0,
    extra_args: dict | None = None,
    vllm_image: str = "",
    tags: list[str] | None = None,
    notes: str = "",
    revision: str = "",
) -> dict:
    async with _ctx() as (db, user):
        try:
            spec = await catalog_api.create_spec(
                body=ModelSpecIn(
                    key=key, display_name=display_name, hf_repo=hf_repo, revision=revision,
                    params_b=params_b, quantization=quantization,
                    min_gpu_memory_gb=min_gpu_memory_gb, recommended_tp=recommended_tp,
                    max_model_len=max_model_len, extra_args=extra_args or {},
                    vllm_image=vllm_image, tags=tags or [], notes=notes,
                ),
                db=db, user=user,
            )
        except HTTPException as exc:
            raise _fail(exc) from exc
    return {"saved": True, "key": spec.key, "id": spec.id, "name": spec.display_name}


@mcp.tool(description="Operator-defined node groupings, with how many nodes each "
                      "holds. Clusters are labels for organising the fleet; they "
                      "never constrain where a model can be placed.")
async def list_clusters() -> dict:
    async with _ctx() as (db, user):
        clusters = await clusters_api.list_clusters(db=db, _=user)
    return {"clusters": [c.model_dump() for c in clusters]}


@mcp.tool(description="Recent fleet events, newest first. severity filters to "
                      "'error', 'warning' or 'info'.")
async def list_events(limit: int = 50, severity: str | None = None) -> dict:
    async with _ctx() as (db, user):
        events = await fleet_api.list_events(limit=limit, severity=severity, db=db, _=user)
    return {"events": [
        {"time": e.ts.isoformat(), "severity": e.severity, "source": e.source,
         "source_id": e.source_id, "message": e.message}
        for e in events]}


@mcp.tool(description="Whether the LiteLLM proxy is reachable and which model groups "
                      "it is routing.")
async def litellm_status() -> dict:
    async with _ctx() as (db, user):
        return await litellm_api.status(_=user)


# ------------------------------------------------------------------ write

@mcp.tool(description="Dry run: where would this model land, and why can the other "
                      "nodes not take it. Always safe; changes nothing.")
async def plan_deployment(
    model: str,
    replicas: int = 1,
    node_ids: list[str] | None = None,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 0,
) -> dict:
    body = _deploy_request(model, replicas, node_ids, tensor_parallel_size, max_model_len, None)
    async with _ctx() as (db, user):
        try:
            plan = await deployments_api.plan_deployment(body=body, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return {
        "fits": len(plan.placements) > 0,
        "needs_gb_per_gpu": plan.per_gpu_gb,
        "tensor_parallel": plan.tensor_parallel_size,
        "would_place_on": [
            {"node": p.node_name, "gpus": p.gpu_indices, "gpu_model": p.gpu_model,
             "free_gb_per_gpu": p.free_gb_per_gpu}
            for p in plan.placements[:replicas]],
        "other_candidates": len(plan.placements) - min(replicas, len(plan.placements)),
        "rejected": [{"node": r.node_name, "reason": r.reason} for r in plan.rejections],
        "command": " ".join(plan.argv),
    }


@mcp.tool(description="Serve a model. `model` is a catalog key or a Hugging Face repo. "
                      "Placement is automatic; pass node_ids to confine it. Deploying an "
                      "existing served_model_name adds a load-balanced replica. Refuses "
                      "with a reason rather than half-placing.")
async def deploy_model(
    model: str,
    replicas: int = 1,
    node_ids: list[str] | None = None,
    served_model_name: str | None = None,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 0,
) -> dict:
    body = _deploy_request(model, replicas, node_ids, tensor_parallel_size,
                           max_model_len, served_model_name)
    async with _ctx() as (db, user):
        try:
            deps = await deployments_api.create_deployment(body=body, db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    started = [_dep_brief(d) for d in deps]
    return {
        "started": len(started),
        "requested": replicas,
        "short_by": max(0, replicas - len(started)),
        "deployments": started,
        "note": ("Weights load before a model answers; poll get_deployment until status "
                 "is 'healthy', or call deployment_logs if it turns 'failed'."),
    }


@mcp.tool(description="Stop a deployment. It is removed from LiteLLM first, then the "
                      "container stops. In-flight requests will fail.")
async def stop_deployment(deployment_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            d = await deployments_api.stop_deployment(dep_id=deployment_id, db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _dep_brief(d)


@mcp.tool(description="Recreate a deployment with identical arguments on the same GPUs. "
                      "The usual fix for a hung engine. Causes downtime while weights reload.")
async def restart_deployment(deployment_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            d = await deployments_api.restart_deployment(dep_id=deployment_id, db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _dep_brief(d)


@mcp.tool(description="Stop scheduling new work on a node without touching what is "
                      "already running. Use undo=true to put it back in rotation.")
async def drain_node(node_id: str, undo: bool = False) -> dict:
    async with _ctx() as (db, user):
        try:
            n = await nodes_api.drain_node(node_id=node_id, undo=undo, db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _node_brief(n)


@mcp.tool(description="Re-probe a node over SSH now, instead of waiting for the poller.")
async def probe_node(node_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            n = await nodes_api.probe_node(node_id=node_id, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _node_brief(n)


@mcp.tool(description="Compare a node's real containers against our records and report "
                      "orphans or deployments whose container has vanished.")
async def reconcile_node(node_id: str) -> dict:
    async with _ctx() as (db, user):
        try:
            return await nodes_api.reconcile_node(node_id=node_id, db=db, _=user)
        except HTTPException as exc:
            raise _fail(exc) from exc


@mcp.tool(description="Register a GPU node. Its SSH key must already be authorised on "
                      "the machine. Probes immediately and reports what it found.")
async def register_node(
    name: str,
    hostname: str,
    kind: str = "dgx",
    cluster_id: str | None = None,
    ssh_port: int = 22,
    ssh_user: str = "",
) -> dict:
    async with _ctx() as (db, user):
        try:
            n = await nodes_api.create_node(
                body=NodeCreate(name=name, hostname=hostname, kind=kind,
                                cluster_id=cluster_id, ssh_port=ssh_port, ssh_user=ssh_user),
                db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return _node_brief(n)


@mcp.tool(description="Create a cluster, optionally moving nodes into it. Clusters are "
                      "labels only; they never constrain scheduling.")
async def create_cluster(name: str, description: str = "",
                         node_ids: list[str] | None = None) -> dict:
    async with _ctx() as (db, user):
        try:
            c = await clusters_api.create_cluster(
                body=ClusterCreate(name=name, description=description,
                                   node_ids=node_ids or []), db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc
    return c.model_dump()


@mcp.tool(description="Move nodes into a cluster, or pass cluster_id=null to release "
                      "them to Unassigned.")
async def move_nodes(node_ids: list[str], cluster_id: str | None = None) -> dict:
    async with _ctx() as (db, user):
        try:
            if cluster_id:
                c = await clusters_api.move_nodes_in(
                    cluster_id=cluster_id, body=NodeMove(node_ids=node_ids), db=db, user=user)
                return c.model_dump()
            return await clusters_api.unassign_nodes(
                body=NodeMove(node_ids=node_ids), db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc


@mcp.tool(description="Make LiteLLM match the fleet: register healthy deployments it is "
                      "missing, drop entries whose deployment is gone. Safe to run any time.")
async def resync_litellm() -> dict:
    async with _ctx() as (db, user):
        try:
            return await litellm_api.resync(db=db, user=user)
        except HTTPException as exc:
            raise _fail(exc) from exc


def _deploy_request(model, replicas, node_ids, tp, max_len, served) -> DeployRequest:
    """A catalog key has no slash; a Hugging Face repo does."""
    is_repo = "/" in model
    return DeployRequest(
        spec_key=None if is_repo else model,
        hf_repo=model if is_repo else None,
        served_model_name=served,
        replicas=replicas,
        node_ids=node_ids or [],
        tensor_parallel_size=tp,
        max_model_len=max_len,
    )


# ------------------------------------------------------------------- mount

def build_mcp_app():
    """The MCP app to mount, or None when it should not be exposed."""
    if not settings.mcp_enabled:
        return None
    if not settings.mcp_token and settings.auth_mode != "dev":
        log.error(
            "MCP endpoint NOT mounted: DGXCTL_MCP_TOKEN is unset while auth_mode=%s. "
            "An unauthenticated MCP endpoint can stop every model in the fleet.",
            settings.auth_mode,
        )
        return None
    if not settings.mcp_token:
        log.warning("MCP endpoint is mounted without a token (auth_mode=dev). "
                    "Set DGXCTL_MCP_TOKEN before exposing this beyond localhost.")

    hosts = settings.mcp_host_list
    permissive = "*" in hosts
    if permissive:
        log.info("MCP Host-header check disabled; the bearer token is the gate.")
    inner = mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,     # each call is self-contained; no session to lose
        json_response=True,      # plain JSON replies rather than an SSE stream
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=not permissive,
            allowed_hosts=[] if permissive else hosts,
            allowed_origins=[] if permissive else hosts,
        ),
    )
    return _RequireToken(inner, settings.mcp_token)


class _RequireToken:
    """Bearer-token gate for the mounted MCP app.

    The sub-app sits outside FastAPI's dependency graph, so the check lives here
    rather than in a Depends.
    """

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            presented = headers.get(b"authorization", b"").decode()
            if presented != f"Bearer {self.token}":
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body",
                            "body": b'{"error":"MCP requires a bearer token"}'})
                return
        await self.app(scope, receive, send)
