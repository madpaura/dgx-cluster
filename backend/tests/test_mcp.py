"""The MCP endpoint, exercised over its real wire protocol.

Tools are not called as Python functions here: every assertion goes through an
HTTP JSON-RPC round trip against the mounted app, because that is what an agent
will actually do.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport

from app.main import app
from app.mcp_server import AGENT_EMAIL, build_mcp_app, mcp
from tests.conftest import deploy_undersized, pump, register_fleet

HDRS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}


def auth_headers() -> dict:
    """Authenticate when a token is configured. Running the suite inside the
    shipped image means production settings apply, token included."""
    from app.config import settings

    if settings.mcp_token:
        return {**HDRS, "authorization": f"Bearer {settings.mcp_token}"}
    return HDRS


class Agent:
    """A minimal MCP client: initialize once, then call tools."""

    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self._id = 0

    async def rpc(self, method: str, params: dict | None = None) -> httpx.Response:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params
        return await self.http.post("/", headers=auth_headers(), json=body)

    async def initialize(self) -> dict:
        r = await self.rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test-agent", "version": "1"}})
        return r.json()["result"]

    async def tools(self) -> list[dict]:
        return (await self.rpc("tools/list")).json()["result"]["tools"]

    async def call(self, tool: str, /, **arguments):
        """Returns the tool's parsed payload, or raises with the agent-visible
        error text."""
        r = await self.rpc("tools/call", {"name": tool, "arguments": arguments})
        result = r.json()["result"]
        text = result["content"][0]["text"]
        if result.get("isError"):
            raise AssertionError(f"tool error: {text}")
        return json.loads(text)

    async def call_expecting_error(self, tool: str, /, **arguments) -> str:
        r = await self.rpc("tools/call", {"name": tool, "arguments": arguments})
        result = r.json()["result"]
        assert result.get("isError"), f"expected a refusal, got {result}"
        return result["content"][0]["text"]


@pytest.fixture
async def agent():
    """A fresh MCP app per test.

    Two constraints from the SDK shape this. A session manager can only be run
    once per instance, so each test builds a new app (which creates a new
    manager). And its task group's cancel scope must be exited by the task that
    entered it, which a pytest-asyncio generator fixture cannot promise — hence
    the dedicated runner task.

    Only the MCP machinery runs here, not the worker loops, so tests stay
    deterministic.
    """
    guarded = build_mcp_app()
    started, stop = asyncio.Event(), asyncio.Event()

    async def runner():
        async with mcp.session_manager.run():
            started.set()
            await stop.wait()

    task = asyncio.create_task(runner())
    await started.wait()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=guarded),
                                     base_url="http://dgxctl.test") as http:
            a = Agent(http)
            await a.initialize()
            yield a
    finally:
        stop.set()
        await task


# ------------------------------------------------------------------ protocol

async def test_handshake_identifies_the_server(agent):
    info = await agent.initialize()
    assert info["serverInfo"]["name"] == "dgxctl"
    assert info["serverInfo"]["version"] == "0.4.0"
    assert "tools" in info["capabilities"]


async def test_the_server_tells_an_agent_how_to_work(agent):
    info = await agent.initialize()
    hint = info.get("instructions", "")
    assert "fleet_summary" in hint
    assert "deployment_logs" in hint and "fix" in hint
    assert "plan_deployment" in hint


async def test_every_tool_is_described_and_typed(agent):
    tools = await agent.tools()
    assert len(tools) == 21
    for t in tools:
        assert t["description"] and len(t["description"]) > 40, f"{t['name']} is underdescribed"
        assert t["inputSchema"]["type"] == "object"


async def test_required_arguments_are_marked_required(agent):
    tools = {t["name"]: t for t in await agent.tools()}
    assert tools["deploy_model"]["inputSchema"]["required"] == ["model"]
    assert tools["stop_deployment"]["inputSchema"]["required"] == ["deployment_id"]
    assert "required" not in tools["fleet_summary"]["inputSchema"] or \
           not tools["fleet_summary"]["inputSchema"]["required"]


async def test_an_unknown_tool_is_refused_not_crashed(agent):
    msg = await agent.call_expecting_error("drop_the_database")
    assert "Unknown tool" in msg or "not found" in msg.lower()


# ---------------------------------------------------------------- read tools

async def test_fleet_summary_leads_with_what_is_wrong(agent, client):
    await register_fleet()
    s = await agent.call("fleet_summary")
    assert s["nodes_total"] == 7
    assert s["nodes_unreachable"] == 1
    assert any("unreachable" in item for item in s["needs_attention"])


async def test_a_healthy_fleet_has_nothing_to_report(agent, client):
    await register_fleet(["dgx-01"])
    s = await agent.call("fleet_summary")
    assert s["needs_attention"] == ["LiteLLM proxy unreachable"]


async def test_list_nodes_is_compact_but_decision_ready(agent, client):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})
    node = (await agent.call("list_nodes"))["nodes"][0]

    assert node["name"] == "dgx-01"
    assert node["gpus_total"] == 8
    assert node["gpus_busy"] == [0, 1]
    assert node["gpus_free"] == [2, 3, 4, 5, 6, 7]
    assert node["models"] == ["qwen3-32b"]
    assert "gpu_detail" not in node, "per-GPU noise must be opt-in"


async def test_node_detail_is_available_on_request(agent, client):
    await register_fleet(["dgx-01"])
    node = (await agent.call("list_nodes", detail=True))["nodes"][0]
    assert len(node["gpu_detail"]) == 8
    assert {"index", "util_pct", "vram_used_mb", "temp_c", "ecc_errors"} <= set(node["gpu_detail"][0])


async def test_list_models_groups_replicas(agent, client):
    await register_fleet(["dgx-01", "dgx-02"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 2})
    await pump(2, gap=0.6)

    models = (await agent.call("list_models"))["models"]
    assert len(models) == 1
    assert models[0]["model"] == "llama3.1-8b"
    assert models[0]["replicas"] == 2 and models[0]["healthy"] == 2
    assert models[0]["tokens_per_second"] > 0


async def test_the_catalog_exposes_what_placement_needs(agent):
    models = {m["key"]: m for m in (await agent.call("list_catalog"))["models"]}
    assert models["qwen3-32b"]["vram_gb_per_gpu"] == 40
    assert models["qwen3-32b"]["tensor_parallel"] == 2
    assert "Hopper" in models["llama3.1-70b-fp8"]["notes"]


async def test_events_are_readable_and_filterable(agent, client):
    await register_fleet(["dgx-04"])
    errors = (await agent.call("list_events", severity="error"))["events"]
    assert errors and all(e["severity"] == "error" for e in errors)
    assert "unreachable" in errors[0]["message"]


# --------------------------------------------------------- diagnosis (the point)

async def test_logs_come_back_diagnosed_with_a_fix(agent, client):
    """An agent must not have to parse a vLLM traceback."""
    ids = await register_fleet(["rtx-ws-01"])
    dep = await deploy_undersized(client, ids["rtx-ws-01"], [0, 1])
    await pump()

    body = await agent.call("deployment_logs", deployment_id=dep["id"])
    oom = next(f for f in body["findings"] if f["severity"] == "error")
    assert "does not fit" in oom["problem"]
    assert "tensor-parallel" in oom["fix"]
    assert "OutOfMemoryError" in oom["evidence"]


async def test_node_diagnosis_explains_unreachability(agent, client):
    ids = await register_fleet(["dgx-04"])
    findings = (await agent.call("diagnose_node", node_id=ids["dgx-04"]))["findings"]
    assert findings[0]["severity"] == "error"
    assert "authorized_keys" in findings[0]["fix"]


# --------------------------------------------------------------- write tools

async def test_plan_is_a_dry_run_that_changes_nothing(agent, client):
    await register_fleet(["dgx-01"])
    plan = await agent.call("plan_deployment", model="qwen3-32b")
    assert plan["fits"] is True
    assert plan["would_place_on"][0]["node"] == "dgx-01"
    assert plan["would_place_on"][0]["gpus"] == [0, 1]
    assert "--model Qwen/Qwen3-32B" in plan["command"]
    assert (await client.get("/api/deployments")).json() == [], "a plan must not deploy"


async def test_plan_explains_why_it_does_not_fit(agent, client):
    await register_fleet(["rtx-ws-01"])
    plan = await agent.call("plan_deployment", model="meta-llama/Llama-3.1-405B",
                            tensor_parallel_size=8)
    assert plan["fits"] is False
    assert plan["rejected"][0]["reason"] == "has 2 GPUs, needs 8"


async def test_deploy_accepts_a_catalog_key(agent, client):
    await register_fleet(["dgx-01"])
    r = await agent.call("deploy_model", model="qwen3-32b")
    assert r["started"] == 1 and r["short_by"] == 0
    assert r["deployments"][0]["model"] == "qwen3-32b"
    assert r["deployments"][0]["gpus"] == [0, 1]


async def test_deploy_accepts_a_hugging_face_repo(agent, client):
    await register_fleet(["dgx-01"])
    r = await agent.call("deploy_model", model="Qwen/Qwen2.5-14B-Instruct",
                         served_model_name="qwen-14b")
    assert r["deployments"][0]["model"] == "qwen-14b"
    assert r["deployments"][0]["hf_repo"] == "Qwen/Qwen2.5-14B-Instruct"


async def test_deploy_reports_a_shortfall_rather_than_pretending(agent, client):
    await register_fleet(["rtx-ws-01"])
    r = await agent.call("deploy_model", model="qwen3-32b", replicas=3)
    assert r["started"] == 1
    assert r["requested"] == 3
    assert r["short_by"] == 2


async def test_deploy_refuses_with_a_reason_an_agent_can_act_on(agent, client):
    await register_fleet(["rtx-ws-01"])
    msg = await agent.call_expecting_error(
        "deploy_model", model="meta-llama/Llama-3.1-405B", tensor_parallel_size=8)
    assert "has 2 GPUs, needs 8" in msg
    assert "HTTP 409" in msg


async def test_a_second_deploy_of_one_name_is_a_replica(agent, client):
    await register_fleet(["dgx-01", "dgx-02"])
    await agent.call("deploy_model", model="llama3.1-8b")
    await agent.call("deploy_model", model="llama3.1-8b")
    models = (await agent.call("list_models"))["models"]
    assert len(models) == 1 and models[0]["replicas"] == 2


async def test_deploy_can_be_confined_to_named_nodes(agent, client):
    ids = await register_fleet(["dgx-01", "dgx-02"])
    r = await agent.call("deploy_model", model="llama3.1-8b", node_ids=[ids["dgx-02"]])
    assert r["deployments"][0]["node"] == "dgx-02"


async def test_stop_and_restart(agent, client):
    await register_fleet(["dgx-01"])
    dep = (await agent.call("deploy_model", model="llama3.1-8b"))["deployments"][0]

    new = await agent.call("restart_deployment", deployment_id=dep["id"])
    assert new["id"] != dep["id"] and new["status"] == "starting"

    stopped = await agent.call("stop_deployment", deployment_id=new["id"])
    assert stopped["status"] == "stopped"


async def test_drain_and_undrain(agent, client):
    ids = await register_fleet(["dgx-01"])
    assert (await agent.call("drain_node", node_id=ids["dgx-01"]))["status"] == "draining"
    plan = await agent.call("plan_deployment", model="llama3.1-8b")
    assert plan["rejected"][0]["reason"] == "node is draining"
    assert (await agent.call("drain_node", node_id=ids["dgx-01"], undo=True))["status"] == "online"


async def test_register_a_node_and_group_it(agent, client):
    cluster = await agent.call("create_cluster", name="Agent Pod", description="made by an agent")
    node = await agent.call("register_node", name="dgx-03", hostname="dgx-03.sim.local",
                            cluster_id=cluster["id"])
    assert node["status"] == "online" and node["gpus_total"] == 8
    assert node["cluster"] == "Agent Pod"


async def test_move_nodes_between_clusters_and_out(agent, client):
    ids = await register_fleet(["dgx-01"])
    a = await agent.call("create_cluster", name="A", node_ids=[ids["dgx-01"]])
    assert a["node_count"] == 1

    b = await agent.call("create_cluster", name="B")
    moved = await agent.call("move_nodes", node_ids=[ids["dgx-01"]], cluster_id=b["id"])
    assert moved["node_count"] == 1
    assert (await agent.call("list_nodes"))["nodes"][0]["cluster"] == "B"

    await agent.call("move_nodes", node_ids=[ids["dgx-01"]])
    assert (await agent.call("list_nodes"))["nodes"][0]["cluster"] is None


async def test_reconcile_reports_a_vanished_container(agent, client):
    from app.drivers import get_driver

    ids = await register_fleet(["dgx-01"])
    dep = (await agent.call("deploy_model", model="llama3.1-8b"))["deployments"][0]
    get_driver()._nodes["dgx-01"].containers.clear()
    r = await agent.call("reconcile_node", node_id=ids["dgx-01"])
    assert r["missing"] == [dep["id"]]


# ------------------------------------------------------------ accountability

async def test_agent_actions_are_attributed_in_the_audit_log(agent, client):
    """An operator must be able to see what the agent did, distinctly from what
    a person did."""
    await register_fleet(["dgx-01"])
    await agent.call("deploy_model", model="llama3.1-8b")

    entries = (await client.get("/api/audit")).json()
    created = next(e for e in entries if e["action"] == "deployment.create")
    assert created["actor"] == AGENT_EMAIL


async def test_an_unknown_node_id_is_a_clean_refusal(agent):
    msg = await agent.call_expecting_error("drain_node", node_id="does-not-exist")
    assert "node not found" in msg and "HTTP 404" in msg


# -------------------------------------------------------------------- safety

def test_the_endpoint_refuses_to_mount_unauthenticated_in_production(monkeypatch):
    """Anyone who can reach /mcp can stop every model. Without a token, it must
    not come up outside dev mode."""
    from app import mcp_server

    monkeypatch.setattr(mcp_server.settings, "auth_mode", "oidc")
    monkeypatch.setattr(mcp_server.settings, "mcp_token", "")
    assert build_mcp_app() is None

    monkeypatch.setattr(mcp_server.settings, "mcp_token", "secret")
    assert build_mcp_app() is not None


def test_it_can_be_switched_off_entirely(monkeypatch):
    from app import mcp_server

    monkeypatch.setattr(mcp_server.settings, "mcp_enabled", False)
    assert build_mcp_app() is None


async def test_a_token_is_enforced_when_configured(monkeypatch):
    from app import mcp_server

    monkeypatch.setattr(mcp_server.settings, "mcp_token", "s3cret")
    guarded = build_mcp_app()
    started, stop = asyncio.Event(), asyncio.Event()

    async def runner():
        async with mcp.session_manager.run():
            started.set()
            await stop.wait()

    task = asyncio.create_task(runner())
    await started.wait()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=guarded),
                                     base_url="http://dgxctl.test") as http:
            body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

            assert (await http.post("/", headers=HDRS, json=body)).status_code == 401
            bad = await http.post("/", headers={**HDRS, "authorization": "Bearer wrong"}, json=body)
            assert bad.status_code == 401
            ok = await http.post("/", headers={**HDRS, "authorization": "Bearer s3cret"}, json=body)
            assert ok.status_code == 200
    finally:
        stop.set()
        await task


async def test_the_endpoint_is_mounted_on_the_main_app():
    """The fixture above drives a freshly built app; this proves the one the
    server actually serves is reachable at /mcp."""
    from starlette.routing import Mount

    mounts = [r for r in app.router.routes if isinstance(r, Mount) and r.path == "/mcp"]
    assert mounts, "MCP must be mounted before the SPA catch-all"
