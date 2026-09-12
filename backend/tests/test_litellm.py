"""LiteLLM integration, against a stand-in proxy that behaves like the real one.

The contract under test: a healthy deployment appears in the proxy by itself,
replicas of one served name collapse into a single load-balanced group, stopping
removes the entry, and resync repairs drift in both directions.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.models import Role
from app.services import litellm as litellm_svc
from tests.conftest import pump, register_fleet, set_role


def make_fake_proxy() -> tuple[FastAPI, dict]:
    """Minimal LiteLLM: a model registry keyed by model_info.id."""
    state: dict = {"models": {}, "down": False, "no_db": False, "completions": 0}
    proxy = FastAPI()

    @proxy.get("/health/readiness")
    async def readiness():
        return {"status": "healthy", "db": "connected"}

    @proxy.get("/model/info")
    async def info():
        return {"data": list(state["models"].values())}

    @proxy.post("/model/new")
    async def new(body: dict):
        if state["no_db"]:
            # What a real LiteLLM without STORE_MODEL_IN_DB actually answers.
            raise HTTPException(500, "No DB Connected. Set DATABASE_URL")
        state["models"][body["model_info"]["id"]] = body
        return {"message": "ok"}

    @proxy.post("/model/delete")
    async def delete(body: dict):
        state["models"].pop(body["id"], None)
        return {"message": "ok"}

    @proxy.post("/v1/chat/completions")
    async def chat(body: dict):
        state["completions"] += 1
        names = {m["model_name"] for m in state["models"].values()}
        if body["model"] not in names:
            return httpx.Response, {"error": "no such model"}
        return {"choices": [{"message": {"content": "pong"}}], "usage": {"total_tokens": 3}}

    return proxy, state


@pytest.fixture
def proxy(monkeypatch):
    """Point every LiteLLMClient construction at the fake, and turn
    auto-registration back on (the default test env disables it)."""
    app_, state = make_fake_proxy()

    class Flaky(httpx.AsyncBaseTransport):
        """Unreachable has to look like a connection failure, not an exception
        raised inside the app — that is the shape the client must survive."""

        def __init__(self):
            self._inner = httpx.ASGITransport(app=app_)

        async def handle_async_request(self, request):
            if state["down"]:
                raise httpx.ConnectError("All connection attempts failed", request=request)
            return await self._inner.handle_async_request(request)

    class FakeClient(litellm_svc.LiteLLMClient):
        def __init__(self, base_url=None, master_key=None):
            self.base_url = "http://litellm.test"
            self.key = "sk-test"
            self._client = httpx.AsyncClient(
                transport=Flaky(),
                base_url="http://litellm.test",
                headers={"Authorization": "Bearer sk-test"},
                timeout=5.0,
            )

    for module in ("app.services.litellm", "app.api.litellm_api", "app.api.fleet"):
        monkeypatch.setattr(f"{module}.LiteLLMClient", FakeClient, raising=False)
    monkeypatch.setattr(litellm_svc.settings, "litellm_auto_register", True)
    return state


async def test_a_healthy_deployment_registers_itself(client, proxy):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).json()[0]
    await pump()

    entry = proxy["models"][dep["id"]]
    assert entry["model_name"] == "qwen3-32b"
    # hosted_vllm/ is what tells LiteLLM to speak OpenAI to a vLLM server
    assert entry["litellm_params"]["model"] == "hosted_vllm/qwen3-32b"
    assert entry["litellm_params"]["api_base"] == "http://dgx-01.sim.local:8100/v1"
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["litellm_registered"] is True


async def test_a_deployment_that_never_goes_healthy_is_not_registered(client, proxy):
    ids = await register_fleet(["rtx-ws-01"])
    await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "too-big",
        "tensor_parallel_size": 2,
        "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0, 1]}]})
    await pump()
    assert proxy["models"] == {}


async def test_replicas_become_one_load_balanced_group(client, proxy):
    """This is how a model scales horizontally: deploy it again, nothing else."""
    await register_fleet(["dgx-01", "dgx-02"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 2})
    await pump()

    groups = (await client.get("/api/litellm/status")).json()["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["model_name"] == "llama3.1-8b"
    assert len(group["members"]) == 2
    assert all(m["managed_by_dgxctl"] for m in group["members"])
    assert len({m["api_base"] for m in group["members"]}) == 2


async def test_stopping_removes_it_from_the_proxy_first(client, proxy):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()
    assert dep["id"] in proxy["models"]

    await client.post(f"/api/deployments/{dep['id']}/stop")
    assert proxy["models"] == {}


async def test_status_reports_reachability(client, proxy):
    s = (await client.get("/api/litellm/status")).json()
    assert s["reachable"] is True
    assert s["auto_register"] is True


async def test_resync_is_a_no_op_when_everything_matches(client, proxy):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    await pump()
    r = (await client.post("/api/litellm/resync")).json()
    assert r == {"added": [], "removed": [], "errors": []}


async def test_resync_re_adds_an_entry_deleted_behind_our_back(client, proxy):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()
    proxy["models"].clear()

    r = (await client.post("/api/litellm/resync")).json()
    assert r["added"] == ["llama3.1-8b@dgx-01"]
    assert dep["id"] in proxy["models"]


async def test_resync_removes_an_entry_whose_deployment_is_gone(client, proxy):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    await pump()
    proxy["models"]["ghost"] = {
        "model_name": "ghost", "litellm_params": {"api_base": "http://gone/v1"},
        "model_info": {"id": "ghost", "dgxctl_deployment_id": "ghost"},
    }
    r = (await client.post("/api/litellm/resync")).json()
    assert r["removed"] == ["ghost"]
    assert "ghost" not in proxy["models"]


async def test_resync_leaves_models_it_does_not_manage_alone(client, proxy):
    """Hand-added entries for hosted APIs must survive a resync."""
    proxy["models"]["external"] = {
        "model_name": "gpt-4o", "litellm_params": {"api_base": "https://api.openai.com/v1"},
        "model_info": {"id": "external"},
    }
    r = (await client.post("/api/litellm/resync")).json()
    assert r["removed"] == []
    assert "external" in proxy["models"]


async def test_a_test_request_round_trips_through_the_proxy(client, proxy):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    await pump()
    r = (await client.post("/api/litellm/test", json={"model_name": "llama3.1-8b"})).json()
    assert r["ok"] is True
    assert r["reply"] == "pong"
    assert proxy["completions"] == 1


async def test_the_proxy_being_down_never_blocks_cluster_operations(client, proxy):
    """Models must keep deploying and serving when LiteLLM is unavailable; only
    routing is affected."""
    proxy["down"] = True
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()

    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "healthy"
    assert after["litellm_registered"] is False

    warnings = [e for e in (await client.get("/api/events")).json()
                if e["source"] == "litellm" and e["severity"] == "warning"]
    assert warnings, "the operator must be told routing is not set up"


async def test_registration_recovers_once_the_proxy_returns(client, proxy):
    proxy["down"] = True
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()
    assert proxy["models"] == {}

    proxy["down"] = False
    r = (await client.post("/api/litellm/resync")).json()
    assert r["added"] == ["llama3.1-8b@dgx-01"]
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["litellm_registered"] is True


async def test_resync_needs_deployer(client, proxy):
    await set_role(Role.viewer)
    assert (await client.post("/api/litellm/resync")).status_code == 403
    assert (await client.get("/api/litellm/status")).status_code == 200


async def test_a_proxy_error_response_is_reported_not_swallowed(client, proxy):
    """A LiteLLM running without its database answers 500 to /model/new. That is
    the real failure mode behind a model showing as 'not routed'."""
    proxy["no_db"] = True
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()

    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["status"] == "healthy"
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["litellm_registered"] is False

    warning = next(e for e in (await client.get("/api/events")).json() if e["source"] == "litellm")
    assert "No DB Connected" in warning["message"]
