"""Fleet summary, the activity feed, and the audit trail."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import fleet as fleet_module
from app.main import app
from app.services.deployments import drain_launches
from tests.conftest import deploy_undersized, pump, register_fleet


@pytest.fixture(autouse=True)
def _reset_litellm_cache():
    """Every test starts and ends with a cold cache, so a reachability answer
    from one test can never leak into the next."""
    fleet_module._litellm_cache["checked_at"] = 0.0
    fleet_module._litellm_cache["reachable"] = False
    yield
    fleet_module._litellm_cache["checked_at"] = 0.0
    fleet_module._litellm_cache["reachable"] = False


async def test_summary_adds_up_across_the_fleet(client):
    await register_fleet()
    s = (await client.get("/api/summary")).json()

    assert s["nodes_total"] == 7
    assert s["nodes_online"] == 6
    assert s["nodes_unreachable"] == 1          # dgx-04 is down in the simulator
    assert s["gpus_total"] == 32                # 3x8 reachable DGX + 2+2+4 RTX
    assert s["gpus_busy"] == 0
    assert s["gpus_free"] == 32
    assert s["vram_total_gb"] > 2000
    assert s["litellm_reachable"] is False      # no proxy in the test environment


async def test_summary_tracks_deployments_and_throughput(client):
    await register_fleet()
    await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 2})
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    await pump(2, gap=0.6)

    s = (await client.get("/api/summary")).json()
    assert s["deployments_healthy"] == 3
    assert s["models_served"] == 2               # distinct served names, not containers
    assert s["gpus_busy"] == 5                   # 2 + 2 + 1
    assert s["tokens_per_second"] > 0
    assert s["requests_running"] >= 0


async def test_summary_counts_failures_separately(client):
    ids = await register_fleet(["rtx-ws-01"])
    await deploy_undersized(client, ids["rtx-ws-01"], [0, 1])
    await pump()
    s = (await client.get("/api/summary")).json()
    assert s["deployments_failed"] == 1
    assert s["deployments_healthy"] == 0
    assert s["gpus_busy"] == 0, "a dead deployment does not hold GPUs"


async def test_events_record_what_the_fleet_did(client):
    await register_fleet(["dgx-01", "dgx-04"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    await pump()

    events = (await client.get("/api/events")).json()
    messages = [e["message"] for e in events]
    assert any("dgx-04 became unreachable" in m for m in messages)
    assert any("llama3.1-8b starting on dgx-01" in m for m in messages)
    assert any("starting -> healthy" in m for m in messages)

    # newest first, so the feed reads as a timeline
    assert events == sorted(events, key=lambda e: e["ts"], reverse=True)


async def test_events_can_be_filtered_to_problems(client):
    await register_fleet(["dgx-04"])
    errors = (await client.get("/api/events", params={"severity": "error"})).json()
    assert errors and all(e["severity"] == "error" for e in errors)


async def test_audit_records_who_did_what(client):
    ids = await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await client.post(f"/api/deployments/{dep['id']}/stop")
    await client.post(f"/api/nodes/{ids['dgx-01']}/drain")

    entries = (await client.get("/api/audit")).json()
    by_action = {e["action"]: e for e in entries}
    assert {"deployment.create", "deployment.stop", "node.drain"} <= set(by_action)
    assert all(e["actor"] == "dev@localhost" for e in entries)
    assert by_action["deployment.create"]["target_id"] == dep["id"]
    assert by_action["deployment.create"]["ok"] is True


async def test_a_failed_action_is_audited_as_failed(client):
    """A launch that the node refuses must be recorded, not silently dropped."""
    ids = await register_fleet(["dgx-01"])
    from app.drivers import get_driver
    get_driver()._nodes["dgx-01"].unreachable = True
    await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "targets": [{"node_id": ids["dgx-01"], "gpu_indices": [0]}]})
    await drain_launches()
    get_driver()._nodes["dgx-01"].unreachable = False

    failed = [e for e in (await client.get("/api/audit")).json() if not e["ok"]]
    assert failed and failed[0]["action"] == "deployment.create"


async def test_runtime_config_tells_the_ui_it_is_simulated(client):
    cfg = (await client.get("/api/config")).json()
    assert cfg["driver"] == "sim"
    assert cfg["simulated"] is True
    assert cfg["auth_mode"] == "dev"


async def test_healthz_needs_no_auth(client):
    r = await client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


async def test_the_summary_ignores_failures_that_are_no_longer_actionable(client):
    """Node reboots accumulate dead deployments. Counting them forever turns
    the headline number into noise that hides a real, current failure."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Deployment

    ids = await register_fleet(["rtx-ws-01"])
    await deploy_undersized(client, ids["rtx-ws-01"], [0, 1])
    await pump()
    assert (await client.get("/api/summary")).json()["deployments_failed"] == 1

    # age it past the window
    async with SessionLocal() as s:
        row = await s.execute(select(Deployment))
        dep = row.scalars().first()
        dep.created_at = datetime.now(timezone.utc) - timedelta(hours=3)
        await s.commit()

    assert (await client.get("/api/summary")).json()["deployments_failed"] == 0
    # ...but the record is still there for anyone who asks
    all_deps = (await client.get("/api/deployments?active_only=false")).json()
    assert [d["status"] for d in all_deps] == ["failed"]


async def test_summary_does_not_re_ask_litellm_within_the_cache_window(client, monkeypatch):
    """A slow or hung proxy must not stall the dashboard poll every 6 seconds."""
    calls = {"n": 0}

    class CountingClient:
        async def health(self):
            calls["n"] += 1
            return {"reachable": True, "detail": {}}

        async def close(self):
            pass

    monkeypatch.setattr(fleet_module, "LiteLLMClient", CountingClient)
    await register_fleet(["dgx-01"])

    first = (await client.get("/api/summary")).json()
    second = (await client.get("/api/summary")).json()

    assert calls["n"] == 1
    assert first["litellm_reachable"] is True
    assert second["litellm_reachable"] is True


async def test_the_websocket_rejects_an_unauthenticated_connection(monkeypatch):
    """Node names, model names, GPU metrics and fleet events must not be
    handed to anyone who can merely reach the port."""
    monkeypatch.setattr(fleet_module.settings, "auth_mode", "oidc")

    # No `with TestClient(app) as tc:` here: entering it runs the app's real
    # startup (migrations, seeding, worker loops), which this check has no use
    # for and which fights with the schema `fresh_db` already built for this
    # test. A plain instance still opens a websocket for one request.
    tc = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect("/api/ws"):
            pass


async def test_the_websocket_still_works_with_no_login_in_dev_mode():
    """Dev mode signs everyone in as admin everywhere else in the app; the
    live feed must not be the one place that breaks that promise."""
    tc = TestClient(app)
    with tc.websocket_connect("/api/ws") as ws:
        hello = ws.receive_json()
        assert hello == {"topic": "hello", "data": {"driver": "sim"}}


async def test_retention_trims_stale_events_while_the_audit_log_outlives_them(monkeypatch):
    """The audit log is the record of who did what, so it is kept long after
    the operational noise it happened alongside has been trimmed away."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from app import worker
    from app.db import SessionLocal
    from app.models import AuditLog, Event

    monkeypatch.setattr(worker.settings, "event_retention_days", 1)
    monkeypatch.setattr(worker.settings, "audit_retention_days", 30)

    stale = datetime.now(timezone.utc) - timedelta(days=5)
    fresh = datetime.now(timezone.utc)

    async with SessionLocal() as db:
        db.add(Event(ts=stale, severity="info", source="node", message="stale"))
        db.add(Event(ts=fresh, severity="info", source="node", message="fresh"))
        db.add(AuditLog(ts=stale, actor="dev@localhost", action="node.drain", summary="stale"))
        db.add(AuditLog(ts=fresh, actor="dev@localhost", action="node.drain", summary="fresh"))
        await db.commit()

    async with SessionLocal() as db:
        await worker.retention_pass(db)

    async with SessionLocal() as db:
        events_left = {e.message for e in (await db.execute(select(Event))).scalars()}
        audit_left = {a.summary for a in (await db.execute(select(AuditLog))).scalars()}

    assert events_left == {"fresh"}, "an event older than the retention window must not linger"
    assert audit_left == {"stale", "fresh"}, "the audit trail outlives the event feed on purpose"
