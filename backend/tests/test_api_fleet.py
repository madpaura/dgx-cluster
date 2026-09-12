"""Fleet summary, the activity feed, and the audit trail."""
from __future__ import annotations

from tests.conftest import pump, register_fleet


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
    await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "too-big",
        "tensor_parallel_size": 2,
        "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0, 1]}]})
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
    await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "too-big",
        "tensor_parallel_size": 2,
        "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0, 1]}]})
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
