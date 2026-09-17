"""Model catalog: the sizing decisions that make deploying one click."""
from __future__ import annotations

from app.models import Role
from app.services.deployments import drain_launches
from tests.conftest import register_fleet, set_role


async def test_seeded_entries_carry_sizing_and_defaults(client):
    entries = {e["key"]: e for e in (await client.get("/api/catalog")).json()}
    assert len(entries) >= 8

    qwen = entries["qwen3-32b"]
    assert qwen["hf_repo"] == "Qwen/Qwen3-32B"
    assert qwen["recommended_tp"] == 2
    assert qwen["min_gpu_memory_gb"] == 40
    assert qwen["extra_args"] == {"--enable-prefix-caching": True}

    # every entry must be sized, or placement is guesswork
    for e in entries.values():
        assert e["min_gpu_memory_gb"] > 0, f"{e['key']} has no size"
        assert e["recommended_tp"] >= 1


async def test_hardware_caveats_are_recorded_where_they_matter(client):
    entries = {e["key"]: e for e in (await client.get("/api/catalog")).json()}
    assert "Hopper" in entries["llama3.1-70b-fp8"]["notes"]
    assert "HF_TOKEN" in entries["llama3.1-8b"]["notes"]


async def test_create_update_delete(client):
    body = {"key": "mine", "display_name": "My Model", "hf_repo": "me/model",
            "params_b": 13, "min_gpu_memory_gb": 30, "recommended_tp": 1,
            "tags": ["internal"]}
    created = (await client.post("/api/catalog", json=body)).json()
    assert created["key"] == "mine"

    updated = (await client.put(f"/api/catalog/{created['id']}",
                                json={**body, "min_gpu_memory_gb": 44})).json()
    assert updated["min_gpu_memory_gb"] == 44

    assert (await client.delete(f"/api/catalog/{created['id']}")).status_code == 204
    assert "mine" not in {e["key"] for e in (await client.get("/api/catalog")).json()}


async def test_duplicate_key_is_rejected(client):
    body = {"key": "qwen3-32b", "display_name": "Dupe", "hf_repo": "x/y"}
    r = await client.post("/api/catalog", json=body)
    assert r.status_code == 409


async def test_a_custom_entry_drives_placement(client):
    """The catalog is not decoration: its numbers decide where a model fits."""
    await register_fleet(["rtx-ws-01"])
    await client.post("/api/catalog", json={
        "key": "huge", "display_name": "Huge", "hf_repo": "me/huge",
        "min_gpu_memory_gb": 200, "recommended_tp": 1})
    plan = (await client.post("/api/deployments/plan", json={"spec_key": "huge"})).json()
    assert plan["per_gpu_gb"] == 200
    assert plan["placements"] == []
    assert "needs 200 GiB free per GPU" in plan["rejections"][0]["reason"]


async def test_catalog_extra_args_reach_the_container(client):
    await register_fleet(["dgx-01"])
    await client.post("/api/catalog", json={
        "key": "flagged", "display_name": "Flagged", "hf_repo": "me/flagged",
        "min_gpu_memory_gb": 10, "recommended_tp": 1,
        "extra_args": {"--max-num-seqs": 64, "--enable-chunked-prefill": True}})
    dep = (await client.post("/api/deployments", json={"spec_key": "flagged", "replicas": 1})).json()[0]
    argv = dep["vllm_args"]["argv"]
    assert argv[argv.index("--max-num-seqs") + 1] == "64"
    assert "--enable-chunked-prefill" in argv


async def test_a_request_override_beats_the_catalog(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={
        "spec_key": "qwen3-32b", "replicas": 1,
        "tensor_parallel_size": 4, "max_model_len": 8192,
        "gpu_memory_utilization": 0.8})).json()[0]
    argv = dep["vllm_args"]["argv"]
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.8"
    assert len(dep["gpu_indices"]) == 4


async def test_seeding_is_idempotent(client):
    before = len((await client.get("/api/catalog")).json())
    assert (await client.post("/api/catalog/seed")).json() == {"added": 0}
    assert len((await client.get("/api/catalog")).json()) == before


async def test_catalog_edits_need_deployer(client):
    await set_role(Role.viewer)
    r = await client.post("/api/catalog", json={"key": "x", "display_name": "X", "hf_repo": "a/b"})
    assert r.status_code == 403
    assert (await client.get("/api/catalog")).status_code == 200


async def test_deleting_an_entry_that_is_still_deployed_says_where(client):
    """Deployments carry a foreign key back to the entry they came from. Left
    alone that turns a delete into a 500; what an operator needs is the name of
    the box still running it."""
    from tests.conftest import register_fleet

    await register_fleet(["dgx-01"])
    spec = (await client.post("/api/catalog", json={
        "key": "still-running", "display_name": "Still Running", "hf_repo": "org/m",
        "params_b": 1, "min_gpu_memory_gb": 4, "recommended_tp": 1,
    })).json()
    assert (await client.post("/api/deployments", json={
        "spec_key": "still-running", "replicas": 1})).status_code == 201

    r = await client.delete(f"/api/catalog/{spec['id']}")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "dgx-01" in detail and "Stop those deployments first" in detail
    assert len((await client.get("/api/catalog")).json()) > 0


async def test_a_finished_deployment_does_not_block_the_delete(client, db):
    """History keeps its own repo and model name, so the entry can go."""
    from sqlalchemy import select

    from app.models import Deployment, DeployStatus
    from app.services.deployments import drain_launches
    from tests.conftest import register_fleet

    await register_fleet(["dgx-01"])
    spec = (await client.post("/api/catalog", json={
        "key": "finished", "display_name": "Finished", "hf_repo": "org/m",
        "params_b": 1, "min_gpu_memory_gb": 4, "recommended_tp": 1,
    })).json()
    dep = (await client.post("/api/deployments", json={
        "spec_key": "finished", "replicas": 1})).json()[0]

    # Let the background launch finish first, or it writes the status back
    # underneath us and the entry still looks live.
    await drain_launches()
    row = await db.get(Deployment, dep["id"])
    await db.refresh(row)
    row.status = DeployStatus.stopped
    await db.commit()

    assert (await client.delete(f"/api/catalog/{spec['id']}")).status_code == 204

    # The deployment survives, detached rather than deleted.
    kept = (await db.execute(select(Deployment).where(Deployment.id == dep["id"]))).scalar_one()
    await db.refresh(kept)
    assert kept.spec_id is None
    assert kept.hf_repo == "org/m"
