"""Deployment lifecycle: plan, launch, observe, stop, restart."""
from __future__ import annotations

from app.drivers import get_driver
from app.models import Role
from app.services.deployments import drain_launches
from tests.conftest import deploy_that_fails, pump, register_fleet, set_role


# ------------------------------------------------------------------- planning

async def test_plan_reports_where_it_lands_and_the_exact_command(client):
    await register_fleet(["dgx-01"])
    plan = (await client.post("/api/deployments/plan", json={"spec_key": "qwen3-32b"})).json()

    assert plan["tensor_parallel_size"] == 2          # from the catalog entry
    assert plan["per_gpu_gb"] == 40.0
    assert plan["placements"][0]["node_name"] == "dgx-01"
    assert plan["placements"][0]["gpu_indices"] == [0, 1]

    argv = plan["argv"]
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3-32B"
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-32b"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-prefix-caching" in argv           # catalog extra_args


async def test_plan_explains_every_rejection(client):
    await register_fleet(["rtx-ws-01", "dgx-04"])
    plan = (await client.post("/api/deployments/plan", json={
        "hf_repo": "meta-llama/Llama-3.1-405B", "tensor_parallel_size": 8,
    })).json()
    assert plan["placements"] == []
    reasons = {r["node_name"]: r["reason"] for r in plan["rejections"]}
    assert reasons["rtx-ws-01"] == "has 2 GPUs, needs 8"
    assert reasons["dgx-04"] == "node is unreachable"


async def test_plan_sizes_an_uncatalogued_model_from_its_repo_name(client):
    await register_fleet(["dgx-01"])
    small = (await client.post("/api/deployments/plan", json={"hf_repo": "org/thing-8b"})).json()
    big = (await client.post("/api/deployments/plan", json={"hf_repo": "org/thing-70b"})).json()
    quant = (await client.post("/api/deployments/plan", json={"hf_repo": "org/thing-70b-fp8"})).json()
    assert small["per_gpu_gb"] < big["per_gpu_gb"]
    assert quant["per_gpu_gb"] < big["per_gpu_gb"], "a quantised build must be sized smaller"


async def test_plan_can_be_scoped_to_a_shortlist_of_nodes(client):
    ids = await register_fleet(["dgx-01", "dgx-02"])
    plan = (await client.post("/api/deployments/plan", json={
        "spec_key": "llama3.1-8b", "node_ids": [ids["dgx-02"]],
    })).json()
    assert [p["node_name"] for p in plan["placements"]] == ["dgx-02"]


async def test_plan_requires_a_model(client):
    assert (await client.post("/api/deployments/plan", json={})).status_code == 400
    assert (await client.post("/api/deployments/plan", json={"spec_key": "ghost"})).status_code == 404


# ------------------------------------------------------------------ launching

async def test_deploy_launches_a_container_with_the_planned_arguments(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).json()[0]

    # The claim is committed before the container exists: pulling an image can
    # take minutes and no HTTP client waits for it.
    assert dep["status"] == "pending"
    await drain_launches()
    dep = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert dep["status"] == "starting"
    assert dep["gpu_indices"] == [0, 1]
    assert dep["tensor_parallel_size"] == 2
    assert dep["port"] == 8100
    assert dep["container_name"].startswith("dgxctl-qwen3-32b-")
    assert dep["endpoint"] == "http://dgx-01.sim.local:8100"

    # and the fake node really is running it, labelled back to the deployment
    containers = await get_driver().list_containers(type("N", (), {"name": "dgx-01"})())
    assert containers[0].labels["dgxctl.deployment"] == dep["id"]


async def test_replicas_spread_across_nodes_not_onto_one(client):
    """One node failing must never take a whole model offline."""
    await register_fleet(["dgx-01", "dgx-02", "dgx-03"])
    deps = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 3})).json()
    assert len({d["node_name"] for d in deps}) == 3
    assert len({d["served_model_name"] for d in deps}) == 1


async def test_partial_placement_returns_what_it_could_place(client):
    await register_fleet(["rtx-ws-01"])
    deps = (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 4})).json()
    assert len(deps) == 1, "only one node can host it; the rest are reported as short"


async def test_no_capacity_is_a_409_with_the_reason(client):
    await register_fleet(["rtx-ws-01"])
    r = await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.1-405B", "tensor_parallel_size": 8, "replicas": 1,
    })
    assert r.status_code == 409
    assert "has 2 GPUs, needs 8" in r.json()["detail"]


async def test_explicit_gpu_targets_are_honoured(client):
    ids = await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={
        "spec_key": "qwen3-32b",
        "targets": [{"node_id": ids["dgx-01"], "gpu_indices": [4, 5]}],
    })).json()[0]
    assert dep["gpu_indices"] == [4, 5]


async def test_a_second_model_may_share_a_gpu_that_has_room(client):
    """22 GiB of an 80 GiB H100 leaves plenty; a whole card per small model is
    exactly the waste this is meant to avoid."""
    ids = await register_fleet(["dgx-01"])
    target = [{"node_id": ids["dgx-01"], "gpu_indices": [0]}]
    first = await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "targets": target})
    second = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "served_model_name": "second-8b", "targets": target})

    assert first.status_code == 201 and second.status_code == 201
    assert first.json()[0]["gpu_indices"] == second.json()[0]["gpu_indices"] == [0]
    assert first.json()[0]["port"] != second.json()[0]["port"]


async def test_each_tenant_is_told_to_take_only_its_share(client):
    """--gpu-memory-utilization is a fraction of the WHOLE card, so a fixed 0.9
    would leave nothing for anyone else however little the model needs."""
    ids = await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b",
        "targets": [{"node_id": ids["dgx-01"], "gpu_indices": [0]}]})).json()[0]

    argv = dep["vllm_args"]["argv"]
    share = float(argv[argv.index("--gpu-memory-utilization") + 1])
    assert 0.2 < share < 0.4, f"a 22 GiB model on an 80 GiB card asked for {share}"
    assert dep["reserved_mb_per_gpu"] == 22 * 1024


async def test_sharing_is_refused_once_the_vram_runs_out(client):
    ids = await register_fleet(["rtx-ws-01"])       # 2x 48 GiB
    target = [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0]}]
    assert (await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "targets": target})).status_code == 201
    r = await client.post("/api/deployments", json={
        "spec_key": "qwen3-32b", "served_model_name": "too-big-now",
        "tensor_parallel_size": 1, "targets": target})
    assert r.status_code == 409
    assert "cannot fit" in r.json()["detail"] and "GPU 0 has" in r.json()["detail"]


async def test_targeting_a_gpu_that_does_not_exist_is_refused(client):
    ids = await register_fleet(["rtx-ws-01"])
    r = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [7]}]})
    assert r.status_code == 400
    assert "has no GPU [7]" in r.json()["detail"]


async def test_ports_are_allocated_without_collision(client):
    await register_fleet(["dgx-01"])
    ports = set()
    for _ in range(3):
        dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
        ports.add(dep["port"])
    assert ports == {8100, 8101, 8102}


async def test_a_stopped_deployment_releases_its_port(client):
    await register_fleet(["dgx-01"])
    first = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await drain_launches()
    await client.post(f"/api/deployments/{first['id']}/stop")
    second = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    assert second["port"] == first["port"]


async def test_replicas_must_be_sane(client):
    await register_fleet(["dgx-01"])
    assert (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 0})).status_code == 400
    assert (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 99})).status_code == 400


# -------------------------------------------------------------- observability

async def test_a_deployment_becomes_healthy_and_reports_metrics(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump(2, gap=0.6)

    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "healthy"
    assert after["status_reason"] == "serving"
    assert after["healthy_since"] is not None

    m = after["last_metrics"]
    assert m["gen_tps"] > 0, "throughput must be derived once there are two samples"
    assert m["kv_cache_pct"] > 0
    assert m["ttft_avg_ms"] > 0


async def test_series_accumulates_samples_for_charting(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump(3)
    points = (await client.get(f"/api/deployments/{dep['id']}/series")).json()["points"]
    assert len(points) >= 2
    assert {"gen_tps", "running", "kv_cache_pct", "ttft_ms"} <= set(points[0]["values"])


async def test_logs_come_back_with_the_shard_progress(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await drain_launches()
    body = (await client.get(f"/api/deployments/{dep['id']}/logs")).json()
    assert "Loading safetensors checkpoint shards" in body["text"]
    assert "Uvicorn running" in body["text"]


async def test_a_healthy_deployment_is_not_nagged_about_loading_weights(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()
    body = (await client.get(f"/api/deployments/{dep['id']}/logs")).json()
    assert [f for f in body["findings"] if f["severity"] == "info"] == []


async def test_a_model_too_big_for_the_hardware_is_refused_not_launched(client):
    """The failure this exists to prevent. vLLM asked for more memory than the
    card has does not fail politely — it takes the node down with it, and a
    wedged DGX is a trip to the machine room."""
    ids = await register_fleet(["rtx-ws-01"])       # 2x 48 GiB
    r = await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "too-big",
        "tensor_parallel_size": 1,
        "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0]}],
    })
    assert r.status_code == 409
    assert (await client.get("/api/deployments")).json() == [], "nothing may be claimed"


async def test_a_catalog_entry_that_understates_a_model_cannot_get_through(client):
    """The catalog is hand-maintained, so its numbers are a floor rather than a
    fact: a 70B model with 10 written against it must not be placed on a card
    that cannot hold it."""
    await register_fleet(["rtx-ws-01"])
    await client.post("/api/catalog", json={
        "key": "mis-sized", "display_name": "Mis-sized 70B",
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct",
        "params_b": 70, "min_gpu_memory_gb": 10, "recommended_tp": 1,
    })
    r = await client.post("/api/deployments", json={"spec_key": "mis-sized", "replicas": 1})
    assert r.status_code == 409
    assert "needs at least" in r.json()["detail"]


async def test_a_refusal_says_what_would_work(client):
    """A dead end is not an answer. Every suggestion is checked against the
    hardware actually present before it is offered."""
    await register_fleet(["dgx-01"])
    r = await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "big",
        "tensor_parallel_size": 1, "replicas": 1,
    })
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Try:" in detail
    assert "tensor-parallel" in detail or "max-model-len" in detail or "build of this model" in detail


async def test_a_failed_deployment_gives_its_gpus_back(client):
    ids = await register_fleet(["rtx-ws-01"])
    await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct", "served_model_name": "too-big",
        "tensor_parallel_size": 2,
        "targets": [{"node_id": ids["rtx-ws-01"], "gpu_indices": [0, 1]}]})
    await pump()
    plan = (await client.post("/api/deployments/plan", json={"spec_key": "llama3.1-8b"})).json()
    assert plan["placements"], "GPUs held by a dead deployment must be reusable"


async def test_an_unreachable_node_degrades_rather_than_fails_its_models(client):
    """The model is probably still serving; only our view of it is broken."""
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await pump()
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["status"] == "healthy"

    get_driver()._nodes["dgx-01"].unreachable = True
    from app.db import SessionLocal
    from app import worker
    async with SessionLocal() as s:
        await worker.inventory_pass(s)
    await pump()

    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "degraded"
    assert "node unreachable" in after["status_reason"]
    get_driver()._nodes["dgx-01"].unreachable = False


# ------------------------------------------------------------------- lifecycle

async def test_stop_removes_the_container(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    r = await client.post(f"/api/deployments/{dep['id']}/stop")
    assert r.json()["status"] == "stopped"
    assert get_driver()._nodes["dgx-01"].containers == {}


async def test_restart_recreates_with_identical_arguments(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).json()[0]
    await drain_launches()
    new = (await client.post(f"/api/deployments/{dep['id']}/restart")).json()
    await drain_launches()

    assert new["id"] != dep["id"]
    assert new["vllm_args"]["argv"] == dep["vllm_args"]["argv"]
    assert new["gpu_indices"] == dep["gpu_indices"]
    assert (await client.get(f"/api/deployments/{new['id']}")).json()["status"] == "starting"
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["status"] == "stopped"


async def test_bulk_stop_and_restart(client):
    await register_fleet(["dgx-01", "dgx-02"])
    deps = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 2})).json()
    ids = [d["id"] for d in deps]

    restarted = (await client.post("/api/deployments/bulk/restart", json={"deployment_ids": ids})).json()
    assert len(restarted) == 2

    stopped = (await client.post("/api/deployments/bulk/stop",
                                 json={"deployment_ids": [d["id"] for d in restarted]})).json()
    assert {d["status"] for d in stopped} == {"stopped"}


async def test_listing_filters_to_active_by_default(client):
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await client.post(f"/api/deployments/{dep['id']}/stop")
    assert (await client.get("/api/deployments")).json() == []
    assert len((await client.get("/api/deployments?active_only=false")).json()) == 1


async def test_listing_can_be_scoped_to_a_node(client):
    ids = await register_fleet(["dgx-01", "dgx-02"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 2})
    only = (await client.get(f"/api/deployments?node_id={ids['dgx-01']}")).json()
    assert len(only) == 1 and only[0]["node_name"] == "dgx-01"


# ------------------------------------------------------------------ permissions

async def test_a_viewer_cannot_deploy(client):
    await register_fleet(["dgx-01"])
    await set_role(Role.viewer)
    r = await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    assert r.status_code == 403
    assert "requires deployer" in r.json()["detail"]


async def test_a_viewer_can_still_plan_and_read(client):
    await register_fleet(["dgx-01"])
    await set_role(Role.viewer)
    assert (await client.post("/api/deployments/plan", json={"spec_key": "llama3.1-8b"})).status_code == 200
    assert (await client.get("/api/deployments")).status_code == 200


async def test_unknown_deployment_is_a_404(client):
    assert (await client.get("/api/deployments/nope")).status_code == 404
    assert (await client.post("/api/deployments/nope/stop")).status_code == 404
    assert (await client.post("/api/deployments/nope/restart")).status_code == 404


# ------------------------------------------------------------ image handling

async def test_the_image_is_pulled_before_the_container_starts(client):
    """A node that has never run this image has several gigabytes to fetch
    first, and the operator should see that rather than an unexplained wait."""
    from app.drivers import get_driver

    await register_fleet(["dgx-01"])
    sim = get_driver()._nodes["dgx-01"]
    assert sim.images == set(), "a fresh node has no images"

    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})).json()[0]
    await drain_launches()

    assert "vllm/vllm-openai:latest" in sim.images
    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "starting"


async def test_pulling_is_announced_while_it_happens(client):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})
    await drain_launches()

    messages = [e["message"] for e in (await client.get("/api/events")).json()]
    assert any("pulling vllm/vllm-openai:latest onto dgx-01" in m for m in messages)


async def test_an_image_already_on_the_node_is_not_pulled_again(client):
    await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})
    await drain_launches()

    before = len((await client.get("/api/events")).json())
    await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "served_model_name": "second"})
    await drain_launches()

    pulls = [e for e in (await client.get("/api/events")).json() if "pulling" in e["message"]]
    assert len(pulls) == 1, "the second deployment reuses the image"
    assert before  # the first deploy did log


async def test_an_image_that_cannot_be_pulled_fails_with_the_registry_message(client):
    """A typo in an image tag should read as a typo, not as a mystery."""
    await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "image": "vllm/vllm-openai:missing"})).json()[0]
    await drain_launches()

    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "failed"
    assert "manifest" in after["status_reason"]

    plan = (await client.post("/api/deployments/plan", json={"spec_key": "llama3.1-8b"})).json()
    assert plan["placements"], "a failed pull must release the GPUs it claimed"
