"""Two callers must never be handed the same GPU or the same port.

The MCP endpoint exists so an agent can drive the fleet while a human is at the
dashboard, which makes simultaneous deploys an ordinary event rather than a
corner case. These tests use real concurrency — asyncio.gather over the HTTP
surface — because a sequential test cannot fail the way the bug does.
"""
from __future__ import annotations

import asyncio
from collections import Counter

from app.services.deployments import drain_launches
from tests.conftest import register_fleet


async def _deploy(client, **body):
    return await client.post("/api/deployments", json=body)


async def _deploy_all(client, n: int, **body) -> list[int]:
    """Fire n deploys at once and return their status codes.

    Exceptions are re-raised rather than collected: a request that blew up
    instead of answering is the failure mode these tests exist to catch, and
    swallowing it would let the invariant assertions pass on an empty fleet.
    """
    results = await asyncio.gather(
        *[_deploy(client, **body) for _ in range(n)], return_exceptions=True
    )
    blew_up = [r for r in results if isinstance(r, BaseException)]
    if blew_up:
        raise AssertionError(
            f"{len(blew_up)} of {n} concurrent deploys raised instead of answering: "
            f"{type(blew_up[0]).__name__}: {blew_up[0]}"
        ) from blew_up[0]
    return [r.status_code for r in results]


async def test_simultaneous_deploys_never_oversubscribe_a_gpu(client):
    """Sharing is allowed, overcommitting is not: the reservations on any one
    card must never exceed what the card has."""
    ids = await register_fleet(["rtx-ws-01"])       # 2x 48 GiB, model wants 22

    codes = await _deploy_all(client, 8, spec_key="llama3.1-8b", replicas=1,
                              node_ids=[ids["rtx-ws-01"]])
    assert 201 in codes, f"at least one caller must succeed: {codes}"

    node = (await client.get(f"/api/nodes/{ids['rtx-ws-01']}")).json()
    live = (await client.get("/api/deployments")).json()

    per_gpu: dict[int, int] = {}
    for d in live:
        for i in d["gpu_indices"]:
            per_gpu[i] = per_gpu.get(i, 0) + d["reserved_mb_per_gpu"]
    for gpu in node["gpus"]:
        assert per_gpu.get(gpu["index"], 0) <= gpu["memory_total_mb"], (
            f"GPU {gpu['index']} oversubscribed: "
            f"{per_gpu[gpu['index']]} MB reserved of {gpu['memory_total_mb']} MB"
        )
    assert sum(per_gpu.values()) > 0


async def test_simultaneous_deploys_never_hand_out_the_same_port(client):
    """Two containers on one port means the second `docker run` fails on real
    hardware, after the model has been accepted."""
    await register_fleet(["dgx-01"])

    await _deploy_all(client, 8, spec_key="llama3.1-8b", replicas=1)

    live = (await client.get("/api/deployments")).json()
    by_node: dict[str, list[int]] = {}
    for d in live:
        by_node.setdefault(d["node_name"], []).append(d["port"])
    for node, ports in by_node.items():
        assert len(ports) == len(set(ports)), f"{node} reused a port: {sorted(ports)}"


async def test_a_caller_that_named_exact_gpus_is_refused_rather_than_moved(client):
    """Two 40 GiB models cannot both fit a 48 GiB card. Whoever loses asked for
    GPU 0 specifically and must be told, not silently placed elsewhere."""
    ids = await register_fleet(["dgx-01"])
    target = [{"node_id": ids["dgx-01"], "gpu_indices": [0]}]

    results = await asyncio.gather(
        _deploy(client, spec_key="qwen3-32b", tensor_parallel_size=1, targets=target),
        _deploy(client, spec_key="qwen3-32b", served_model_name="rival",
                tensor_parallel_size=1, targets=target),
    )
    codes = sorted(r.status_code for r in results)
    assert codes == [201, 409], f"expected one win and one refusal, got {codes}"

    loser = next(r for r in results if r.status_code == 409)
    assert "cannot fit" in loser.json()["detail"]


async def test_an_automatically_placed_deploy_is_re_placed_after_losing_a_race(client):
    """Asking for capacity anywhere should not fail because one candidate was
    taken mid-flight — there is other hardware."""
    await register_fleet(["dgx-01", "dgx-02", "dgx-03"])

    codes = await _deploy_all(client, 6, spec_key="llama3.1-8b", replicas=1)
    assert codes.count(201) == 6, f"the fleet had room for all six: {codes}"

    # Several may legitimately land on one card; what must hold is that no card
    # promised more than it has.
    live = (await client.get("/api/deployments")).json()
    nodes = {n["id"]: n for n in (await client.get("/api/nodes")).json()}
    per_gpu: dict[tuple[str, int], int] = {}
    for d in live:
        for i in d["gpu_indices"]:
            per_gpu[(d["node_id"], i)] = per_gpu.get((d["node_id"], i), 0) + d["reserved_mb_per_gpu"]
    for (node_id, index), reserved in per_gpu.items():
        total = next(g["memory_total_mb"] for g in nodes[node_id]["gpus"] if g["index"] == index)
        assert reserved <= total, f"{nodes[node_id]['name']} GPU {index} oversubscribed"


async def test_a_claim_is_visible_to_other_callers_before_the_container_starts(client):
    """The window between choosing hardware and the container existing is the
    whole bug: the claim has to be committed, not held in an open transaction."""
    ids = await register_fleet(["rtx-ws-01"])
    await _deploy(client, spec_key="llama3.1-8b", replicas=1, node_ids=[ids["rtx-ws-01"]])

    node = (await client.get(f"/api/nodes/{ids['rtx-ws-01']}")).json()
    busy = [g for g in node["gpus"] if g["deployment_id"]]
    assert len(busy) == 1, "the claim must be visible as soon as it is made"


async def test_a_container_that_fails_to_start_gives_its_gpus_back(client):
    """A reservation outlives the attempt only if the attempt succeeded."""
    from app.drivers import get_driver

    ids = await register_fleet(["rtx-ws-01"])
    get_driver()._nodes["rtx-ws-01"].unreachable = True
    try:
        r = await _deploy(client, spec_key="llama3.1-8b", replicas=1,
                          node_ids=[ids["rtx-ws-01"]])
        assert r.status_code == 201
        await drain_launches()
        assert (await client.get(f"/api/deployments/{r.json()[0]['id']}")
                ).json()["status"] == "failed"
    finally:
        get_driver()._nodes["rtx-ws-01"].unreachable = False

    plan = (await client.post("/api/deployments/plan",
                              json={"spec_key": "llama3.1-8b"})).json()
    assert plan["placements"], "GPUs held by a claim that never started must be reusable"


async def test_a_stop_that_fails_keeps_its_gpus_reserved(client):
    """If the container might still be running it is still using the hardware,
    and the next deploy must not land on top of it."""
    from app.drivers import get_driver

    ids = await register_fleet(["rtx-ws-01"])
    dep = (await _deploy(client, spec_key="qwen3-32b", replicas=1,
                         node_ids=[ids["rtx-ws-01"]])).json()[0]

    get_driver()._nodes["rtx-ws-01"].unreachable = True
    try:
        after = (await client.post(f"/api/deployments/{dep['id']}/stop")).json()
    finally:
        get_driver()._nodes["rtx-ws-01"].unreachable = False

    assert after["status"] != "stopped"
    assert "may still be running" in after["status_reason"]

    node = (await client.get(f"/api/nodes/{ids['rtx-ws-01']}")).json()
    assert [g for g in node["gpus"] if g["deployment_id"]], "GPUs must stay reserved"
