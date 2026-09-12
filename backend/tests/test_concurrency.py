"""Two callers must never be handed the same GPU or the same port.

The MCP endpoint exists so an agent can drive the fleet while a human is at the
dashboard, which makes simultaneous deploys an ordinary event rather than a
corner case. These tests use real concurrency — asyncio.gather over the HTTP
surface — because a sequential test cannot fail the way the bug does.
"""
from __future__ import annotations

import asyncio
from collections import Counter

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


async def test_simultaneous_deploys_never_hand_out_the_same_gpu(client):
    """rtx-ws-01 has two GPUs and the model needs one each, so at most two of
    the six callers can win. What matters is that no GPU appears twice."""
    ids = await register_fleet(["rtx-ws-01"])

    codes = await _deploy_all(client, 6, spec_key="llama3.1-8b", replicas=1,
                              node_ids=[ids["rtx-ws-01"]])
    assert 201 in codes, f"at least one caller must succeed: {codes}"

    live = (await client.get("/api/deployments")).json()
    claimed = [tuple(d["gpu_indices"]) for d in live]
    duplicated = [gpus for gpus, n in Counter(claimed).items() if n > 1]
    assert not duplicated, f"GPUs handed out more than once: {duplicated}"
    assert len(live) <= 2, f"node has 2 GPUs but {len(live)} deployments claim it"


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
    """Someone who asked for GPU 0 specifically must not be silently placed
    somewhere else; they need to know their choice was taken."""
    ids = await register_fleet(["dgx-01"])
    target = [{"node_id": ids["dgx-01"], "gpu_indices": [0]}]

    results = await asyncio.gather(
        _deploy(client, spec_key="llama3.1-8b", targets=target),
        _deploy(client, spec_key="llama3.1-8b", targets=target),
    )
    codes = sorted(r.status_code for r in results)
    assert codes == [201, 409], f"expected one win and one refusal, got {codes}"

    loser = next(r for r in results if r.status_code == 409)
    assert "already in use" in loser.json()["detail"]


async def test_an_automatically_placed_deploy_is_re_placed_after_losing_a_race(client):
    """Asking for capacity anywhere should not fail because one candidate was
    taken mid-flight — there is other hardware."""
    await register_fleet(["dgx-01", "dgx-02", "dgx-03"])

    codes = await _deploy_all(client, 6, spec_key="llama3.1-8b", replicas=1)
    assert codes.count(201) == 6, f"the fleet had room for all six: {codes}"

    live = (await client.get("/api/deployments")).json()
    claimed = [(d["node_name"], tuple(d["gpu_indices"])) for d in live]
    assert len(claimed) == len(set(claimed)), f"double-booked: {claimed}"


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
        assert r.json()[0]["status"] == "failed"
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
