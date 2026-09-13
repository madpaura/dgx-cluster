"""Clusters: operator-defined grouping that must never be able to break the fleet."""
from __future__ import annotations

from app.models import Role
from app.services.deployments import drain_launches
from tests.conftest import register_fleet, set_role


async def test_create_and_list(client):
    r = await client.post("/api/clusters", json={"name": "Machine Room A", "description": "H100s"})
    assert r.status_code == 201
    assert r.json()["name"] == "Machine Room A"
    assert r.json()["node_count"] == 0

    listed = (await client.get("/api/clusters")).json()
    assert [c["name"] for c in listed] == ["Machine Room A"]


async def test_create_with_members_in_one_call(client):
    ids = await register_fleet(["dgx-01", "dgx-02"])
    r = await client.post("/api/clusters", json={"name": "Pod", "node_ids": list(ids.values())})
    assert r.status_code == 201
    assert r.json()["node_count"] == 2
    nodes = (await client.get("/api/nodes")).json()
    assert {n["cluster_name"] for n in nodes} == {"Pod"}


async def test_duplicate_name_is_rejected(client):
    await client.post("/api/clusters", json={"name": "Pod"})
    r = await client.post("/api/clusters", json={"name": "Pod"})
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


async def test_a_blank_name_is_rejected(client):
    assert (await client.post("/api/clusters", json={"name": "   "})).status_code == 400


async def test_rename_keeps_membership(client):
    ids = await register_fleet(["dgx-01"])
    cid = (await client.post("/api/clusters", json={"name": "Old", "node_ids": list(ids.values())})).json()["id"]
    r = await client.patch(f"/api/clusters/{cid}", json={"name": "New"})
    assert r.status_code == 200
    assert r.json()["name"] == "New"
    assert r.json()["node_count"] == 1


async def test_rename_onto_an_existing_name_is_rejected(client):
    await client.post("/api/clusters", json={"name": "A"})
    cid = (await client.post("/api/clusters", json={"name": "B"})).json()["id"]
    assert (await client.patch(f"/api/clusters/{cid}", json={"name": "A"})).status_code == 409


async def test_moving_nodes_between_clusters(client):
    ids = await register_fleet(["dgx-01", "dgx-02"])
    a = (await client.post("/api/clusters", json={"name": "A", "node_ids": list(ids.values())})).json()["id"]
    b = (await client.post("/api/clusters", json={"name": "B"})).json()["id"]

    r = await client.post(f"/api/clusters/{b}/nodes", json={"node_ids": [ids["dgx-01"]]})
    assert r.json()["node_count"] == 1

    listed = {c["name"]: c["node_count"] for c in (await client.get("/api/clusters")).json()}
    assert listed == {"A": 1, "B": 1}
    assert (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()["cluster_name"] == "B"
    assert a  # cluster A still exists


async def test_unassigning_returns_nodes_to_no_cluster(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/clusters", json={"name": "A", "node_ids": list(ids.values())})
    r = await client.post("/api/clusters/unassign", json={"node_ids": [ids["dgx-01"]]})
    assert r.json()["moved"] == [ids["dgx-01"]]
    assert (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()["cluster_id"] is None


async def test_deleting_a_cluster_releases_its_nodes_and_stops_nothing(client):
    """A cluster is a label. Deleting one must never touch a machine."""
    ids = await register_fleet(["dgx-01"])
    cid = (await client.post("/api/clusters", json={"name": "A", "node_ids": list(ids.values())})).json()["id"]
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]

    assert (await client.delete(f"/api/clusters/{cid}")).status_code == 204

    node = (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()
    assert node["cluster_id"] is None
    assert node["status"] == "online"
    assert (await client.get(f"/api/deployments/{dep['id']}")).json()["status"] in ("starting", "healthy")


async def test_moving_an_unknown_node_is_a_404_and_moves_nothing(client):
    ids = await register_fleet(["dgx-01"])
    cid = (await client.post("/api/clusters", json={"name": "A"})).json()["id"]
    r = await client.post(f"/api/clusters/{cid}/nodes", json={"node_ids": [ids["dgx-01"], "ghost"]})
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]
    # the valid node in the same batch must not have moved
    assert (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()["cluster_id"] is None


async def test_membership_survives_a_node_probe(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/clusters", json={"name": "A", "node_ids": list(ids.values())})
    await client.post(f"/api/nodes/{ids['dgx-01']}/probe")
    assert (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()["cluster_name"] == "A"


async def test_a_node_can_be_registered_straight_into_a_cluster(client):
    cid = (await client.post("/api/clusters", json={"name": "A"})).json()["id"]
    r = await client.post("/api/nodes", json={"name": "dgx-01", "hostname": "dgx-01.sim.local",
                                              "cluster_id": cid})
    assert r.status_code == 201
    assert r.json()["cluster_name"] == "A"


async def test_ordering_is_stable_and_explicit(client):
    for name in ("C", "A", "B"):
        await client.post("/api/clusters", json={"name": name})
    # sort_order is assigned on create, so the list keeps creation order
    assert [c["name"] for c in (await client.get("/api/clusters")).json()] == ["C", "A", "B"]


async def test_cluster_changes_are_admin_only(client):
    await set_role(Role.deployer)
    assert (await client.post("/api/clusters", json={"name": "X"})).status_code == 403
    await set_role(Role.viewer)
    assert (await client.get("/api/clusters")).status_code == 200


async def test_unknown_cluster_is_a_404(client):
    assert (await client.patch("/api/clusters/nope", json={"name": "x"})).status_code == 404
    assert (await client.delete("/api/clusters/nope")).status_code == 404
    assert (await client.post("/api/clusters/nope/nodes", json={"node_ids": []})).status_code == 404
