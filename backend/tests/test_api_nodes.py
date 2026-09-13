"""Node inventory over the real HTTP surface, against the simulated fleet."""
from __future__ import annotations

from app.models import Role
from app.services.deployments import drain_launches
from tests.conftest import pump, register_fleet, set_role


async def test_registering_a_node_probes_it_immediately(client):
    r = await client.post("/api/nodes", json={"name": "dgx-01", "hostname": "dgx-01.sim.local"})
    assert r.status_code == 201
    node = r.json()
    # The point of probing on create: you learn at once whether SSH works.
    assert node["status"] == "online"
    assert len(node["gpus"]) == 8
    assert node["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3"
    assert node["driver_version"] == "550.90.07"
    assert node["cpu_count"] == 224


async def test_an_unreachable_node_registers_but_reports_the_error(client):
    r = await client.post("/api/nodes", json={"name": "dgx-04", "hostname": "dgx-04.sim.local"})
    assert r.status_code == 201
    node = r.json()
    assert node["status"] == "unreachable"
    assert "No route to host" in node["last_error"]
    assert node["gpus"] == []


async def test_duplicate_node_name_is_rejected(client):
    await client.post("/api/nodes", json={"name": "dgx-01", "hostname": "h"})
    r = await client.post("/api/nodes", json={"name": "dgx-01", "hostname": "h"})
    assert r.status_code == 409
    assert "already registered" in r.json()["detail"]


async def test_workstations_and_dgx_are_the_same_kind_of_thing(client):
    await client.post("/api/nodes", json={"name": "rtx-ws-03", "hostname": "rtx-ws-03.sim.local",
                                          "kind": "workstation"})
    nodes = (await client.get("/api/nodes")).json()
    ws = nodes[0]
    assert ws["kind"] == "workstation"
    assert len(ws["gpus"]) == 4
    assert "RTX 6000" in ws["gpus"][0]["name"]


async def test_gpu_rows_carry_what_is_running_on_them(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    node = (await client.get(f"/api/nodes/{ids['dgx-01']}")).json()
    claimed = [g for g in node["gpus"] if g["deployment_id"]]
    assert len(claimed) == 1
    assert claimed[0]["model_name"] == "llama3.1-8b"
    assert all(g["model_name"] is None for g in node["gpus"] if not g["deployment_id"])


async def test_probe_refreshes_a_node_on_demand(client):
    ids = await register_fleet(["dgx-01"])
    r = await client.post(f"/api/nodes/{ids['dgx-01']}/probe")
    assert r.status_code == 200
    assert r.json()["last_seen"] is not None


async def test_draining_stops_new_placement_without_touching_what_runs(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})

    r = await client.post(f"/api/nodes/{ids['dgx-01']}/drain")
    assert r.json()["status"] == "draining"

    # existing deployment untouched
    deps = (await client.get("/api/deployments")).json()
    assert len(deps) == 1

    # but nothing new lands here
    plan = (await client.post("/api/deployments/plan", json={"spec_key": "llama3.1-8b"})).json()
    assert plan["placements"] == []
    assert plan["rejections"][0]["reason"] == "node is draining"

    r = await client.post(f"/api/nodes/{ids['dgx-01']}/drain", params={"undo": "true"})
    assert r.json()["status"] == "online"


async def test_a_drained_node_refuses_an_explicit_deploy(client):
    ids = await register_fleet(["dgx-01"])
    await client.post(f"/api/nodes/{ids['dgx-01']}/drain")
    r = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b",
        "targets": [{"node_id": ids["dgx-01"], "gpu_indices": [0]}],
    })
    assert r.status_code == 409
    assert "draining" in r.json()["detail"]


async def test_a_node_with_live_deployments_cannot_be_deleted(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    r = await client.delete(f"/api/nodes/{ids['dgx-01']}")
    assert r.status_code == 409
    assert "active deployment" in r.json()["detail"]


async def test_a_node_can_be_deleted_once_it_is_empty(client):
    ids = await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await client.post(f"/api/deployments/{dep['id']}/stop")
    assert (await client.delete(f"/api/nodes/{ids['dgx-01']}")).status_code == 204
    assert (await client.get("/api/nodes")).json() == []


async def test_reconcile_reports_a_node_that_matches_our_records(client):
    ids = await register_fleet(["dgx-01"])
    await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    r = (await client.post(f"/api/nodes/{ids['dgx-01']}/reconcile")).json()
    assert r == {"error": "", "orphans": [], "missing": []}


async def test_reconcile_catches_a_container_that_vanished(client):
    """Simulates the node rebooting behind our back."""
    from app.drivers import get_driver

    ids = await register_fleet(["dgx-01"])
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
    await drain_launches()
    get_driver()._nodes["dgx-01"].containers.clear()

    r = (await client.post(f"/api/nodes/{ids['dgx-01']}/reconcile")).json()
    assert r["missing"] == [dep["id"]]
    after = (await client.get(f"/api/deployments/{dep['id']}")).json()
    assert after["status"] == "failed"
    assert "gone from the node" in after["status_reason"]


async def test_node_diagnostics_surface_unreachability(client):
    ids = await register_fleet(["dgx-04"])
    findings = (await client.get(f"/api/nodes/{ids['dgx-04']}/diagnostics")).json()
    assert findings[0]["code"] == "node_unreachable"
    assert findings[0]["severity"] == "error"


async def test_registering_a_node_is_admin_only(client):
    await set_role(Role.deployer)
    r = await client.post("/api/nodes", json={"name": "dgx-01", "hostname": "h"})
    assert r.status_code == 403
    assert "requires admin" in r.json()["detail"]


async def test_everyone_can_read_the_fleet(client):
    await register_fleet(["dgx-01"])
    await set_role(Role.viewer)
    assert (await client.get("/api/nodes")).status_code == 200


async def test_missing_node_is_a_404(client):
    assert (await client.get("/api/nodes/nope")).status_code == 404
    assert (await client.post("/api/nodes/nope/probe")).status_code == 404
