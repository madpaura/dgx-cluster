"""Bringing a node dgxctl has never met under management.

A freshly racked machine has no reason to trust the control server, so key
authentication fails and the node registers as unreachable. Asking the operator
to go and run ssh-copy-id by hand is the point at which a dashboard stops being
one, so it is done here: a password, used once, never kept.
"""
from __future__ import annotations

import pathlib

import pytest

from app.config import settings
from app.drivers import get_driver
from app.drivers.sim_driver import SIM_NODE_PASSWORD
from app.models import Role
from tests.conftest import register_fleet, set_role

NEW_NODE = {"name": "dgx-05", "hostname": "dgx-05.sim.local"}


@pytest.fixture
def public_key(tmp_path, monkeypatch):
    """A key pair for the control server, as ./setup.sh keygen would leave."""
    key = tmp_path / "fleet_key"
    key.write_text("PRIVATE")
    pub = tmp_path / "fleet_key.pub"
    pub.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI dgxctl\n")
    monkeypatch.setattr(settings, "ssh_key_path", str(key))
    return pub.read_text().strip()


async def test_a_node_that_has_never_met_us_registers_as_unreachable(client, public_key):
    r = await client.post("/api/nodes", json=NEW_NODE)
    assert r.status_code == 201
    node = r.json()
    assert node["status"] == "unreachable"
    assert "publickey" in node["last_error"]


async def test_a_password_installs_the_key_and_the_node_comes_up(client, public_key):
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    r = await client.post(f"/api/nodes/{node['id']}/authorize",
                          json={"password": SIM_NODE_PASSWORD})
    assert r.status_code == 200
    after = r.json()
    assert after["status"] == "online"
    assert len(after["gpus"]) > 0, "probing must succeed on the key alone"


async def test_the_wrong_password_is_refused_without_locking_anything_up(client, public_key):
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    r = await client.post(f"/api/nodes/{node['id']}/authorize", json={"password": "hunter2"})
    assert r.status_code == 400
    assert "not accepted" in r.json()["detail"]

    # and the right one still works afterwards
    ok = await client.post(f"/api/nodes/{node['id']}/authorize",
                           json={"password": SIM_NODE_PASSWORD})
    assert ok.status_code == 200


async def test_the_password_is_never_written_to_the_audit_trail(client, public_key):
    """The whole point of using it once is that it does not persist anywhere."""
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    await client.post(f"/api/nodes/{node['id']}/authorize", json={"password": SIM_NODE_PASSWORD})

    entries = (await client.get("/api/audit")).json()
    assert SIM_NODE_PASSWORD not in repr(entries)
    installed = next(e for e in entries if e["action"] == "node.authorize")
    assert installed["ok"] is True
    assert "installed the control server's key" in installed["summary"]


async def test_a_failed_attempt_is_audited_too(client, public_key):
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    await client.post(f"/api/nodes/{node['id']}/authorize", json={"password": "wrong"})

    entries = (await client.get("/api/audit")).json()
    assert "wrong" not in repr(entries)
    failed = next(e for e in entries if e["action"] == "node.authorize")
    assert failed["ok"] is False


async def test_installing_a_key_twice_is_harmless(client, public_key):
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    for _ in range(2):
        r = await client.post(f"/api/nodes/{node['id']}/authorize",
                              json={"password": SIM_NODE_PASSWORD})
        assert r.status_code == 200


async def test_a_missing_public_key_says_how_to_make_one(client, monkeypatch):
    monkeypatch.setattr(settings, "ssh_key_path", "/nonexistent/fleet_key")
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    r = await client.post(f"/api/nodes/{node['id']}/authorize",
                          json={"password": SIM_NODE_PASSWORD})
    assert r.status_code == 500
    assert "setup.sh keygen" in r.json()["detail"]


async def test_an_unreachable_node_cannot_be_bootstrapped(client, public_key):
    """No password helps when the machine is not answering at all."""
    ids = await register_fleet(["dgx-04"])
    r = await client.post(f"/api/nodes/{ids['dgx-04']}/authorize",
                          json={"password": SIM_NODE_PASSWORD})
    assert r.status_code == 400
    assert "could not reach" in r.json()["detail"]


async def test_only_an_admin_may_install_a_key(client, public_key):
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    await set_role(Role.deployer)
    r = await client.post(f"/api/nodes/{node['id']}/authorize",
                          json={"password": SIM_NODE_PASSWORD})
    assert r.status_code == 403


async def test_the_key_survives_for_later_deployments(client, public_key):
    """Once is once: nothing afterwards should ask for the password again."""
    node = (await client.post("/api/nodes", json=NEW_NODE)).json()
    await client.post(f"/api/nodes/{node['id']}/authorize", json={"password": SIM_NODE_PASSWORD})

    probed = await client.post(f"/api/nodes/{node['id']}/probe")
    assert probed.json()["status"] == "online"

    deployed = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "node_ids": [node["id"]]})
    assert deployed.status_code == 201
