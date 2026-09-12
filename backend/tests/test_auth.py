"""Roles, team scoping and quotas."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.auth import ROLE_RANK, role_for_groups
from app.db import SessionLocal
from app.models import Deployment, Role, Team, User
from tests.conftest import register_fleet, set_role


async def make_team(name: str, max_gpus: int = 0) -> str:
    async with SessionLocal() as s:
        team = Team(id=str(uuid.uuid4()), name=name, max_gpus=max_gpus)
        s.add(team)
        await s.commit()
        return team.id


# ------------------------------------------------------------- group mapping

def test_idp_groups_map_onto_roles():
    assert role_for_groups(["gpu-admins"]) is Role.admin
    assert role_for_groups(["gpu-deployers"]) is Role.deployer
    assert role_for_groups(["some-other-team"]) is Role.viewer
    assert role_for_groups([]) is Role.viewer


def test_admin_wins_when_a_user_is_in_both_groups():
    assert role_for_groups(["gpu-deployers", "gpu-admins"]) is Role.admin


def test_roles_are_ordered_so_checks_are_inclusive():
    assert ROLE_RANK[Role.admin] > ROLE_RANK[Role.deployer] > ROLE_RANK[Role.viewer]


# -------------------------------------------------------------- enforcement

async def test_dev_mode_provisions_an_admin(client):
    me = (await client.get("/api/auth/me")).json()
    assert me["email"] == "dev@localhost"
    assert me["role"] == "admin"


async def test_a_viewer_is_read_only_everywhere_that_matters(client):
    ids = await register_fleet(["dgx-01"])
    await set_role(Role.viewer)

    writes = [
        ("post", "/api/nodes", {"name": "x", "hostname": "y"}),
        ("post", "/api/clusters", {"name": "x"}),
        ("post", "/api/deployments", {"spec_key": "llama3.1-8b"}),
        ("post", "/api/catalog", {"key": "x", "display_name": "X", "hf_repo": "a/b"}),
        ("post", "/api/litellm/resync", None),
        ("post", f"/api/nodes/{ids['dgx-01']}/drain", None),
    ]
    for method, path, body in writes:
        r = await getattr(client, method)(path, json=body) if body else await getattr(client, method)(path)
        assert r.status_code == 403, f"{path} should be forbidden for a viewer"

    for path in ("/api/nodes", "/api/clusters", "/api/deployments", "/api/catalog",
                 "/api/summary", "/api/events", "/api/audit"):
        assert (await client.get(path)).status_code == 200, f"{path} should be readable"


async def test_a_deployer_can_deploy_but_not_change_the_fleet(client):
    await register_fleet(["dgx-01"])
    await set_role(Role.deployer)
    assert (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})).status_code == 201
    assert (await client.post("/api/nodes", json={"name": "x", "hostname": "y"})).status_code == 403
    assert (await client.post("/api/clusters", json={"name": "x"})).status_code == 403


async def test_the_user_list_is_admin_only(client):
    assert (await client.get("/api/auth/users")).status_code == 200
    await set_role(Role.deployer)
    assert (await client.get("/api/auth/users")).status_code == 403


# ------------------------------------------------------------------- quotas

async def test_a_team_quota_caps_gpu_usage(client):
    await register_fleet(["dgx-01"])
    team = await make_team("research", max_gpus=2)
    await set_role(Role.deployer, team_id=team)

    ok = await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})
    assert ok.status_code == 201, "2 GPUs is exactly the quota"

    over = await client.post("/api/deployments", json={"spec_key": "llama3.1-8b", "replicas": 1})
    assert over.status_code == 409
    assert "quota is 2 GPUs" in over.json()["detail"]
    assert "2 in use" in over.json()["detail"]


async def test_stopping_frees_quota(client):
    await register_fleet(["dgx-01"])
    team = await make_team("research", max_gpus=2)
    await set_role(Role.deployer, team_id=team)

    dep = (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).json()[0]
    await client.post(f"/api/deployments/{dep['id']}/stop")
    assert (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).status_code == 201


async def test_zero_means_unlimited(client):
    await register_fleet(["dgx-01"])
    team = await make_team("unbounded", max_gpus=0)
    await set_role(Role.deployer, team_id=team)
    assert (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).status_code == 201
    assert (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).status_code == 201


async def test_an_admin_is_not_capped(client):
    await register_fleet(["dgx-01"])
    team = await make_team("research", max_gpus=1)
    await set_role(Role.admin, team_id=team)
    assert (await client.post("/api/deployments", json={"spec_key": "qwen3-32b", "replicas": 1})).status_code == 201


async def test_a_deployment_records_its_team(client):
    await register_fleet(["dgx-01"])
    team = await make_team("research")
    await set_role(Role.deployer, team_id=team)
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})).json()[0]
    assert dep["team_id"] == team
    assert dep["created_by"] == "dev@localhost"


# ---------------------------------------------------------------- ownership

async def test_a_deployer_cannot_touch_another_teams_model(client):
    await register_fleet(["dgx-01"])
    mine = await make_team("mine")
    theirs = await make_team("theirs")

    await set_role(Role.admin)
    dep = (await client.post("/api/deployments",
                             json={"spec_key": "llama3.1-8b", "team_id": theirs})).json()[0]

    async with SessionLocal() as s:
        row = await s.execute(select(Deployment).where(Deployment.id == dep["id"]))
        row.scalar_one().created_by = "someone.else@corp"
        await s.commit()

    await set_role(Role.deployer, team_id=mine)
    r = await client.post(f"/api/deployments/{dep['id']}/stop")
    assert r.status_code == 403
    assert "another team" in r.json()["detail"]


async def test_a_deployer_can_stop_their_own_teams_model(client):
    await register_fleet(["dgx-01"])
    team = await make_team("mine")
    await set_role(Role.deployer, team_id=team)
    dep = (await client.post("/api/deployments", json={"spec_key": "llama3.1-8b"})).json()[0]
    assert (await client.post(f"/api/deployments/{dep['id']}/stop")).status_code == 200


async def test_an_admin_can_stop_anything(client):
    await register_fleet(["dgx-01"])
    theirs = await make_team("theirs")
    dep = (await client.post("/api/deployments",
                             json={"spec_key": "llama3.1-8b", "team_id": theirs})).json()[0]
    await set_role(Role.admin)
    assert (await client.post(f"/api/deployments/{dep['id']}/stop")).status_code == 200


async def test_teams_are_listed(client):
    await make_team("alpha")
    await make_team("beta")
    assert {t["name"] for t in (await client.get("/api/auth/teams")).json()} == {"alpha", "beta"}
