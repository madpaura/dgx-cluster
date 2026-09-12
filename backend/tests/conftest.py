"""Test harness.

Everything runs against the simulated fleet and a throwaway SQLite file, in
process, with no network. The worker loops are driven by hand so each test
observes a deterministic point in the reconcile cycle instead of racing a
background task.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

# Settings are read at import time, so the environment has to be set up first.
_TMP = Path(tempfile.mkdtemp(prefix="dgxctl-test-"))
os.environ.update(
    DGXCTL_DATABASE_URL=f"sqlite+aiosqlite:///{_TMP / 'test.db'}",
    DGXCTL_DRIVER="sim",
    DGXCTL_AUTH_MODE="dev",
    DGXCTL_LITELLM_BASE_URL="http://litellm.invalid",
    DGXCTL_LITELLM_AUTO_REGISTER="false",
    DGXCTL_SECRET_KEY="test-secret",
)

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from app import worker  # noqa: E402
from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.drivers import get_driver  # noqa: E402
from app.drivers.sim_driver import SimDriver  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base, Cluster, Node, NodeKind, Role, User  # noqa: E402
from app.services.catalog import seed as seed_catalog  # noqa: E402

# Weights "load" instantly; the delay is realism we do not want in tests.
SimDriver.STARTUP_SECONDS = 0.0


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db():
    async with SessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
async def fresh_db():
    """Drop and rebuild the schema between tests, and reset the fake fleet so
    containers from one test never leak into the next."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    driver = get_driver()
    if isinstance(driver, SimDriver):
        driver._nodes.clear()
    async with SessionLocal() as session:
        await seed_catalog(session)
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------- fleet setup

SIM_NODES = [
    ("dgx-01", NodeKind.dgx),
    ("dgx-02", NodeKind.dgx),
    ("dgx-03", NodeKind.dgx),
    ("dgx-04", NodeKind.dgx),        # the simulator keeps this one unreachable
    ("rtx-ws-01", NodeKind.workstation),
    ("rtx-ws-02", NodeKind.workstation),
    ("rtx-ws-03", NodeKind.workstation),
]


async def register_fleet(names: list[str] | None = None) -> dict[str, str]:
    """Create node rows and probe them once. Returns name -> id."""
    wanted = names or [n for n, _ in SIM_NODES]
    ids: dict[str, str] = {}
    async with SessionLocal() as session:
        for name, kind in SIM_NODES:
            if name not in wanted:
                continue
            node = Node(
                id=str(uuid.uuid4()), name=name, hostname=f"{name}.sim.local", kind=kind,
                labels={"sim": "true"},
            )
            session.add(node)
            ids[name] = node.id
        await session.commit()
        await worker.inventory_pass(session)
    return ids


async def make_cluster(name: str, node_ids: list[str] | None = None) -> str:
    async with SessionLocal() as session:
        cluster = Cluster(id=str(uuid.uuid4()), name=name)
        session.add(cluster)
        await session.flush()
        if node_ids:
            for nid in node_ids:
                node = await session.get(Node, nid)
                node.cluster_id = cluster.id
        await session.commit()
        return cluster.id


async def set_role(role: Role, team_id: str | None = None) -> None:
    """The dev-auth user is an admin by default; tests that check permissions
    downgrade it."""
    from sqlalchemy import select

    async with SessionLocal() as session:
        rows = await session.execute(select(User))
        user = rows.scalars().first()
        if user is None:
            user = User(email="dev@localhost", name="Local Admin")
            session.add(user)
        user.role = role
        user.team_id = team_id
        await session.commit()


async def pump(times: int = 1) -> None:
    """Run one full worker cycle: observe containers, then scrape metrics."""
    for _ in range(times):
        async with SessionLocal() as session:
            await worker.reconcile_pass(session)
        async with SessionLocal() as session:
            await worker.metrics_pass(session)
