from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware

from . import worker
from .mcp_server import build_mcp_app, mcp
from .api import auth_api, catalog, clusters, deployments, fleet, litellm_api, nodes
from .config import settings
from .db import SessionLocal, init_db
from .drivers import close_driver
from .models import Cluster, Node, NodeKind
from .services.catalog import seed as seed_catalog

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("dgxctl")

_tasks: list[asyncio.Task] = []


async def bootstrap() -> None:
    async with SessionLocal() as db:
        added = await seed_catalog(db)
        if added:
            log.info("seeded %d catalog entries", added)

        if settings.driver == "sim":
            existing = await db.execute(select(Node.name))
            have = set(existing.scalars().all())
            from .drivers.sim_driver import SIM_HARDWARE

            # Two starter clusters so the grouping is visible from first boot.
            clusters: dict[str, Cluster] = {}
            for order, (key, cname, desc) in enumerate(
                [
                    ("dgx", "DGX Pod", "H100 / A100 servers in the machine room"),
                    ("rtx", "Lab Workstations", "RTX 6000 Ada boxes at desks"),
                ]
            ):
                row = await db.execute(select(Cluster).where(Cluster.name == cname))
                cluster = row.scalar_one_or_none()
                if cluster is None:
                    cluster = Cluster(name=cname, description=desc, sort_order=order)
                    db.add(cluster)
                    await db.flush()
                clusters[key] = cluster

            new = 0
            for name in SIM_HARDWARE:
                if name in have:
                    continue
                is_dgx = name.startswith("dgx")
                db.add(
                    Node(
                        name=name,
                        hostname=f"{name}.sim.local",
                        kind=NodeKind.dgx if is_dgx else NodeKind.workstation,
                        cluster_id=clusters["dgx" if is_dgx else "rtx"].id,
                        labels={"rack": "r1" if is_dgx else "lab", "sim": "true"},
                    )
                )
                new += 1
            if new:
                await db.commit()
                log.info("registered %d simulated nodes (DGXCTL_DRIVER=sim)", new)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await bootstrap()
    worker.start(_tasks)
    log.info("dgxctl up: driver=%s auth=%s mcp=%s",
             settings.driver, settings.auth_mode, "on" if _mcp_app else "off")
    # The mounted MCP app has its own lifespan that FastAPI will not run, so
    # its session manager is started here.
    async with contextlib.AsyncExitStack() as stack:
        if _mcp_app is not None:
            await stack.enter_async_context(mcp.session_manager.run())
        try:
            yield
        finally:
            for t in _tasks:
                t.cancel()
            await asyncio.gather(*_tasks, return_exceptions=True)
            await close_driver()


_mcp_app = build_mcp_app()

app = FastAPI(title="dgxctl", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    same_site="lax",
    https_only=settings.session_https_only,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (
    auth_api.router, fleet.router, clusters.router, nodes.router,
    deployments.router, catalog.router, litellm_api.router,
):
    app.include_router(r)


if _mcp_app is not None:
    # Mounted before the SPA catch-all, which would otherwise swallow /mcp.
    app.mount("/mcp", _mcp_app)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True, "driver": settings.driver}


# Serve the built UI from the same origin in production, so there is one port to
# open and no CORS to think about.
_static = Path(__file__).resolve().parent.parent / "static"
if _static.is_dir():
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        candidate = _static / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_static / "index.html")
