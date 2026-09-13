from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import current_user, require_admin
from ..config import settings
from ..db import get_db
from ..drivers import get_driver
from ..models import ACTIVE_STATUSES, Deployment, Node, NodeStatus, User
from ..schemas import (
    FindingOut, GpuTenant, NodeAuthorize, NodeCreate, NodeOut, NodeUpdate,
)
from ..services import diagnostics
from ..services import nodes as node_svc

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


async def _load(db: AsyncSession, node_id: str) -> Node | None:
    """Fetch a node with its relationships eagerly loaded.

    populate_existing because this is called after probing, which inserts GPU
    rows: without it the session hands back the instance it already holds, whose
    collections were loaded before those rows existed.
    """
    rows = await db.execute(
        select(Node).where(Node.id == node_id).execution_options(populate_existing=True)
    )
    return rows.scalars().unique().one_or_none()


def to_out(node: Node) -> NodeOut:
    """Attach what is running on each GPU so the fleet view needs one call.

    Plural: a GPU holds as many models as its VRAM allows. deployment_id and
    model_name name the first of them, which is what a single-line summary can
    show; tenants carries the rest.
    """
    residents: dict[int, list] = {}
    for d in node.deployments:
        if d.status in ACTIVE_STATUSES:
            for i in d.gpu_indices:
                residents.setdefault(int(i), []).append(d)

    out = NodeOut.model_validate(node)
    out.cluster_name = node.cluster.name if node.cluster else ""
    for gpu in out.gpus:
        here = residents.get(gpu.index, [])
        gpu.tenants = [
            GpuTenant(deployment_id=d.id, model_name=d.served_model_name,
                      reserved_mb=int(d.reserved_mb_per_gpu or 0))
            for d in here
        ]
        gpu.reserved_mb = sum(t.reserved_mb for t in gpu.tenants)
        if here:
            gpu.deployment_id, gpu.model_name = here[0].id, here[0].served_model_name
    return out


@router.get("", response_model=list[NodeOut])
async def list_nodes(db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    rows = await db.execute(select(Node).order_by(Node.name))
    return [to_out(n) for n in rows.scalars().unique()]


@router.get("/{node_id}", response_model=NodeOut)
async def get_node(node_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    return to_out(node)


@router.post("", response_model=NodeOut, status_code=201)
async def create_node(body: NodeCreate, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)):
    exists = await db.execute(select(Node).where(Node.name == body.name))
    if exists.scalar_one_or_none():
        raise HTTPException(409, f"node '{body.name}' already registered")
    node = Node(**body.model_dump())
    db.add(node)
    await db.commit()
    await audit.record(
        db, actor=user.email, action="node.create", target_type="node", target_id=node.id,
        summary=f"registered {node.name} ({node.hostname})",
    )
    await node_svc.refresh(db, node)   # immediate feedback: did SSH work?
    await db.commit()
    # Re-read so the relationship loaders populate gpus/deployments/cluster for
    # serialisation; the object we just built has never been through a loader.
    return to_out(await _load(db, node.id))


@router.patch("/{node_id}", response_model=NodeOut)
async def update_node(
    node_id: str, body: NodeUpdate, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(node, k, v)
    await audit.record(
        db, actor=user.email, action="node.update", target_type="node", target_id=node.id,
        summary=f"updated {node.name}", detail=body.model_dump(exclude_unset=True),
    )
    await db.commit()
    return to_out(node)


@router.delete("/{node_id}", status_code=204)
async def delete_node(node_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    active = [d for d in node.deployments if d.status in ACTIVE_STATUSES]
    if active:
        raise HTTPException(409, f"{node.name} still has {len(active)} active deployment(s); stop them first")
    name = node.name
    # Stopped and failed deployments still reference this node, and node_id is
    # NOT NULL, so they go with it. The audit log keeps the history.
    await db.execute(delete(Deployment).where(Deployment.node_id == node_id))
    await db.delete(node)
    await audit.record(
        db, actor=user.email, action="node.delete", target_type="node", target_id=node_id,
        summary=f"removed {name}",
    )
    await db.commit()


@router.post("/{node_id}/probe", response_model=NodeOut)
async def probe_node(node_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    await node_svc.refresh(db, node)
    return to_out(node)


@router.post("/{node_id}/drain", response_model=NodeOut)
async def drain_node(
    node_id: str, undo: bool = False, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    """Stop scheduling new work here without touching what is already running."""
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    node.status = NodeStatus.unknown if undo else NodeStatus.draining
    await audit.record(
        db, actor=user.email, action="node.drain", target_type="node", target_id=node.id,
        summary=f"{'undrained' if undo else 'drained'} {node.name}",
    )
    await db.commit()
    if undo:
        await node_svc.refresh(db, node)
    return to_out(node)


@router.post("/{node_id}/reconcile")
async def reconcile_node(node_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    """Compare the node's real containers with our records; surface orphans."""
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    return await node_svc.adopt_and_prune(db, node)


@router.post("/{node_id}/authorize", response_model=NodeOut)
async def authorize_node(
    node_id: str,
    body: NodeAuthorize,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Install the control server's public key on a node, using a password once.

    A node that has never met dgxctl refuses key authentication, which is what
    an operator hits the first time they rack hardware. Rather than asking them
    to go and run ssh-copy-id, this does it — once — and then never needs the
    password again.
    """
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")

    public_key_path = f"{settings.ssh_key_path}.pub"
    try:
        public_key = Path(public_key_path).read_text().strip()
    except OSError as exc:
        raise HTTPException(
            500,
            f"cannot read the control server's public key at {public_key_path}. "
            f"Generate the pair with ./setup.sh keygen.",
        ) from exc

    try:
        await get_driver().install_authorized_key(node, body.password, public_key)
    except NotImplementedError as exc:
        raise HTTPException(400, "this driver cannot install keys") from exc
    except Exception as exc:
        await audit.record(
            db, actor=user.email, action="node.authorize", target_type="node", target_id=node.id,
            summary=f"could not install a key on {node.name}", detail={"error": str(exc)[:300]},
            ok=False,
        )
        await db.commit()
        raise HTTPException(400, str(exc)) from exc

    await audit.record(
        db, actor=user.email, action="node.authorize", target_type="node", target_id=node.id,
        summary=f"installed the control server's key on {node.name}",
    )
    await node_svc.refresh(db, node)     # prove the key works before answering
    await db.commit()

    loaded = await _load(db, node.id)
    if loaded.status == NodeStatus.unreachable:
        raise HTTPException(
            502,
            f"the key was installed but {node.name} still will not accept it: "
            f"{loaded.last_error[:200]}",
        )
    return to_out(loaded)


@router.get("/{node_id}/diagnostics", response_model=list[FindingOut])
async def node_diagnostics(node_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(404, "node not found")
    return [FindingOut(**f.__dict__) for f in diagnostics.analyze_node(node)]
