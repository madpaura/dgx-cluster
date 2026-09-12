from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit, events
from ..auth import current_user, require_admin
from ..db import get_db
from ..models import Cluster, Node, User
from ..schemas import ClusterCreate, ClusterOut, ClusterUpdate, NodeMove

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


@router.get("", response_model=list[ClusterOut])
async def list_clusters(db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    rows = await db.execute(select(Cluster).order_by(Cluster.sort_order, Cluster.name))
    clusters = list(rows.scalars())
    counts = dict(
        (await db.execute(select(Node.cluster_id, func.count(Node.id)).group_by(Node.cluster_id))).all()
    )
    out = []
    for c in clusters:
        item = ClusterOut.model_validate(c)
        item.node_count = counts.get(c.id, 0)
        out.append(item)
    return out


@router.post("", response_model=ClusterOut, status_code=201)
async def create_cluster(
    body: ClusterCreate, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "a cluster needs a name")
    dupe = await db.execute(select(Cluster).where(Cluster.name == name))
    if dupe.scalar_one_or_none():
        raise HTTPException(409, f"a cluster called '{name}' already exists")

    if body.sort_order is None:
        last = await db.execute(select(func.max(Cluster.sort_order)))
        order = (last.scalar() or 0) + 1
    else:
        order = body.sort_order

    cluster = Cluster(
        name=name, description=body.description.strip(), color=body.color, sort_order=order
    )
    db.add(cluster)
    await db.flush()

    if body.node_ids:
        await _assign(db, body.node_ids, cluster.id)

    await audit.record(
        db, actor=user.email, action="cluster.create", target_type="cluster", target_id=cluster.id,
        summary=f"created cluster '{name}'", detail={"nodes": body.node_ids},
    )
    await db.commit()
    events.publish("cluster", {"id": cluster.id, "action": "created"})
    out = ClusterOut.model_validate(cluster)
    out.node_count = len(body.node_ids)
    return out


@router.patch("/{cluster_id}", response_model=ClusterOut)
async def update_cluster(
    cluster_id: str, body: ClusterUpdate, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    cluster = await db.get(Cluster, cluster_id)
    if not cluster:
        raise HTTPException(404, "cluster not found")
    changes = body.model_dump(exclude_unset=True)
    if "name" in changes:
        name = (changes["name"] or "").strip()
        if not name:
            raise HTTPException(400, "a cluster needs a name")
        dupe = await db.execute(select(Cluster).where(Cluster.name == name, Cluster.id != cluster_id))
        if dupe.scalar_one_or_none():
            raise HTTPException(409, f"a cluster called '{name}' already exists")
        changes["name"] = name
    for k, v in changes.items():
        setattr(cluster, k, v)
    await audit.record(
        db, actor=user.email, action="cluster.update", target_type="cluster", target_id=cluster_id,
        summary=f"updated cluster '{cluster.name}'", detail=changes,
    )
    await db.commit()
    events.publish("cluster", {"id": cluster_id, "action": "updated"})
    counted = await db.execute(select(func.count(Node.id)).where(Node.cluster_id == cluster_id))
    out = ClusterOut.model_validate(cluster)
    out.node_count = counted.scalar() or 0
    return out


@router.delete("/{cluster_id}", status_code=204)
async def delete_cluster(
    cluster_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    """Deleting a cluster is a labelling change, never a fleet change: its nodes
    are released to Unassigned and keep serving."""
    cluster = await db.get(Cluster, cluster_id)
    if not cluster:
        raise HTTPException(404, "cluster not found")
    released = await db.execute(
        update(Node).where(Node.cluster_id == cluster_id).values(cluster_id=None)
    )
    name = cluster.name
    await db.delete(cluster)
    await audit.record(
        db, actor=user.email, action="cluster.delete", target_type="cluster", target_id=cluster_id,
        summary=f"deleted cluster '{name}', released {released.rowcount or 0} node(s)",
    )
    await db.commit()
    events.publish("cluster", {"id": cluster_id, "action": "deleted"})


@router.post("/{cluster_id}/nodes", response_model=ClusterOut)
async def move_nodes_in(
    cluster_id: str, body: NodeMove, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    cluster = await db.get(Cluster, cluster_id)
    if not cluster:
        raise HTTPException(404, "cluster not found")
    moved = await _assign(db, body.node_ids, cluster_id)
    await audit.record(
        db, actor=user.email, action="cluster.assign", target_type="cluster", target_id=cluster_id,
        summary=f"moved {len(moved)} node(s) into '{cluster.name}'", detail={"nodes": moved},
    )
    await db.commit()
    events.publish("cluster", {"id": cluster_id, "action": "assigned"})
    counted = await db.execute(select(func.count(Node.id)).where(Node.cluster_id == cluster_id))
    out = ClusterOut.model_validate(cluster)
    out.node_count = counted.scalar() or 0
    return out


@router.post("/unassign", status_code=200)
async def unassign_nodes(
    body: NodeMove, db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)
):
    moved = await _assign(db, body.node_ids, None)
    await audit.record(
        db, actor=user.email, action="cluster.unassign", target_type="cluster", target_id="",
        summary=f"released {len(moved)} node(s) to Unassigned", detail={"nodes": moved},
    )
    await db.commit()
    events.publish("cluster", {"id": "", "action": "unassigned"})
    return {"moved": moved}


async def _assign(db: AsyncSession, node_ids: list[str], cluster_id: str | None) -> list[str]:
    if not node_ids:
        return []
    rows = await db.execute(select(Node).where(Node.id.in_(node_ids)))
    nodes = list(rows.scalars().unique())
    found = {n.id for n in nodes}
    missing = [n for n in node_ids if n not in found]
    if missing:
        raise HTTPException(404, f"unknown node(s): {', '.join(missing)}")
    for node in nodes:
        node.cluster_id = cluster_id
    return [n.id for n in nodes]
