from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import current_user, require_deployer
from ..db import get_db
from ..models import ACTIVE_STATUSES, Deployment, ModelSpec, Node, User
from ..schemas import CatalogDraftOut, CatalogImportIn, ModelSpecIn, ModelSpecOut
from ..services import catalog_import, llm
from ..services.catalog import seed

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("", response_model=list[ModelSpecOut])
async def list_specs(db: AsyncSession = Depends(get_db), _: User = Depends(current_user)):
    rows = await db.execute(select(ModelSpec).order_by(ModelSpec.display_name))
    return list(rows.scalars())


@router.post("", response_model=ModelSpecOut, status_code=201)
async def create_spec(body: ModelSpecIn, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    dupe = await db.execute(select(ModelSpec).where(ModelSpec.key == body.key))
    if dupe.scalar_one_or_none():
        raise HTTPException(409, f"catalog key '{body.key}' already exists")
    spec = ModelSpec(**body.model_dump())
    db.add(spec)
    await audit.record(
        db, actor=user.email, action="catalog.create", target_type="model_spec", target_id=spec.id,
        summary=f"added {body.display_name} ({body.hf_repo})",
    )
    await db.commit()
    return spec


@router.put("/{spec_id}", response_model=ModelSpecOut)
async def update_spec(
    spec_id: str, body: ModelSpecIn, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)
):
    spec = await db.get(ModelSpec, spec_id)
    if not spec:
        raise HTTPException(404, "catalog entry not found")
    for k, v in body.model_dump().items():
        setattr(spec, k, v)
    await audit.record(
        db, actor=user.email, action="catalog.update", target_type="model_spec", target_id=spec_id,
        summary=f"updated {body.display_name}",
    )
    await db.commit()
    return spec


@router.delete("/{spec_id}", status_code=204)
async def delete_spec(spec_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_deployer)):
    spec = await db.get(ModelSpec, spec_id)
    if not spec:
        raise HTTPException(404, "catalog entry not found")

    # Deployments point back at the entry they were launched from. A running one
    # means this is still in use — say so rather than letting the foreign key
    # turn it into a 500. Finished ones only reference it for provenance, and
    # they carry their own repo and model name, so detaching loses nothing.
    # Node names come from the join rather than dep.node: a relationship access
    # here is a lazy load, which async cannot do.
    rows = await db.execute(
        select(Deployment, Node.name)
        .outerjoin(Node, Deployment.node_id == Node.id)
        .where(Deployment.spec_id == spec_id)
    )
    using = [(dep, node_name) for dep, node_name in rows.unique()]
    live = [(d, n) for d, n in using if d.status in ACTIVE_STATUSES]
    if live:
        where = ", ".join(sorted({n for _, n in live if n})) or "the fleet"
        raise HTTPException(
            409,
            f"{spec.display_name} is still deployed ({len(live)} on {where}). "
            "Stop those deployments first, or edit the entry instead of deleting it.",
        )
    for dep, _name in using:
        dep.spec_id = None

    await db.delete(spec)
    await audit.record(
        db, actor=user.email, action="catalog.delete", target_type="model_spec", target_id=spec_id,
        summary=f"removed {spec.display_name}",
    )
    await db.commit()


@router.post("/seed")
async def reseed(db: AsyncSession = Depends(get_db), _: User = Depends(require_deployer)):
    return {"added": await seed(db)}


@router.post("/import", response_model=CatalogDraftOut)
async def import_from_url(
    body: CatalogImportIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_deployer),
):
    """Read a model page and propose a catalog entry. Saves nothing.

    The draft comes back for the operator to check and then POST to /api/catalog
    like any other entry — a model card is someone else's text, and it is not
    going to write to our catalog unreviewed.
    """
    cfg = await llm.load_config(db)
    try:
        draft = await catalog_import.build_draft(cfg, body.url)
    except catalog_import.ImportError_ as exc:
        raise HTTPException(400, str(exc)) from exc
    except llm.LLMNotConfigured as exc:
        raise HTTPException(409, str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    await audit.record(
        db, actor=user.email, action="catalog.import", target_type="model_spec", target_id="",
        summary=f"drafted {draft.display_name or draft.key} from {body.url}",
    )
    await db.commit()
    return CatalogDraftOut(**draft.__dict__)
