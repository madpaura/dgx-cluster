from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import current_user, require_deployer
from ..db import get_db
from ..models import ModelSpec, User
from ..schemas import ModelSpecIn, ModelSpecOut
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
    await db.delete(spec)
    await audit.record(
        db, actor=user.email, action="catalog.delete", target_type="model_spec", target_id=spec_id,
        summary=f"removed {spec.display_name}",
    )
    await db.commit()


@router.post("/seed")
async def reseed(db: AsyncSession = Depends(get_db), _: User = Depends(require_deployer)):
    return {"added": await seed(db)}
