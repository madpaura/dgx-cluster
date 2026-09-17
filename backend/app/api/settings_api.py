"""Portal settings an admin edits in the browser.

Only the drafting LLM for now. Everything else about dgxctl is decided at deploy
time by env vars, which is right for anything needed before the database is up;
this is for the handful of things an operator should be able to change without a
redeploy.

The API key is write-only: it is accepted on PUT and never returned on GET.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth import require_admin
from ..db import get_db
from ..models import AppSetting, User
from ..schemas import LLMSettingsIn, LLMSettingsOut, LLMTestOut
from ..services import llm

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _out(cfg: llm.LLMConfig, row: AppSetting | None) -> LLMSettingsOut:
    return LLMSettingsOut(
        enabled=cfg.enabled,
        base_url=cfg.base_url,
        model=cfg.model,
        timeout_s=cfg.timeout_s,
        max_input_chars=cfg.max_input_chars,
        has_api_key=bool(cfg.api_key),
        api_key_hint=f"…{cfg.api_key[-4:]}" if len(cfg.api_key) > 4 else "",
        effective_base_url=cfg.effective_base_url,
        updated_at=row.updated_at if row else None,
        updated_by=row.updated_by if row else "",
    )


@router.get("/llm", response_model=LLMSettingsOut)
async def get_llm_settings(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    cfg = await llm.load_config(db)
    return _out(cfg, await db.get(AppSetting, llm.SETTING_KEY))


@router.put("/llm", response_model=LLMSettingsOut)
async def put_llm_settings(
    body: LLMSettingsIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    cfg = await llm.load_config(db)
    cfg.enabled = body.enabled
    cfg.base_url = body.base_url.strip()
    cfg.model = body.model.strip()
    cfg.timeout_s = body.timeout_s
    cfg.max_input_chars = body.max_input_chars

    # "" keeps whatever is stored (the form never sees it, so it cannot resend
    # it); null clears it deliberately.
    if body.api_key is None:
        cfg.api_key = ""
    elif body.api_key.strip():
        cfg.api_key = body.api_key.strip()

    await llm.save_config(db, cfg, actor=user.email)
    await audit.record(
        db, actor=user.email, action="settings.llm", target_type="settings", target_id="llm",
        summary=f"LLM {'enabled' if cfg.enabled else 'disabled'}"
                f"{f' — {cfg.model} via {cfg.effective_base_url}' if cfg.enabled else ''}",
    )
    await db.commit()
    return _out(cfg, await db.get(AppSetting, llm.SETTING_KEY))


@router.post("/llm/test", response_model=LLMTestOut)
async def test_llm(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    """Prove the settings work now, rather than when someone tries an import."""
    cfg = await llm.load_config(db)
    try:
        reply = await llm.complete(
            cfg,
            system="Reply with the single word: ok",
            user="ping",
            max_tokens=8,
        )
    except llm.LLMNotConfigured as exc:
        raise HTTPException(400, str(exc)) from exc
    except llm.LLMError as exc:
        return LLMTestOut(ok=False, detail=str(exc))
    return LLMTestOut(
        ok=True,
        detail=f"Answered in {len(reply.text.strip())} characters.",
        model=reply.model,
    )
