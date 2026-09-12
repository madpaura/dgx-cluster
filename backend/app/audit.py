from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from . import events
from .models import AuditLog, Event

log = logging.getLogger(__name__)


async def record(
    db: AsyncSession,
    *,
    actor: str,
    action: str,
    target_type: str = "",
    target_id: str = "",
    summary: str = "",
    detail: dict | None = None,
    ok: bool = True,
) -> None:
    db.add(
        AuditLog(
            actor=actor, action=action, target_type=target_type, target_id=target_id,
            summary=summary, detail=detail or {}, ok=ok,
        )
    )
    log.info("audit actor=%s action=%s target=%s ok=%s", actor, action, target_id, ok)


async def emit(
    db: AsyncSession,
    *,
    severity: str,
    source: str,
    source_id: str,
    message: str,
    detail: dict | None = None,
) -> None:
    """Operator-facing event. Persisted and pushed to connected dashboards."""
    db.add(Event(severity=severity, source=source, source_id=source_id, message=message, detail=detail or {}))
    events.publish(
        "event",
        {"severity": severity, "source": source, "source_id": source_id, "message": message},
    )
