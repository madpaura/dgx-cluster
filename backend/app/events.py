"""In-process pub/sub so the UI updates without polling.

Single control-server process, so a plain asyncio fan-out is enough. If you ever
run the API replicated, swap this for Redis pub/sub — the interface is two calls.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def publish(topic: str, payload: dict[str, Any]) -> None:
    if not _subscribers:
        return
    msg = json.dumps({"topic": topic, "data": payload}, default=str)
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            # Slow client: drop it rather than stalling the poller.
            _subscribers.discard(q)


@contextlib.contextmanager
def subscription():
    q = subscribe()
    try:
        yield q
    finally:
        unsubscribe(q)
