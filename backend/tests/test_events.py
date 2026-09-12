"""The live-update bus behind the dashboard websocket."""
from __future__ import annotations

import asyncio
import json

from app import events
from tests.conftest import pump, register_fleet


def test_publish_reaches_every_subscriber():
    with events.subscription() as a, events.subscription() as b:
        events.publish("node", {"id": "n1", "status": "online"})
        for q in (a, b):
            msg = json.loads(q.get_nowait())
            assert msg == {"topic": "node", "data": {"id": "n1", "status": "online"}}


def test_unsubscribing_stops_delivery():
    q = events.subscribe()
    events.unsubscribe(q)
    events.publish("node", {"id": "n1"})
    assert q.empty()


def test_publishing_with_nobody_listening_is_free():
    events.publish("node", {"id": "n1"})          # must not raise


def test_a_subscriber_that_stops_reading_is_dropped_not_tolerated():
    """A dashboard that stalls must never back up the poller."""
    q = events.subscribe()
    try:
        for i in range(400):                       # queue caps at 200
            events.publish("metrics", {"i": i})
        assert q not in events._subscribers
    finally:
        events.unsubscribe(q)


def test_non_serialisable_payloads_do_not_break_the_bus():
    from datetime import datetime

    with events.subscription() as q:
        events.publish("event", {"ts": datetime.now()})
        assert "topic" in json.loads(q.get_nowait())


async def test_deployment_transitions_are_published(client):
    """What the UI actually relies on: state changes arrive without polling."""
    await register_fleet(["dgx-01"])
    with events.subscription() as q:
        dep = (await client.post("/api/deployments",
                                 json={"spec_key": "llama3.1-8b", "replicas": 1})).json()[0]
        await pump()

        topics, messages = set(), []
        while not q.empty():
            m = json.loads(q.get_nowait())
            topics.add(m["topic"])
            messages.append(m)

    assert "deployment" in topics
    assert "metrics" in topics
    assert any(m["data"].get("id") == dep["id"] and m["data"].get("status") == "healthy"
               for m in messages if m["topic"] == "deployment")


async def test_node_probes_publish_gpu_state(client):
    with events.subscription() as q:
        await register_fleet(["dgx-01"])
        payloads = [json.loads(q.get_nowait()) for _ in range(q.qsize())]

    node_msg = next(m for m in payloads if m["topic"] == "node")
    assert len(node_msg["data"]["gpus"]) == 8
    assert {"index", "util", "mem_used_mb", "temp_c"} <= set(node_msg["data"]["gpus"][0])


async def test_cluster_changes_are_published(client):
    with events.subscription() as q:
        await client.post("/api/clusters", json={"name": "A"})
        msgs = [json.loads(q.get_nowait()) for _ in range(q.qsize())]
    assert any(m["topic"] == "cluster" and m["data"]["action"] == "created" for m in msgs)


async def test_the_websocket_route_is_mounted(client):
    """Exercised for real by the browser; here we only prove it is wired up.

    Included routers are nested, and this FastAPI version wraps them, so walk
    through `original_router` rather than assuming a flat list."""
    from starlette.routing import WebSocketRoute

    from app.main import app

    def walk(router):
        for route in getattr(router, "routes", []):
            yield route
            inner = getattr(route, "original_router", None) or getattr(route, "app", None)
            if inner is not None:
                yield from walk(inner)

    sockets = [r for r in walk(app.router) if isinstance(r, WebSocketRoute)]
    assert [r.path for r in sockets] == ["/api/ws"]


def test_slow_subscriber_cleanup_leaves_others_working():
    slow = events.subscribe()
    with events.subscription() as fast:
        for i in range(400):
            events.publish("metrics", {"i": i})
            while not fast.empty():
                fast.get_nowait()
        assert slow not in events._subscribers
        events.publish("node", {"id": "still-working"})
        assert json.loads(fast.get_nowait())["data"]["id"] == "still-working"
