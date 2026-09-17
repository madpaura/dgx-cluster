"""vLLM flag validation at the edges.

The broad "is this flag allowed" cases live in test_api_deployments.py; this
file holds the ones that came out of running the real thing — a value of the
wrong type reaching sizing before validation, and fix text that has to read as
a sentence because an operator sees it verbatim.
"""
from __future__ import annotations


async def test_a_flag_with_the_wrong_type_is_refused_not_crashed(client):
    """Sizing reads --max-num-seqs to work out the KV cache. It runs before the
    flags are judged, so a non-numeric value used to reach int() and 500."""
    from tests.conftest import register_fleet

    await register_fleet(["dgx-01"])
    r = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "replicas": 1,
        "extra_args": {"--max-num-seqs": "lots"},
    })
    assert r.status_code == 422, r.text
    assert "whole number" in r.json()["detail"]
    assert (await client.get("/api/deployments")).json() == []


async def test_planning_survives_a_flag_with_the_wrong_type(client):
    from tests.conftest import register_fleet

    await register_fleet(["dgx-01"])
    plan = await client.post("/api/deployments/plan", json={
        "spec_key": "llama3.1-8b", "extra_args": {"--max-num-seqs": "lots"}})
    assert plan.status_code == 200
    assert plan.json()["blocked"] is True


def test_every_managed_flag_reads_as_a_sentence():
    """These are shown to an operator verbatim, so "Remove it and use dgxctl
    allocates the port" is a bug, not a wording preference."""
    from app.services.vllm_args import MANAGED, validate

    for flag in MANAGED:
        issue = validate({flag: "x"})[0]
        assert issue.fix[0].isupper(), f"{flag}: fix does not start a sentence"
        assert issue.fix.endswith("."), f"{flag}: fix is not a sentence"
        assert "use dgxctl" not in issue.fix, f"{flag}: {issue.fix}"
