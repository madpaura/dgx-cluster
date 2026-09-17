"""Portal settings, and the import endpoint that uses them."""
from __future__ import annotations

import pytest

from app.services import catalog_import, llm, sizing
from tests.test_catalog_import import CARD, CONFIG, GOOD_REPLY


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    sizing._CONFIG_CACHE.clear()

    async def card(url, headers=None):
        return CARD if url.endswith("README.md") else ""

    async def config(repo, revision="main", token=""):
        return CONFIG

    async def reply(cfg, *, system, user, **kw):
        return llm.Completion(text=GOOD_REPLY, model="gpt-oss")

    monkeypatch.setattr(catalog_import, "_get_text", card)
    monkeypatch.setattr(sizing, "fetch_config", config)
    monkeypatch.setattr(catalog_import, "complete", reply)


async def configure(client, **over):
    body = {"enabled": True, "base_url": "", "model": "gpt-oss",
            "api_key": "sk-secret-value-1234", "timeout_s": 60, "max_input_chars": 24000}
    body.update(over)
    return await client.put("/api/settings/llm", json=body)


# ---------------------------------------------------------------- settings

async def test_settings_start_empty_and_default_to_the_fleets_own_proxy(client):
    got = (await client.get("/api/settings/llm")).json()
    assert got["enabled"] is False
    assert got["has_api_key"] is False
    # Blank base_url means LiteLLM next door: no external account to serve models.
    assert got["base_url"] == ""
    assert "litellm" in got["effective_base_url"]


async def test_the_api_key_goes_in_but_never_comes_back(client):
    """A settings page that renders the key is a settings page that leaks it into
    browser history, screenshots and support tickets."""
    saved = (await configure(client)).json()
    assert saved["has_api_key"] is True
    assert saved["api_key_hint"] == "…1234"

    body = (await client.get("/api/settings/llm")).text
    assert "sk-secret-value-1234" not in body
    assert "api_key" not in (await client.get("/api/settings/llm")).json()


async def test_saving_again_without_the_key_keeps_the_stored_one(client):
    """The form cannot resend what it was never shown, so blank must mean
    "leave it alone" rather than "delete it"."""
    await configure(client)
    again = (await configure(client, api_key="", model="other-model")).json()
    assert again["has_api_key"] is True
    assert again["model"] == "other-model"


async def test_a_null_key_clears_it_deliberately(client):
    await configure(client)
    cleared = (await configure(client, api_key=None)).json()
    assert cleared["has_api_key"] is False


async def test_settings_survive_a_round_trip(client):
    await configure(client, base_url="https://api.openai.com", model="gpt-4o-mini")
    got = (await client.get("/api/settings/llm")).json()
    assert got["model"] == "gpt-4o-mini"
    assert got["effective_base_url"] == "https://api.openai.com"
    assert got["updated_by"]


async def test_nonsense_timeouts_are_rejected(client):
    assert (await configure(client, timeout_s=99999)).status_code == 422


# ------------------------------------------------------------------ import

async def test_import_returns_a_draft_and_saves_nothing(client):
    await configure(client)
    before = len((await client.get("/api/catalog")).json())

    draft = (await client.post("/api/catalog/import", json={
        "url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"})).json()

    assert draft["display_name"] == "Llama 3.1 8B Instruct"
    assert draft["min_gpu_memory_gb"] > 0
    assert draft["sizing_source"] == "model config"
    assert len((await client.get("/api/catalog")).json()) == before, \
        "the operator reviews the draft; import must not write to the catalog"


async def test_the_draft_can_then_be_saved_through_the_normal_endpoint(client):
    await configure(client)
    draft = (await client.post("/api/catalog/import", json={
        "url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"})).json()

    body = {k: v for k, v in draft.items()
            if k not in ("source_url", "sizing_source", "warnings")}
    created = await client.post("/api/catalog", json=body)
    assert created.status_code == 201, created.text
    assert created.json()["key"] == draft["key"]


async def test_import_without_an_llm_points_at_settings(client):
    r = await client.post("/api/catalog/import", json={"url": "https://huggingface.co/org/model"})
    assert r.status_code == 409
    assert "Settings" in r.json()["detail"]


async def test_import_refuses_a_host_outside_the_allowlist(client):
    await configure(client)
    r = await client.post("/api/catalog/import", json={"url": "http://169.254.169.254/latest/"})
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"]


async def test_an_llm_that_is_unreachable_is_a_bad_gateway_not_a_crash(client, monkeypatch):
    await configure(client)

    async def broken(cfg, *, system, user, **kw):
        raise llm.LLMError("Could not reach the model at http://llm.invalid: timed out")

    monkeypatch.setattr(catalog_import, "complete", broken)
    r = await client.post("/api/catalog/import", json={
        "url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"})
    assert r.status_code == 502
    assert "Could not reach" in r.json()["detail"]


async def test_the_test_button_reports_a_failure_instead_of_erroring(client):
    """Settings → Test exists to show you what is wrong, so a broken endpoint is
    a 200 carrying the reason, not a 500."""
    await configure(client, base_url="http://nothing-here.invalid")
    r = await client.post("/api/settings/llm/test")
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["detail"]
