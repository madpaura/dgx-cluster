"""Drafting a catalog entry from a model-card URL.

Nothing here touches the network: the fetches and the LLM are both stubbed, the
same way test_sizing.py stubs config.json.
"""
from __future__ import annotations

import pytest

from app.services import catalog_import, llm, sizing
from app.services.catalog_import import ImportError_, coerce, parse_url
from app.services.llm import LLMConfig, extract_json

CARD = """
# Llama 3.1 8B Instruct

Meta's 8 billion parameter instruction-tuned model. 128k context.
Licensed under the Llama 3.1 Community License. Gated: accept the terms first.
"""

CONFIG = {
    "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
    "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
}

GOOD_REPLY = """```json
{"key": "Llama 3.1 8B!", "display_name": "Llama 3.1 8B Instruct",
 "hf_repo": "meta-llama/Llama-3.1-8B-Instruct", "params_b": 8,
 "quantization": "", "recommended_tp": 1, "max_model_len": 8192,
 "extra_args": {"--enable-prefix-caching": true}, "tags": ["chat", "Instruct"],
 "notes": "Gated on Hugging Face; accept the licence before deploying."}
```"""

ENABLED = LLMConfig(enabled=True, model="gpt-oss", base_url="http://llm.invalid")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Card, config and model all served locally; each test overrides what it needs."""
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


def serve_reply(monkeypatch, text):
    async def reply(cfg, *, system, user, **kw):
        return llm.Completion(text=text, model="gpt-oss")
    monkeypatch.setattr(catalog_import, "complete", reply)


# ------------------------------------------------------------------- the URL

def test_a_hugging_face_model_url_is_understood():
    src = parse_url("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    assert src.kind == "huggingface"
    assert src.repo == "meta-llama/Llama-3.1-8B-Instruct"


def test_a_bare_host_is_treated_as_https():
    assert parse_url("huggingface.co/org/model").repo == "org/model"


def test_a_github_url_is_understood():
    src = parse_url("https://github.com/vllm-project/vllm")
    assert src.kind == "github" and src.repo == "vllm-project/vllm"


def test_a_tree_or_blob_suffix_does_not_confuse_it():
    assert parse_url("https://huggingface.co/org/model/tree/main").repo == "org/model"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/org/model",
        "http://169.254.169.254/latest/meta-data/",      # cloud metadata
        "http://localhost:8080/api/summary",             # our own control plane
        "http://postgres:5432/",                         # a service on the compose network
        "file:///etc/passwd",
    ],
)
def test_only_hugging_face_and_github_may_be_fetched(url):
    """The control server sits inside the office network. An endpoint that
    fetches whatever it is handed is an SSRF hole aimed at your own machines."""
    with pytest.raises(ImportError_):
        parse_url(url)


def test_a_dataset_page_is_not_a_model():
    with pytest.raises(ImportError_) as exc:
        parse_url("https://huggingface.co/datasets/squad")
    assert "not a model" in str(exc.value)


def test_a_url_with_no_repo_is_rejected():
    with pytest.raises(ImportError_):
        parse_url("https://huggingface.co/meta-llama")


# ------------------------------------------------------------------ the draft

async def test_a_model_card_becomes_a_reviewable_draft():
    draft = await catalog_import.build_draft(ENABLED, "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    assert draft.display_name == "Llama 3.1 8B Instruct"
    assert draft.hf_repo == "meta-llama/Llama-3.1-8B-Instruct"
    assert draft.key == "llama-3.1-8b"          # punctuation and spaces cleaned out
    assert draft.tags == ["chat", "instruct"]   # lowercased
    assert draft.extra_args == {"--enable-prefix-caching": True}
    assert "licence" in draft.notes.lower()
    assert draft.source_url.endswith("Llama-3.1-8B-Instruct")


async def test_the_vram_figure_is_measured_not_drafted():
    """The LLM is not allowed near the number that gates deployment: a confident
    guess there would defeat the OOM check it feeds."""
    draft = await catalog_import.build_draft(ENABLED, "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    assert draft.sizing_source == "model config"
    assert 18 < draft.min_gpu_memory_gb < 40
    assert not any("estimate" in w for w in draft.warnings)


async def test_an_unreadable_config_is_flagged_on_the_draft(monkeypatch):
    async def missing(repo, revision="main", token=""):
        return None
    monkeypatch.setattr(sizing, "fetch_config", missing)

    draft = await catalog_import.build_draft(ENABLED, "https://huggingface.co/private/model")
    assert draft.sizing_source == "repo name"
    assert any("estimate" in w for w in draft.warnings)


async def test_a_github_page_without_a_hugging_face_repo_says_so(monkeypatch):
    serve_reply(monkeypatch, '{"key": "vllm", "display_name": "vLLM", "hf_repo": ""}')
    draft = await catalog_import.build_draft(ENABLED, "https://github.com/vllm-project/vllm")
    assert draft.hf_repo == ""
    assert draft.sizing_source == "not sized"
    assert any("Fill in the repo" in w for w in draft.warnings)


async def test_a_page_with_nothing_readable_is_refused(monkeypatch):
    async def nothing(url, headers=None):
        return ""

    async def no_config(repo, revision="main", token=""):
        return None

    monkeypatch.setattr(catalog_import, "_get_text", nothing)
    monkeypatch.setattr(sizing, "fetch_config", no_config)
    with pytest.raises(ImportError_) as exc:
        await catalog_import.build_draft(ENABLED, "https://huggingface.co/private/gated")
    assert "private or gated" in str(exc.value)


async def test_junk_from_the_model_is_an_error_not_a_broken_draft(monkeypatch):
    serve_reply(monkeypatch, "I'm sorry, I can't help with that.")
    with pytest.raises(llm.LLMError):
        await catalog_import.build_draft(ENABLED, "https://huggingface.co/org/model")


async def test_an_unconfigured_llm_says_where_to_configure_it():
    with pytest.raises(llm.LLMNotConfigured) as exc:
        await catalog_import.build_draft(LLMConfig(), "https://huggingface.co/org/model")
    assert "Settings" in str(exc.value)


# ------------------------------------------------- coercing a creative reply

def test_the_repo_on_a_hugging_face_import_is_ours_not_the_models():
    """The URL already told us the repo. A model card that names a different one
    — by mistake or on purpose — does not get to redirect where we pull from."""
    src = catalog_import.Source(kind="huggingface", repo="org/real", url="u")
    assert coerce({"hf_repo": "attacker/other"}, src).hf_repo == "org/real"


def test_absurd_values_are_bounded_rather_than_trusted():
    src = catalog_import.Source(kind="huggingface", repo="org/m", url="u")
    draft = coerce(
        {"recommended_tp": 999, "params_b": "not a number", "quantization": "magic",
         "max_model_len": -5, "tags": ["a"] * 50, "extra_args": {"--x": {"nested": 1}}},
        src,
    )
    assert draft.recommended_tp == 1
    assert draft.params_b == 0.0
    assert draft.quantization == ""
    assert draft.max_model_len == 0
    assert len(draft.tags) <= 5
    assert draft.extra_args == {}, "only scalar flag values survive"


def test_a_missing_key_falls_back_to_the_repo_name():
    src = catalog_import.Source(kind="huggingface", repo="org/Mixtral-8x7B", url="u")
    assert coerce({}, src).key == "mixtral-8x7b"


# --------------------------------------------------------------- JSON rescue

@pytest.mark.parametrize(
    "text",
    [
        '{"key": "a"}',
        '```json\n{"key": "a"}\n```',
        '```\n{"key": "a"}\n```',
        'Sure! Here is the entry:\n\n{"key": "a"}\n\nLet me know if you need changes.',
    ],
)
def test_json_is_recovered_however_the_model_wraps_it(text):
    assert extract_json(text)["key"] == "a"


def test_a_reply_with_no_json_is_an_error():
    with pytest.raises(llm.LLMError):
        extract_json("no json here")
