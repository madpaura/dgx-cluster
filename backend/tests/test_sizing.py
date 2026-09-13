"""Sizing a model and judging its settings before anything is launched."""
from __future__ import annotations

import pytest

from app.services import sizing
from app.services.sizing import assess, kv_cache_gb, params_from_config, params_from_name

# Shaped like a real Llama-3.1-8B config.json.
LLAMA_8B = {
    "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
    "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
}
# 70B, and deliberately 80 heads so 3-way tensor parallel does not divide.
LLAMA_70B = {
    "num_hidden_layers": 80, "hidden_size": 8192, "intermediate_size": 28672,
    "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Never reach huggingface.co from a test; each test says what it returns."""
    sizing._CONFIG_CACHE.clear()
    monkeypatch.setattr(sizing, "fetch_config", _served(None))


def _served(config):
    async def _fetch(repo, revision="main", token=""):
        return config
    return _fetch


def serve(monkeypatch, config):
    monkeypatch.setattr(sizing, "fetch_config", _served(config))


# --------------------------------------------------------------- estimating

def test_parameter_count_is_read_from_the_architecture():
    """Within a few percent of the 8.03B the model card states."""
    assert 7.0 < params_from_config(LLAMA_8B) < 9.0
    assert 60 < params_from_config(LLAMA_70B) < 80


def test_the_repo_name_is_the_last_resort():
    assert params_from_name("meta-llama/Llama-3.1-8B-Instruct") == 8.0
    assert params_from_name("Qwen/Qwen3-32B") == 32.0
    assert params_from_name("org/something-unlabelled") == 7.0


def test_grouped_query_attention_is_what_makes_the_kv_cache_affordable():
    """8 KV heads rather than 32 is a fourfold saving, and the whole reason a
    long context fits at all."""
    gqa = kv_cache_gb(LLAMA_8B, 8192, 64, 1, 2)
    mha = kv_cache_gb({**LLAMA_8B, "num_key_value_heads": 32}, 8192, 64, 1, 2)
    assert mha == pytest.approx(gqa * 4, rel=0.01)


def test_the_kv_cache_grows_with_context_and_batch():
    small = kv_cache_gb(LLAMA_8B, 4096, 32, 1, 2)
    longer = kv_cache_gb(LLAMA_8B, 8192, 32, 1, 2)
    busier = kv_cache_gb(LLAMA_8B, 4096, 64, 1, 2)
    assert longer == pytest.approx(small * 2, rel=0.01)
    assert busier == pytest.approx(small * 2, rel=0.01)


def test_tensor_parallel_divides_the_requirement(monkeypatch):
    assert kv_cache_gb(LLAMA_70B, 8192, 32, 4, 2) == pytest.approx(
        kv_cache_gb(LLAMA_70B, 8192, 32, 1, 2) / 4, rel=0.01)


async def test_an_8b_model_is_sized_at_roughly_what_it_really_needs(monkeypatch):
    serve(monkeypatch, LLAMA_8B)
    a = await assess(hf_repo="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel=1,
                     max_model_len=8192, max_num_seqs=64)
    assert a.estimate.source == "model config"
    assert 15 < a.estimate.weights_gb < 18          # ~8B at 2 bytes
    assert 18 < a.estimate.total_gb_per_gpu < 32     # comfortably inside a 48 GiB card
    assert a.estimate.minimum_gb_per_gpu < a.estimate.total_gb_per_gpu
    assert a.estimate.concurrent_sequences == 8
    assert not a.blocking


async def test_quantization_shrinks_the_weights(monkeypatch):
    serve(monkeypatch, LLAMA_70B)
    full = await assess(hf_repo="x/llama-70b", tensor_parallel=2, max_model_len=8192)
    fp8 = await assess(hf_repo="x/llama-70b", tensor_parallel=2, max_model_len=8192,
                       quantization="fp8", gpu_model="NVIDIA H100 80GB HBM3")
    assert fp8.estimate.weights_gb == pytest.approx(full.estimate.weights_gb / 2, rel=0.05)


async def test_the_estimate_explains_itself(monkeypatch):
    serve(monkeypatch, LLAMA_8B)
    a = await assess(hf_repo="x/y", tensor_parallel=1, max_model_len=4096, max_num_seqs=32)
    assert "parameters" in a.estimate.detail and "KV cache" in a.estimate.detail
    assert a.estimate.weights_gb + a.estimate.kv_cache_gb + a.estimate.overhead_gb == \
        pytest.approx(a.estimate.total_gb_per_gpu, abs=0.2)


async def test_an_unreadable_config_falls_back_and_says_so(monkeypatch):
    a = await assess(hf_repo="private/model", tensor_parallel=1)
    assert a.estimate.source == "repo name"
    assert any(c.title == "Model config could not be read" for c in a.checks)
    assert "HF_TOKEN" in next(c for c in a.checks if c.severity == "warning").fix
    assert not a.blocking, "a guess is a warning, not a refusal"


async def test_a_catalog_entry_beats_a_guess(monkeypatch):
    a = await assess(hf_repo="private/model", tensor_parallel=1, catalog_gb=44)
    assert a.estimate.source == "catalog"
    assert a.estimate.total_gb_per_gpu == 44


# --------------------------------------------------------------- validating

async def test_a_context_longer_than_the_model_supports_is_refused(monkeypatch):
    serve(monkeypatch, {**LLAMA_8B, "max_position_embeddings": 8192})
    a = await assess(hf_repo="x/y", tensor_parallel=1, max_model_len=32768)
    blocked = a.blocking[0]
    assert blocked.title == "Context length exceeds what the model supports"
    assert "8192" in blocked.detail and "8192" in blocked.fix


async def test_a_tensor_parallel_size_that_does_not_divide_the_heads_is_refused(monkeypatch):
    serve(monkeypatch, LLAMA_8B)                  # 32 attention heads
    a = await assess(hf_repo="x/y", tensor_parallel=3)
    blocked = a.blocking[0]
    assert "does not divide" in blocked.title
    assert "32" in blocked.detail
    assert "2" in blocked.fix and "4" in blocked.fix


async def test_kv_heads_that_do_not_divide_are_only_a_warning(monkeypatch):
    serve(monkeypatch, LLAMA_8B)                  # 8 KV heads, 32 attention heads
    a = await assess(hf_repo="x/y", tensor_parallel=16)
    kv_warning = next(c for c in a.checks if "Key/value heads" in c.title)
    assert kv_warning.severity == "warning"
    assert "replicate" in kv_warning.detail


async def test_fp8_on_an_ampere_card_is_refused(monkeypatch):
    serve(monkeypatch, LLAMA_70B)
    a = await assess(hf_repo="x/y", tensor_parallel=2, quantization="fp8",
                     gpu_model="NVIDIA A100-SXM4-80GB")
    blocked = a.blocking[0]
    assert blocked.title == "FP8 needs Hopper or newer"
    assert "AWQ" in blocked.fix


async def test_fp8_on_hopper_is_fine(monkeypatch):
    serve(monkeypatch, LLAMA_70B)
    a = await assess(hf_repo="x/y", tensor_parallel=2, quantization="fp8",
                     gpu_model="NVIDIA H100 80GB HBM3")
    assert not a.blocking


async def test_a_quantization_that_contradicts_the_checkpoint_is_refused(monkeypatch):
    serve(monkeypatch, {**LLAMA_8B, "quantization_config": {"quant_method": "awq", "bits": 4}})
    a = await assess(hf_repo="x/y", tensor_parallel=1, quantization="gptq")
    blocked = a.blocking[0]
    assert blocked.title == "Quantization does not match the checkpoint"
    assert "awq" in blocked.fix


async def test_asking_for_more_gpus_than_the_node_has_is_refused(monkeypatch):
    serve(monkeypatch, LLAMA_70B)
    a = await assess(hf_repo="x/y", tensor_parallel=8, gpus_available=4)
    assert any(c.title == "Not enough GPUs for this tensor-parallel size" for c in a.blocking)


async def test_an_expensive_context_length_is_flagged(monkeypatch):
    """131k tokens fits on paper and serves almost nothing in practice."""
    serve(monkeypatch, {**LLAMA_70B, "max_position_embeddings": 131072})
    a = await assess(hf_repo="x/y", tensor_parallel=8, max_model_len=131072, max_num_seqs=1)
    warning = next(c for c in a.checks if "expensive to serve" in c.title)
    assert warning.severity == "warning"
    assert "Halving" in warning.fix


async def test_the_floor_is_reported_separately_from_the_comfortable_size(monkeypatch):
    """vLLM refuses to start below one full sequence of KV cache; above that it
    is a throughput choice, not a yes/no."""
    serve(monkeypatch, LLAMA_8B)
    a = await assess(hf_repo="x/y", tensor_parallel=1, max_model_len=8192, max_num_seqs=64)
    assert a.estimate.minimum_gb_per_gpu == pytest.approx(
        a.estimate.weights_gb + a.estimate.kv_cache_gb / 8 + a.estimate.overhead_gb, abs=0.3)
    assert "one request at a time" in a.estimate.detail


async def test_the_config_is_fetched_once_and_cached(monkeypatch):
    calls = {"n": 0}

    async def counting(repo, revision="main", token=""):
        calls["n"] += 1
        return LLAMA_8B

    monkeypatch.setattr(sizing, "fetch_config", counting)
    for _ in range(3):
        await assess(hf_repo="x/y", tensor_parallel=1)
    assert calls["n"] == 3, "assess always asks; fetch_config is what caches"


async def test_the_plan_endpoint_carries_the_estimate_and_the_checks(client, monkeypatch):
    """What the deploy dialog and an agent both read before committing to
    anything."""
    from tests.conftest import register_fleet

    serve(monkeypatch, LLAMA_8B)
    await register_fleet(["dgx-01"])
    plan = (await client.post("/api/deployments/plan", json={
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct", "tensor_parallel_size": 1,
        "max_model_len": 8192,
    })).json()

    est = plan["estimate"]
    assert est["source"] == "model config"
    assert est["weights_gb"] > 0 and est["kv_cache_gb"] > 0
    assert est["minimum_gb_per_gpu"] < est["total_gb_per_gpu"]
    assert plan["blocked"] is False


async def test_a_deploy_with_impossible_settings_is_refused_before_anything_starts(client, monkeypatch):
    """The whole point: knowable now, rather than a container that dies several
    minutes into pulling weights."""
    from tests.conftest import register_fleet

    serve(monkeypatch, {**LLAMA_8B, "max_position_embeddings": 8192})
    await register_fleet(["dgx-01"])
    r = await client.post("/api/deployments", json={
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct", "served_model_name": "too-long",
        "tensor_parallel_size": 1, "max_model_len": 200000,
    })
    assert r.status_code == 422
    assert "8192" in r.json()["detail"]
    assert (await client.get("/api/deployments")).json() == [], "nothing may be claimed"


async def test_a_warning_does_not_block_a_deploy(client, monkeypatch):
    """Only settings that cannot work are refused; advice stays advice."""
    from tests.conftest import register_fleet

    serve(monkeypatch, None)          # unreadable config -> warning, not error
    await register_fleet(["dgx-01"])
    r = await client.post("/api/deployments", json={
        "spec_key": "llama3.1-8b", "replicas": 1,
    })
    assert r.status_code == 201
