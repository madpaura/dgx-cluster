"""How much GPU a model will actually need, and whether these settings can work.

Two jobs, both done before anything is launched. Estimating turns "will it fit"
into arithmetic instead of a guess from the repo name. Validating catches the
settings vLLM will reject — a context length the model does not have, a
tensor-parallel size that does not divide its heads — which otherwise surface
as a failed container several minutes into a weight download.

Everything degrades: the model's own config.json is the best source, the
catalog is the operator's measured truth, and the repo name is the last resort.
Whichever was used is reported, because an estimate you cannot trace is worse
than no estimate.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

HF_CONFIG_URL = "https://huggingface.co/{repo}/resolve/{revision}/config.json"
_CONFIG_CACHE: dict[str, tuple[float, dict | None]] = {}
CACHE_SECONDS = 3600

# Bytes per parameter once loaded.
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
QUANT_BYTES = {"fp8": 1, "awq": 0.5, "gptq": 0.5, "int4": 0.5, "bitsandbytes": 0.5, "": None}

# CUDA context, NCCL buffers, cuBLAS workspace and activation peaks. Roughly
# flat per process rather than proportional to the model.
OVERHEAD_GB = 2.0

# How many full-length sequences to budget KV cache for.
#
# Not max_num_seqs: that is an upper bound vLLM will never all reach at full
# context, and sizing against it asks for 80 GiB to serve an 8B model. vLLM
# allocates whatever is left after the weights and serves as many tokens as fit,
# so the useful question is how much concurrency the reservation buys. Its own
# hard floor is one sequence — below that it refuses to start.
KV_SIZING_SEQUENCES = 8


@dataclass
class Estimate:
    total_gb_per_gpu: float
    weights_gb: float
    kv_cache_gb: float                # at KV_SIZING_SEQUENCES concurrent requests
    overhead_gb: float
    tensor_parallel: int
    max_model_len: int
    source: str                       # where the numbers came from
    params_b: float = 0.0
    minimum_gb_per_gpu: float = 0.0   # weights plus one sequence: vLLM's floor
    concurrent_sequences: int = KV_SIZING_SEQUENCES
    detail: str = ""


@dataclass
class Check:
    severity: str                     # "error" blocks a deploy, "warning" does not
    title: str
    detail: str
    fix: str


@dataclass
class Assessment:
    estimate: Estimate
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "error"]


async def fetch_config(repo: str, revision: str = "main", token: str = "") -> dict | None:
    """The model's own config.json, or None when it cannot be read.

    Cached: the deploy dialog re-plans on every keystroke, and this is a network
    call to someone else's server.
    """
    key = f"{repo}@{revision or 'main'}"
    hit = _CONFIG_CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    url = HF_CONFIG_URL.format(repo=repo, revision=revision or "main")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        config = r.json() if r.status_code == 200 else None
        if r.status_code in (401, 403):
            log.info("config.json for %s needs a token", repo)
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("could not read config.json for %s: %s", repo, exc)
        config = None

    _CONFIG_CACHE[key] = (time.time(), config)
    return config


def params_from_name(repo: str) -> float:
    """Last resort: the parameter count most repos put in their name."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", repo.rsplit("/", 1)[-1])
    return float(match.group(1)) if match else 7.0


def params_from_config(config: dict) -> float:
    """Approximate parameter count from the transformer's shape.

    Transformer blocks dominate; embeddings are counted because they matter for
    small models, where they can be a fifth of the total.
    """
    layers = config.get("num_hidden_layers") or config.get("n_layer") or 0
    hidden = config.get("hidden_size") or config.get("n_embd") or 0
    inter = config.get("intermediate_size") or hidden * 4
    vocab = config.get("vocab_size") or 0
    heads = config.get("num_attention_heads") or 1
    kv_heads = config.get("num_key_value_heads") or heads
    if not (layers and hidden):
        return 0.0

    head_dim = config.get("head_dim") or hidden // max(heads, 1)
    attn = hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim) + (heads * head_dim) * hidden
    mlp = 3 * hidden * inter            # gated MLPs have three matrices
    per_layer = attn + mlp
    return (layers * per_layer + 2 * vocab * hidden) / 1e9


def kv_cache_gb(config: dict, max_model_len: int, max_num_seqs: int, tp: int,
                cache_dtype_bytes: float) -> float:
    """VRAM the KV cache needs for the whole batch, split across the TP group.

    2 (K and V) x layers x kv_heads x head_dim x tokens x bytes. Grouped-query
    attention is what makes this affordable on modern models: kv_heads is a
    fraction of the attention heads.
    """
    layers = config.get("num_hidden_layers") or config.get("n_layer") or 0
    hidden = config.get("hidden_size") or config.get("n_embd") or 0
    heads = config.get("num_attention_heads") or 1
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or (hidden // max(heads, 1))
    if not layers or not head_dim:
        return 0.0
    per_token = 2 * layers * kv_heads * head_dim * cache_dtype_bytes
    return per_token * max_model_len * max_num_seqs / (1024 ** 3) / max(tp, 1)


def _weight_bytes(config: dict | None, quantization: str) -> float:
    if quantization and QUANT_BYTES.get(quantization) is not None:
        return QUANT_BYTES[quantization]
    if config:
        declared = (config.get("quantization_config") or {}).get("bits")
        if declared:
            return declared / 8
        dtype = str(config.get("torch_dtype") or "bfloat16")
        return DTYPE_BYTES.get(dtype, 2)
    return 2


async def assess(
    *,
    hf_repo: str,
    tensor_parallel: int,
    max_model_len: int = 0,
    max_num_seqs: int = 256,
    quantization: str = "",
    revision: str = "",
    catalog_gb: float = 0.0,
    hf_token: str = "",
    gpu_model: str = "",
    gpus_available: int = 0,
) -> Assessment:
    """Size the model and check the settings against what it actually declares."""
    config = await fetch_config(hf_repo, revision, hf_token)
    checks: list[Check] = []

    if config is None:
        params = params_from_name(hf_repo)
        weights = params * _weight_bytes(None, quantization) / max(tensor_parallel, 1)
        sized = catalog_gb or (weights + 8)
        estimate = Estimate(
            total_gb_per_gpu=round(sized, 1),
            weights_gb=round(weights, 1),
            kv_cache_gb=round(max(0.0, sized - weights - OVERHEAD_GB), 1),
            overhead_gb=OVERHEAD_GB,
            minimum_gb_per_gpu=round(weights + OVERHEAD_GB, 1),
            tensor_parallel=tensor_parallel,
            max_model_len=max_model_len,
            source="catalog" if catalog_gb else "repo name",
            params_b=round(params, 1),
            detail=("Sized from the catalog entry." if catalog_gb else
                    "Could not read the model's config.json, so this is a guess from "
                    "the repo name. Set the catalog entry from a real deployment."),
        )
        checks.append(Check(
            "warning", "Model config could not be read",
            f"huggingface.co did not return config.json for {hf_repo}.",
            "Check the repo id and, for a gated model, that HF_TOKEN is set on the "
            "control server. Without it the memory estimate is only a guess.",
        ))
        return Assessment(estimate, checks)

    params = params_from_config(config) or params_from_name(hf_repo)
    ctx_limit = config.get("max_position_embeddings") or 0
    effective_len = max_model_len or ctx_limit or 4096

    bytes_per_param = _weight_bytes(config, quantization)
    weights = params * bytes_per_param / max(tensor_parallel, 1)
    batch = min(max_num_seqs, KV_SIZING_SEQUENCES)
    kv = kv_cache_gb(config, effective_len, batch, tensor_parallel, 2)
    floor = kv_cache_gb(config, effective_len, 1, tensor_parallel, 2)
    total = weights + kv + OVERHEAD_GB

    estimate = Estimate(
        total_gb_per_gpu=round(total, 1),
        weights_gb=round(weights, 1),
        kv_cache_gb=round(kv, 1),
        overhead_gb=OVERHEAD_GB,
        tensor_parallel=tensor_parallel,
        max_model_len=effective_len,
        source="model config",
        params_b=round(params, 1),
        minimum_gb_per_gpu=round(weights + floor + OVERHEAD_GB, 1),
        concurrent_sequences=batch,
        detail=(f"{params:.1f}B parameters at {bytes_per_param:g} bytes each, "
                f"split {tensor_parallel} ways, plus a KV cache holding "
                f"{batch} concurrent requests at {effective_len} tokens. "
                f"It will start with as little as "
                f"{weights + floor + OVERHEAD_GB:.0f} GiB, serving one request "
                f"at a time."),
    )

    # ---- settings vLLM will reject -------------------------------------
    if max_model_len and ctx_limit and max_model_len > ctx_limit:
        checks.append(Check(
            "error", "Context length exceeds what the model supports",
            f"--max-model-len {max_model_len} is longer than this model's "
            f"{ctx_limit}-token limit.",
            f"Lower it to {ctx_limit} or less. Going beyond needs RoPE scaling, "
            f"which this checkpoint does not declare.",
        ))

    heads = config.get("num_attention_heads") or 0
    kv_heads = config.get("num_key_value_heads") or heads
    if tensor_parallel > 1 and heads and heads % tensor_parallel:
        checks.append(Check(
            "error", "Tensor-parallel size does not divide the attention heads",
            f"{heads} attention heads cannot be split across {tensor_parallel} GPUs.",
            f"Use a tensor-parallel size that divides {heads} — "
            f"{', '.join(str(n) for n in _divisors(heads) if n <= 8)}.",
        ))
    if tensor_parallel > 1 and kv_heads and kv_heads % tensor_parallel:
        checks.append(Check(
            "warning", "Key/value heads do not divide evenly",
            f"{kv_heads} KV heads across {tensor_parallel} GPUs; vLLM will "
            f"replicate them, costing extra KV cache on every GPU.",
            f"A tensor-parallel size dividing {kv_heads} avoids the duplication.",
        ))

    if quantization == "fp8" and gpu_model and not _is_hopper_or_newer(gpu_model):
        checks.append(Check(
            "error", "FP8 needs Hopper or newer",
            f"{gpu_model} cannot run FP8 kernels.",
            "Deploy this on an H100/H200, or pick an AWQ or GPTQ build for "
            "older cards.",
        ))

    declared = (config.get("quantization_config") or {}).get("quant_method")
    if quantization and declared and quantization != declared:
        checks.append(Check(
            "error", "Quantization does not match the checkpoint",
            f"You asked for {quantization}; the checkpoint declares {declared}.",
            f"Clear the override and let vLLM read it from the config, or use "
            f"{declared}.",
        ))

    if gpus_available and tensor_parallel > gpus_available:
        checks.append(Check(
            "error", "Not enough GPUs for this tensor-parallel size",
            f"Tensor parallel {tensor_parallel} needs {tensor_parallel} GPUs; "
            f"the chosen node has {gpus_available}.",
            "Lower the tensor-parallel size or choose a larger node.",
        ))

    # When one request's KV cache rivals the weights, memory buys concurrency
    # rather than capability, and the model will preempt under very little load.
    if floor > max(2.0, weights * 0.25):
        checks.append(Check(
            "warning", "This context length is expensive to serve",
            f"A single {effective_len}-token request needs {floor:.1f} GiB of KV "
            f"cache, against {weights:.1f} GiB of weights, so concurrency will be "
            f"limited by memory rather than compute.",
            "Lower --max-model-len, or give the model a larger share of the GPU. "
            "Halving the context halves this.",
        ))

    return Assessment(estimate, checks)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0] + \
           [n // d for d in range(int(math.isqrt(n)), 0, -1) if n % d == 0 and d != n // d]


def _is_hopper_or_newer(gpu_model: str) -> bool:
    return any(tag in gpu_model.upper() for tag in ("H100", "H200", "H800", "B100", "B200", "GB200"))
