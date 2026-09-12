"""Seed model catalog.

Sizes are the per-GPU VRAM needed at the listed tensor-parallel size, with room
for a working KV cache — not just the weights. Tune them once against your own
hardware and placement stops being guesswork.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ModelSpec

SEED: list[dict] = [
    {
        "key": "qwen3-32b", "display_name": "Qwen3 32B", "hf_repo": "Qwen/Qwen3-32B",
        "params_b": 32, "min_gpu_memory_gb": 40, "recommended_tp": 2, "max_model_len": 32768,
        "tags": ["general", "chat"], "extra_args": {"--enable-prefix-caching": True},
        "notes": "Solid general-purpose default. Fits 2×RTX 6000 Ada or 1×H100 at TP1.",
    },
    {
        "key": "llama3.1-70b-fp8", "display_name": "Llama 3.1 70B (FP8)",
        "hf_repo": "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8", "params_b": 70,
        "quantization": "fp8", "min_gpu_memory_gb": 48, "recommended_tp": 2, "max_model_len": 32768,
        "tags": ["general", "flagship"], "notes": "FP8 needs Hopper (H100). Do not schedule on A100.",
    },
    {
        "key": "llama3.1-8b", "display_name": "Llama 3.1 8B Instruct",
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct", "params_b": 8,
        "min_gpu_memory_gb": 22, "recommended_tp": 1, "max_model_len": 32768,
        "tags": ["general", "small"], "notes": "Gated repo — needs HF_TOKEN with licence accepted.",
    },
    {
        "key": "qwen2.5-coder-32b", "display_name": "Qwen2.5 Coder 32B",
        "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct", "params_b": 32,
        "min_gpu_memory_gb": 40, "recommended_tp": 2, "max_model_len": 32768,
        "tags": ["code"], "extra_args": {"--enable-prefix-caching": True},
    },
    {
        "key": "mistral-small-24b", "display_name": "Mistral Small 24B",
        "hf_repo": "mistralai/Mistral-Small-24B-Instruct-2501", "params_b": 24,
        "min_gpu_memory_gb": 34, "recommended_tp": 1, "max_model_len": 32768, "tags": ["general"],
    },
    {
        "key": "deepseek-r1-distill-32b", "display_name": "DeepSeek R1 Distill Qwen 32B",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "params_b": 32,
        "min_gpu_memory_gb": 40, "recommended_tp": 2, "max_model_len": 32768, "tags": ["reasoning"],
    },
    {
        "key": "bge-m3-embed", "display_name": "BGE-M3 (embeddings)", "hf_repo": "BAAI/bge-m3",
        "params_b": 0.6, "min_gpu_memory_gb": 8, "recommended_tp": 1,
        "extra_args": {"--task": "embed"}, "tags": ["embedding"],
        "notes": "Park on an RTX workstation; leave the DGX GPUs for generation.",
    },
    {
        "key": "qwen2.5-vl-7b", "display_name": "Qwen2.5-VL 7B (vision)",
        "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct", "params_b": 7,
        "min_gpu_memory_gb": 24, "recommended_tp": 1, "max_model_len": 16384,
        "extra_args": {"--limit-mm-per-prompt": "image=4"}, "tags": ["vision"],
    },
]


async def seed(db: AsyncSession) -> int:
    rows = await db.execute(select(ModelSpec.key))
    have = set(rows.scalars().all())
    added = 0
    for entry in SEED:
        if entry["key"] in have:
            continue
        db.add(ModelSpec(**entry))
        added += 1
    if added:
        await db.commit()
    return added
