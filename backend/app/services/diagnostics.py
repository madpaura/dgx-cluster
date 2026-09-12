"""Turn failures into next actions.

The point of the dashboard is that nobody should have to read 4000 lines of vLLM
traceback to learn they asked for a 70B model on a 48 GB card. Each rule matches
a log signature and states the fix in the operator's own terms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Finding:
    code: str
    severity: str          # error | warning | info
    title: str
    detail: str
    fix: str
    evidence: str = ""
    actions: list[str] = field(default_factory=list)  # UI action hints


@dataclass
class _Rule:
    code: str
    pattern: re.Pattern
    severity: str
    title: str
    detail: str
    fix: str
    actions: list[str] = field(default_factory=list)


RULES: list[_Rule] = [
    _Rule(
        "cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError|c10::OutOfMemoryError", re.I),
        "error", "Model does not fit on the assigned GPUs",
        "vLLM ran out of VRAM while allocating weights or the KV cache.",
        "Increase tensor-parallel size to split across more GPUs, lower --max-model-len, "
        "drop --gpu-memory-utilization to ~0.85, or pick a quantized (AWQ/FP8) build of this model.",
        ["edit-args", "redeploy-larger"],
    ),
    _Rule(
        "kv_cache_too_small",
        re.compile(r"max seq len.*larger than the maximum number of tokens|"
                   r"increase `?gpu_memory_utilization`?|No available memory for the cache blocks", re.I),
        "error", "KV cache too small for the requested context length",
        "The model loaded but there is not enough leftover VRAM to hold the context window you asked for.",
        "Lower --max-model-len, or raise --gpu-memory-utilization if other processes are not sharing the GPU.",
        ["edit-args"],
    ),
    _Rule(
        "hf_auth", re.compile(r"401 Client Error|403 Client Error|gated repo|awaiting a review of your access", re.I),
        "error", "Hugging Face access denied",
        "The repo is private or gated and the token on the control server cannot read it.",
        "Set HF_TOKEN in the control server .env and accept the model licence on huggingface.co with that account.",
        ["check-config"],
    ),
    _Rule(
        "hf_not_found", re.compile(r"RepositoryNotFoundError|is not a local folder and is not a valid model identifier", re.I),
        "error", "Model repo not found",
        "The Hugging Face repo id in the catalog entry does not resolve.",
        "Fix the repo id (owner/name, case-sensitive) in the model catalog and redeploy.",
        ["edit-catalog"],
    ),
    _Rule(
        "port_conflict", re.compile(r"port is already allocated|address already in use|Conflict.*already in use", re.I),
        "error", "Port or container name already taken on the node",
        "Something is already bound to the port this deployment was assigned, usually an orphaned container.",
        "Run Reconcile on the node to adopt or clean up stray dgxctl containers, then redeploy.",
        ["reconcile-node"],
    ),
    _Rule(
        "nccl", re.compile(r"NCCL error|ncclUnhandledCudaError|ncclSystemError|NCCL WARN", re.I),
        "error", "Multi-GPU communication failed",
        "NCCL could not set up peer-to-peer transport across the selected GPUs.",
        "Check the GPUs are in the same NVLink/NVSwitch domain (prefer an aligned group like 0-3 or 4-7). "
        "On workstations without NVLink try NCCL_P2P_DISABLE=1. Verify --ipc=host and shm size.",
        ["repick-gpus"],
    ),
    _Rule(
        "no_device", re.compile(r"no CUDA-capable device|No supported device|Found no NVIDIA driver", re.I),
        "error", "Container cannot see the GPUs",
        "The NVIDIA container runtime did not pass the devices through.",
        "On the node: check `nvidia-smi` works, nvidia-container-toolkit is installed, "
        "and docker's default runtime can do --gpus.",
        ["node-health"],
    ),
    _Rule(
        "host_oom", re.compile(r"Killed\b|exit code 137|signal 9", re.I),
        "error", "Container killed by the host",
        "The Linux OOM killer or an operator stopped the process — usually host RAM, not VRAM.",
        "Check free RAM on the node during weight loading; reduce concurrent deployments on that box.",
        ["node-health"],
    ),
    _Rule(
        "shm", re.compile(r"insufficient shared memory|bus error.*shm|/dev/shm", re.I),
        "error", "Shared memory too small",
        "Tensor-parallel workers need a large /dev/shm.",
        "Raise the container shm-size (dgxctl requests 16g by default) and keep --ipc=host.",
        ["edit-args"],
    ),
    _Rule(
        "quantization", re.compile(r"Unknown quantization method|quantization method .* is not supported", re.I),
        "error", "Quantization setting not supported",
        "The --quantization flag does not match how this checkpoint was produced.",
        "Clear the quantization override and let vLLM read it from the checkpoint config.",
        ["edit-args"],
    ),
    _Rule(
        "engine_dead", re.compile(r"AsyncEngineDeadError|Engine loop has died|background loop has errored", re.I),
        "error", "vLLM engine died while serving",
        "The engine crashed after startup; the HTTP server may still answer but generation will fail.",
        "Restart the deployment. If it recurs, capture logs and check for a bad request pattern or GPU fault (Xid).",
        ["restart"],
    ),
    _Rule(
        "arch_unsupported", re.compile(r"are not supported for now|not supported by vLLM|Unsupported architecture", re.I),
        "error", "Model architecture unsupported by this vLLM version",
        "This vLLM image is too old for the model.",
        "Pin a newer vLLM image on the catalog entry and redeploy.",
        ["edit-catalog"],
    ),
    _Rule(
        "download_slow", re.compile(r"Loading safetensors checkpoint shards", re.I),
        "info", "Downloading / loading weights",
        "Weights are still being fetched or loaded into VRAM. Large models can take many minutes on first run.",
        "Nothing to do. Pre-warm the shared HF cache to make later deployments near-instant.",
        [],
    ),
    _Rule(
        "preemption", re.compile(r"Sequence group .* is preempted|num_preemptions", re.I),
        "warning", "Requests are being preempted",
        "The KV cache is full, so vLLM is evicting and recomputing sequences. Latency suffers.",
        "Reduce --max-num-seqs, lower --max-model-len, or add a replica of this model.",
        ["scale-out"],
    ),
]


def analyze_logs(text: str, *, max_findings: int = 4) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for rule in RULES:
        m = rule.pattern.search(text)
        if not m or rule.code in seen:
            continue
        seen.add(rule.code)
        line = _line_around(text, m.start())
        findings.append(
            Finding(
                code=rule.code, severity=rule.severity, title=rule.title, detail=rule.detail,
                fix=rule.fix, evidence=line, actions=list(rule.actions),
            )
        )
    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return findings[:max_findings]


def analyze_node(node) -> list[Finding]:
    """Hardware-level problems that are invisible until they bite."""
    out: list[Finding] = []
    if node.status.value == "unreachable":
        out.append(
            Finding(
                "node_unreachable", "error", f"{node.name} is unreachable",
                node.last_error or "SSH did not answer.",
                "Check the box is powered and on the network, and that the control server's key is in "
                "authorized_keys for the configured SSH user.",
                actions=["retry-probe"],
            )
        )
        return out

    for g in node.gpus:
        if g.ecc_errors and g.ecc_errors > 0:
            out.append(
                Finding(
                    "ecc", "error", f"GPU {g.index} reports {g.ecc_errors} uncorrected ECC errors",
                    "Uncorrectable memory errors mean this GPU will corrupt results or fall off the bus.",
                    "Drain the node, run a field diagnostic, and open an RMA. Avoid scheduling here meanwhile.",
                    actions=["drain-node"],
                )
            )
        if g.temperature_c >= 87:
            out.append(
                Finding(
                    "thermal", "warning", f"GPU {g.index} at {g.temperature_c:.0f} °C",
                    "The GPU is at or near its thermal limit and is probably clock-throttling.",
                    "Check airflow, inlet temperature and dust filters. Sustained throttling shows up as falling tokens/s.",
                )
            )
        if g.power_limit_w and g.power_draw_w > g.power_limit_w * 0.98:
            out.append(
                Finding(
                    "power_cap", "info", f"GPU {g.index} is at its power cap",
                    "Sustained operation at the cap limits throughput.",
                    "Expected under full load; only act if tokens/s is below what this model should do.",
                )
            )

    models = {g.name for g in node.gpus}
    if len(models) > 1:
        out.append(
            Finding(
                "mixed_gpus", "warning", "Mixed GPU models in one node",
                f"Found {', '.join(sorted(models))}. Tensor parallel across different GPUs is unreliable.",
                "dgxctl already refuses to group unlike GPUs; keep tensor-parallel groups within one model.",
            )
        )
    return out


def _line_around(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : end if end != -1 else len(text)].strip()[:400]
