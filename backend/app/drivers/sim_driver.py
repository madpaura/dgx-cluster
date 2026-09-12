"""Simulated fleet.

Lets the whole dashboard run with no GPUs attached: fake DGX and RTX nodes,
containers that take a realistic time to come up, GPU memory that actually gets
consumed, a vLLM-shaped /metrics endpoint, and failure injection (CUDA OOM when
a model does not fit, one node deliberately unreachable).

Flip DGXCTL_DRIVER=ssh at the office and nothing above this layer changes.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .base import ContainerInfo, GpuProbe, LaunchSpec, NodeDriver, NodeFacts

# name -> (gpu model, count, VRAM MB, unreachable?)
SIM_HARDWARE: dict[str, tuple[str, int, int, bool]] = {
    "dgx-01": ("NVIDIA H100 80GB HBM3", 8, 81559, False),
    "dgx-02": ("NVIDIA H100 80GB HBM3", 8, 81559, False),
    "dgx-03": ("NVIDIA A100-SXM4-80GB", 8, 81920, False),
    "dgx-04": ("NVIDIA A100-SXM4-80GB", 8, 81920, True),   # simulates a node that's down
    "rtx-ws-01": ("NVIDIA RTX 6000 Ada Generation", 2, 49140, False),
    "rtx-ws-02": ("NVIDIA RTX 6000 Ada Generation", 2, 49140, False),
    "rtx-ws-03": ("NVIDIA RTX 6000 Ada Generation", 4, 49140, False),
}
DEFAULT_HW = ("NVIDIA RTX 6000 Ada Generation", 2, 49140, False)


@dataclass
class _Container:
    id: str
    name: str
    image: str
    labels: dict[str, str]
    gpu_indices: list[int]
    port: int
    args: list[str]
    started: float
    state: str = "running"
    exit_code: int | None = None
    fail_reason: str = ""
    vram_mb: int = 0
    # fake counters so rates look real
    prompt_tokens: float = 0.0
    gen_tokens: float = 0.0
    requests_done: float = 0.0
    last_tick: float = 0.0


@dataclass
class _Node:
    name: str
    gpu_model: str
    gpu_count: int
    vram_mb: int
    unreachable: bool
    containers: dict[str, _Container] = field(default_factory=dict)


class SimDriver(NodeDriver):
    STARTUP_SECONDS = 25.0  # pull + load weights + warm up

    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        self._rng = random.Random(7)

    def _node(self, node) -> _Node:
        key = node.name
        if key not in self._nodes:
            model, count, vram, down = SIM_HARDWARE.get(key, DEFAULT_HW)
            self._nodes[key] = _Node(key, model, count, vram, down)
        return self._nodes[key]

    # ---------------------------------------------------------------- probe

    async def probe(self, node) -> NodeFacts:
        sim = self._node(node)
        if sim.unreachable:
            return NodeFacts(reachable=False, error="ssh: connect to host: No route to host (simulated)")

        used = self._vram_used(sim)
        gpus = []
        for i in range(sim.gpu_count):
            busy = used.get(i, 0) > 0
            gpus.append(
                GpuProbe(
                    index=i,
                    uuid=f"GPU-{sim.name}-{i:02d}",
                    name=sim.gpu_model,
                    memory_total_mb=sim.vram_mb,
                    memory_used_mb=min(used.get(i, 0) + 520, sim.vram_mb),
                    utilization=self._wobble(62 if busy else 0, 18 if busy else 1),
                    temperature_c=self._wobble(68 if busy else 34, 5),
                    power_draw_w=self._wobble(540 if busy else 78, 60) if "H100" in sim.gpu_model
                    else self._wobble(240 if busy else 30, 25),
                    power_limit_w=700 if "H100" in sim.gpu_model else 300,
                    ecc_errors=0,
                )
            )
        is_dgx = sim.name.startswith("dgx")
        return NodeFacts(
            reachable=True,
            gpus=gpus,
            driver_version="550.90.07",
            cuda_version="12.4",
            docker_version="26.1.4",
            cpu_count=224 if is_dgx else 64,
            memory_gb=2048.0 if is_dgx else 256.0,
        )

    def _vram_used(self, sim: _Node) -> dict[int, int]:
        used: dict[int, int] = {}
        for c in sim.containers.values():
            if c.state != "running":
                continue
            per = c.vram_mb // max(len(c.gpu_indices), 1)
            for i in c.gpu_indices:
                used[i] = used.get(i, 0) + per
        return used

    def _wobble(self, base: float, spread: float) -> float:
        t = time.time()
        return round(max(0.0, base + math.sin(t / 11.0) * spread * 0.5 + self._rng.uniform(-spread, spread) * 0.3), 1)

    # ------------------------------------------------------------ containers

    async def list_containers(self, node, label_filter: str | None = None) -> list[ContainerInfo]:
        sim = self._node(node)
        if sim.unreachable:
            raise RuntimeError("node unreachable (simulated)")
        self._advance(sim)
        out = []
        for c in sim.containers.values():
            if label_filter and not any(
                f"{k}={v}" == label_filter or k == label_filter for k, v in c.labels.items()
            ):
                continue
            out.append(
                ContainerInfo(
                    id=c.id, name=c.name, image=c.image, state=c.state,
                    exit_code=c.exit_code, started_at=str(c.started), labels=c.labels,
                )
            )
        return out

    async def launch(self, node, spec: LaunchSpec) -> str:
        sim = self._node(node)
        if sim.unreachable:
            raise RuntimeError("Cannot connect to the Docker daemon (simulated: node down)")
        if spec.name in sim.containers:
            raise RuntimeError(f"Conflict. The container name /{spec.name} is already in use")

        want_mb = _estimate_vram_mb(spec.args)
        used = self._vram_used(sim)
        free = sim.vram_mb - max((used.get(i, 0) for i in spec.gpu_indices), default=0)
        cid = f"{self._rng.getrandbits(64):016x}" + f"{self._rng.getrandbits(64):016x}"
        c = _Container(
            id=cid, name=spec.name, image=spec.image, labels=dict(spec.labels),
            gpu_indices=list(spec.gpu_indices), port=spec.host_port, args=list(spec.args),
            started=time.time(), vram_mb=want_mb,
        )
        # per-GPU need after tensor parallel split
        per_gpu = want_mb // max(len(spec.gpu_indices), 1)
        if per_gpu > free:
            c.fail_reason = (
                f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate "
                f"{per_gpu / 1024:.1f} GiB. GPU 0 has {free / 1024:.1f} GiB free."
            )
        sim.containers[spec.name] = c
        return cid

    async def stop(self, node, name: str, remove: bool = True) -> None:
        sim = self._node(node)
        c = sim.containers.get(name)
        if c:
            c.state = "exited"
            c.exit_code = 0
            if remove:
                sim.containers.pop(name, None)

    async def logs(self, node, name: str, tail: int = 200) -> str:
        sim = self._node(node)
        c = sim.containers.get(name)
        if not c:
            return f"Error: No such container: {name}"
        age = time.time() - c.started
        model = _arg_value(c.args, "--model") or "unknown"
        tp = _arg_value(c.args, "--tensor-parallel-size") or "1"
        lines = [
            f"INFO {_ts(c.started)} [api_server.py:659] vLLM API server version 0.8.5",
            f"INFO {_ts(c.started)} [config.py:549] This model supports multiple tasks: generate. Defaulting to 'generate'.",
            f"INFO {_ts(c.started + 1)} [llm_engine.py:234] Initializing engine model={model!r} tensor_parallel_size={tp}",
            f"INFO {_ts(c.started + 3)} [weight_utils.py:265] Loading safetensors checkpoint shards: 0% Completed | 0/12",
        ]
        for pct, shard, at in ((25, 3, 7), (50, 6, 12), (75, 9, 16), (100, 12, 20)):
            if age > at:
                lines.append(
                    f"INFO {_ts(c.started + at)} [weight_utils.py:265] Loading safetensors checkpoint shards: "
                    f"{pct}% Completed | {shard}/12"
                )
        if c.fail_reason and age > 18:
            lines += [
                f"ERROR {_ts(c.started + 18)} [engine.py:389] Engine failed to start",
                f"ERROR {_ts(c.started + 18)} {c.fail_reason}",
                "Traceback (most recent call last):",
                '  File "/usr/local/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 281, in __init__',
                "    self.model_executor = executor_class(vllm_config=vllm_config)",
                f"{c.fail_reason.splitlines()[0]}",
            ]
        elif age > self.STARTUP_SECONDS:
            lines += [
                f"INFO {_ts(c.started + 21)} [gpu_executor.py:76] # GPU blocks: 24812, # CPU blocks: 4096",
                f"INFO {_ts(c.started + 23)} [model_runner.py:1450] Graph capturing finished in 4 secs.",
                f"INFO {_ts(c.started + 25)} [api_server.py:1090] Started server process [1]",
                f"INFO {_ts(c.started + 25)} [serving_chat.py:118] Using default chat template",
                f"INFO {_ts(c.started + 26)} Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
            ]
            n = int((age - self.STARTUP_SECONDS) / 9)
            for k in range(max(0, n - 6), n):
                lines.append(
                    f"INFO {_ts(c.started + self.STARTUP_SECONDS + k * 9)} [metrics.py:481] "
                    f"Avg prompt throughput: {self._rng.uniform(300, 2400):.1f} tokens/s, "
                    f"Avg generation throughput: {self._rng.uniform(40, 260):.1f} tokens/s, "
                    f"Running: {self._rng.randint(0, 12)} reqs, Waiting: {self._rng.randint(0, 3)} reqs, "
                    f"GPU KV cache usage: {self._rng.uniform(4, 71):.1f}%"
                )
        return "\n".join(lines[-tail:])

    async def http_get(self, node, port: int, path: str, timeout: float = 5.0) -> tuple[int, str]:
        sim = self._node(node)
        if sim.unreachable:
            return 0, "connection refused (simulated)"
        self._advance(sim)
        c = next((x for x in sim.containers.values() if x.port == port), None)
        if c is None or c.state != "running":
            return 0, "connection refused"
        if time.time() - c.started < self.STARTUP_SECONDS:
            return 0, "connection refused (still loading)"
        if path.startswith("/health"):
            return 200, ""
        if path.startswith("/v1/models"):
            served = _arg_value(c.args, "--served-model-name") or _arg_value(c.args, "--model") or ""
            return 200, '{"object":"list","data":[{"id":"%s","object":"model"}]}' % served
        if path.startswith("/metrics"):
            return 200, self._metrics(c)
        return 404, ""

    def _advance(self, sim: _Node) -> None:
        """Move containers through their lifecycle and accumulate fake counters."""
        now = time.time()
        for c in sim.containers.values():
            if c.state != "running":
                continue
            age = now - c.started
            if c.fail_reason and age > 18:
                c.state, c.exit_code = "exited", 1
                continue
            if age > self.STARTUP_SECONDS:
                # Advance by real elapsed time so the derived tok/s rates are stable.
                last = c.last_tick or (c.started + self.STARTUP_SECONDS)
                dt = max(0.0, min(now - last, 60.0))
                c.last_tick = now
                c.prompt_tokens += self._rng.uniform(600, 2200) * dt
                c.gen_tokens += self._rng.uniform(80, 320) * dt
                c.requests_done += self._rng.uniform(0.3, 1.6) * dt

    def _metrics(self, c: _Container) -> str:
        served = _arg_value(c.args, "--served-model-name") or _arg_value(c.args, "--model") or "model"
        lbl = f'{{model_name="{served}"}}'
        running = self._rng.randint(0, 14)
        waiting = self._rng.randint(0, 4) if running > 8 else 0
        kv = self._rng.uniform(0.05, 0.78)
        ttft = self._rng.uniform(0.05, 0.4)
        e2e = ttft + self._rng.uniform(0.5, 4.0)
        return "\n".join(
            [
                f"vllm:num_requests_running{lbl} {running}.0",
                f"vllm:num_requests_waiting{lbl} {waiting}.0",
                f"vllm:gpu_cache_usage_perc{lbl} {kv:.4f}",
                f"vllm:prompt_tokens_total{lbl} {c.prompt_tokens:.1f}",
                f"vllm:generation_tokens_total{lbl} {c.gen_tokens:.1f}",
                f'vllm:request_success_total{{model_name="{served}",finished_reason="stop"}} {c.requests_done:.1f}',
                f"vllm:time_to_first_token_seconds_sum{lbl} {ttft * max(c.requests_done, 1):.3f}",
                f"vllm:time_to_first_token_seconds_count{lbl} {max(c.requests_done, 1):.1f}",
                f"vllm:e2e_request_latency_seconds_sum{lbl} {e2e * max(c.requests_done, 1):.3f}",
                f"vllm:e2e_request_latency_seconds_count{lbl} {max(c.requests_done, 1):.1f}",
                f"vllm:num_preemptions_total{lbl} {self._rng.randint(0, 3)}.0",
            ]
        )


def _arg_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return None


def _estimate_vram_mb(args: list[str]) -> int:
    """Rough weights+KV footprint, so 'does it fit' behaves believably."""
    model = (_arg_value(args, "--model") or "").lower()
    tp = int(_arg_value(args, "--tensor-parallel-size") or 1)
    params_b = 7.0
    for token, b in (("405b", 405), ("235b", 235), ("123b", 123), ("70b", 70), ("72b", 72),
                     ("32b", 32), ("30b", 30), ("14b", 14), ("8b", 8), ("7b", 7), ("3b", 3), ("1.5b", 1.5)):
        if token in model:
            params_b = b
            break
    bytes_per_param = 1 if ("fp8" in model or "awq" in model or "int4" in model or "gptq" in model) else 2
    weights_mb = params_b * bytes_per_param * 1024
    kv_mb = 6000 * tp
    return int(weights_mb + kv_mb)


def _ts(epoch: float) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(epoch))
