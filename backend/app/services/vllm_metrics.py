"""Parse vLLM's Prometheus endpoint into the handful of numbers an operator
actually looks at when deciding if a model is healthy."""
from __future__ import annotations

import re
import time

SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eENaninf]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten to name -> value, summing across label sets. Good enough: a
    single vLLM container serves one model."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        name = m.group("name")
        out[name] = out.get(name, 0.0) + value
    return out


def summarize(raw: dict[str, float], prev: dict | None) -> dict:
    """Derive gauges + rates. `prev` is the last summary from this deployment."""
    now = time.time()
    prompt_total = raw.get("vllm:prompt_tokens_total", 0.0)
    gen_total = raw.get("vllm:generation_tokens_total", 0.0)
    req_total = raw.get("vllm:request_success_total", 0.0)

    ttft_sum = raw.get("vllm:time_to_first_token_seconds_sum", 0.0)
    ttft_count = raw.get("vllm:time_to_first_token_seconds_count", 0.0)
    e2e_sum = raw.get("vllm:e2e_request_latency_seconds_sum", 0.0)
    e2e_count = raw.get("vllm:e2e_request_latency_seconds_count", 0.0)

    summary = {
        "ts": now,
        "running": raw.get("vllm:num_requests_running", 0.0),
        "waiting": raw.get("vllm:num_requests_waiting", 0.0),
        "kv_cache_pct": round(raw.get("vllm:gpu_cache_usage_perc", 0.0) * 100, 2),
        "preemptions": raw.get("vllm:num_preemptions_total", 0.0),
        "prompt_tokens_total": prompt_total,
        "generation_tokens_total": gen_total,
        "requests_total": req_total,
        "ttft_avg_ms": round(ttft_sum / ttft_count * 1000, 1) if ttft_count else 0.0,
        "e2e_avg_ms": round(e2e_sum / e2e_count * 1000, 1) if e2e_count else 0.0,
        "prompt_tps": 0.0,
        "gen_tps": 0.0,
        "req_per_min": 0.0,
    }

    if prev and prev.get("ts"):
        dt = now - float(prev["ts"])
        if 0.5 < dt < 600:
            summary["prompt_tps"] = round(max(0.0, prompt_total - prev.get("prompt_tokens_total", 0.0)) / dt, 1)
            summary["gen_tps"] = round(max(0.0, gen_total - prev.get("generation_tokens_total", 0.0)) / dt, 1)
            summary["req_per_min"] = round(max(0.0, req_total - prev.get("requests_total", 0.0)) / dt * 60, 2)
    return summary
