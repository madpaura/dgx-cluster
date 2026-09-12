"""Prometheus scrape -> the handful of numbers an operator reads."""
from __future__ import annotations

import time

from app.services.vllm_metrics import parse_prometheus, summarize

SCRAPE = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen3-32b"} 9.0
vllm:num_requests_waiting{model_name="qwen3-32b"} 2.0
vllm:gpu_cache_usage_perc{model_name="qwen3-32b"} 0.4283
vllm:prompt_tokens_total{model_name="qwen3-32b"} 10000.0
vllm:generation_tokens_total{model_name="qwen3-32b"} 2000.0
vllm:request_success_total{model_name="qwen3-32b",finished_reason="stop"} 40.0
vllm:request_success_total{model_name="qwen3-32b",finished_reason="length"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="qwen3-32b"} 12.5
vllm:time_to_first_token_seconds_count{model_name="qwen3-32b"} 50.0
vllm:e2e_request_latency_seconds_sum{model_name="qwen3-32b"} 150.0
vllm:e2e_request_latency_seconds_count{model_name="qwen3-32b"} 50.0
vllm:num_preemptions_total{model_name="qwen3-32b"} 3.0
"""


def test_comments_and_type_lines_are_ignored():
    raw = parse_prometheus(SCRAPE)
    assert "# HELP" not in raw
    assert raw["vllm:num_requests_running"] == 9.0


def test_same_metric_across_label_sets_is_summed():
    """request_success_total is split by finished_reason; the operator wants
    the total."""
    raw = parse_prometheus(SCRAPE)
    assert raw["vllm:request_success_total"] == 50.0


def test_gauges_and_averages():
    s = summarize(parse_prometheus(SCRAPE), None)
    assert s["running"] == 9.0
    assert s["waiting"] == 2.0
    assert s["kv_cache_pct"] == 42.83
    assert s["ttft_avg_ms"] == 250.0          # 12.5s / 50 requests
    assert s["e2e_avg_ms"] == 3000.0          # 150s / 50 requests
    assert s["preemptions"] == 3.0


def test_rates_need_a_previous_sample():
    s = summarize(parse_prometheus(SCRAPE), None)
    assert s["gen_tps"] == 0.0 and s["prompt_tps"] == 0.0


def test_rates_are_derived_from_counter_deltas():
    prev = {"ts": time.time() - 10, "prompt_tokens_total": 8000.0,
            "generation_tokens_total": 1000.0, "requests_total": 30.0}
    s = summarize(parse_prometheus(SCRAPE), prev)
    assert s["prompt_tps"] == 200.0           # 2000 tokens over 10s
    assert s["gen_tps"] == 100.0              # 1000 tokens over 10s
    assert s["req_per_min"] == 120.0          # 20 requests over 10s


def test_a_counter_reset_never_reports_negative_throughput():
    """vLLM restarting zeroes its counters; that must read as 0, not a
    negative rate."""
    prev = {"ts": time.time() - 10, "prompt_tokens_total": 999999.0,
            "generation_tokens_total": 999999.0, "requests_total": 999999.0}
    s = summarize(parse_prometheus(SCRAPE), prev)
    assert s["prompt_tps"] == 0.0 and s["gen_tps"] == 0.0 and s["req_per_min"] == 0.0


def test_a_stale_previous_sample_is_not_used():
    """A gap longer than the guard means the rate would be meaningless."""
    prev = {"ts": time.time() - 5000, "prompt_tokens_total": 0.0,
            "generation_tokens_total": 0.0, "requests_total": 0.0}
    assert summarize(parse_prometheus(SCRAPE), prev)["gen_tps"] == 0.0


def test_zero_request_count_does_not_divide_by_zero():
    s = summarize(parse_prometheus("vllm:time_to_first_token_seconds_count 0.0"), None)
    assert s["ttft_avg_ms"] == 0.0


def test_garbage_lines_are_skipped_not_fatal():
    raw = parse_prometheus("not a metric\nvllm:num_requests_running 4.0\n{}{}\n")
    assert raw == {"vllm:num_requests_running": 4.0}
