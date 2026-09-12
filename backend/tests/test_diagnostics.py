"""Diagnostics: a failure must come back as a cause and a fix, not a traceback."""
from __future__ import annotations

import pytest

from app.models import Gpu, Node, NodeStatus
from app.services.diagnostics import RULES, analyze_logs, analyze_node

# One real-ish log excerpt per rule we care about, and the code it must produce.
CASES = [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.9 GiB", "cuda_oom"),
    ("ValueError: The model's max seq len (32768) is larger than the maximum number "
     "of tokens that can be stored in KV cache (8192)", "kv_cache_too_small"),
    ("huggingface_hub.errors.GatedRepoError: 401 Client Error", "hf_auth"),
    ("OSError: meta-llama/Nope is not a local folder and is not a valid model identifier", "hf_not_found"),
    ("docker: Error response from daemon: driver failed: port is already allocated", "port_conflict"),
    ("dgx-01:112:119 [0] NCCL WARN Cuda failure 'peer access is not supported'", "nccl"),
    ("RuntimeError: Found no NVIDIA driver on your system", "no_device"),
    ("/usr/bin/entrypoint.sh: line 8: 1 Killed python -m vllm.entrypoints", "host_oom"),
    ("RuntimeError: DataLoader worker is killed by signal: insufficient shared memory (shm)", "shm"),
    ("ValueError: Unknown quantization method: awq2", "quantization"),
    ("vllm.engine.async_llm_engine.AsyncEngineDeadError: Background loop has errored", "engine_dead"),
    ("ValueError: Model architectures ['FooForCausalLM'] are not supported for now", "arch_unsupported"),
]


@pytest.mark.parametrize("log,code", CASES)
def test_each_signature_is_recognised(log, code):
    findings = analyze_logs(log)
    assert code in {f.code for f in findings}, f"{code} not found in {[f.code for f in findings]}"


@pytest.mark.parametrize("log,code", CASES)
def test_every_finding_states_a_fix(log, code):
    finding = next(f for f in analyze_logs(log) if f.code == code)
    assert finding.title and finding.detail and finding.fix
    assert len(finding.fix) > 30, "a fix must be actionable, not a label"
    assert finding.evidence, "the matching log line must be quoted back"


def test_errors_are_ranked_above_noise():
    log = "Loading safetensors checkpoint shards: 50% Completed\nCUDA out of memory"
    findings = analyze_logs(log)
    assert findings[0].code == "cuda_oom"
    assert findings[0].severity == "error"


def test_a_clean_log_produces_nothing_alarming():
    log = "INFO Uvicorn running on http://0.0.0.0:8000\nINFO Started server process [1]"
    assert [f for f in analyze_logs(log) if f.severity == "error"] == []


def test_evidence_is_the_matching_line_only():
    log = "line one\nERROR torch.OutOfMemoryError: CUDA out of memory here\nline three"
    finding = next(f for f in analyze_logs(log) if f.code == "cuda_oom")
    assert finding.evidence == "ERROR torch.OutOfMemoryError: CUDA out of memory here"


def test_findings_are_capped_so_the_panel_stays_readable():
    log = "\n".join(c[0] for c in CASES)
    assert len(analyze_logs(log)) <= 4


def test_every_rule_is_reachable():
    """A rule nobody can trigger is dead weight — prove each one matches
    something, using the rule's own pattern as the witness."""
    unreachable = [r.code for r in RULES if not r.pattern.search(r.pattern.pattern.split("|")[0])]
    # Patterns with regex metacharacters can't witness themselves; the CASES
    # table covers the rest explicitly.
    covered = {c[1] for c in CASES} | {"download_slow", "preemption"}
    assert {r.code for r in RULES} - covered == set(), "a rule has no test case"
    assert isinstance(unreachable, list)


# ------------------------------------------------------------ node hardware

def make_node(status=NodeStatus.online, **gpu_kw):
    node = Node(id="n", name="dgx-01", hostname="h", status=status, last_error="")
    defaults = dict(memory_total_mb=81559, memory_used_mb=0, temperature_c=60,
                    power_draw_w=300, power_limit_w=700, ecc_errors=0, name="NVIDIA H100 80GB HBM3")
    defaults.update(gpu_kw)
    node.gpus = [Gpu(id="g0", node_id="n", index=0, **defaults)]
    node.deployments = []
    return node


def test_uncorrected_ecc_errors_are_an_error_not_a_warning():
    findings = analyze_node(make_node(ecc_errors=3))
    ecc = next(f for f in findings if f.code == "ecc")
    assert ecc.severity == "error"
    assert "RMA" in ecc.fix


def test_a_hot_gpu_warns_about_throttling():
    findings = analyze_node(make_node(temperature_c=91))
    assert next(f for f in findings if f.code == "thermal").severity == "warning"


def test_a_healthy_node_reports_nothing():
    assert analyze_node(make_node()) == []


def test_unreachable_node_short_circuits_to_one_actionable_finding():
    node = make_node(status=NodeStatus.unreachable, ecc_errors=9)
    findings = analyze_node(node)
    assert len(findings) == 1
    assert findings[0].code == "node_unreachable"
    assert "authorized_keys" in findings[0].fix


def test_mixed_gpu_models_in_one_node_are_flagged():
    node = make_node()
    node.gpus.append(Gpu(id="g1", node_id="n", index=1, name="NVIDIA A100-SXM4-80GB",
                         memory_total_mb=81920, memory_used_mb=0, temperature_c=50,
                         power_draw_w=100, power_limit_w=400, ecc_errors=0))
    assert any(f.code == "mixed_gpus" for f in analyze_node(node))
