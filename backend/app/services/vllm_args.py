"""What vLLM will accept, checked before a container is started.

Two failure modes this closes. A flag dgxctl already sets — --tensor-parallel-size,
--gpu-memory-utilization — passed again by hand silently wins or conflicts, and
in the memory case that is how a node ends up oversubscribed and wedged. And a
flag that simply does not exist makes vLLM exit during startup, minutes into a
weight download, with an argparse error nobody sees until they read the log.

The registry is deliberately permissive about unknown flags: vLLM adds arguments
every release, and refusing one dgxctl has not heard of would make the dashboard
the thing holding you back. Unknown is a warning; wrong type or a flag we own is
an error.
"""
from __future__ import annotations

from dataclasses import dataclass

# Flags dgxctl derives itself, and the field that should be used instead.
MANAGED: dict[str, str] = {
    "--model": "the model / hf_repo field",
    "--served-model-name": "the 'served as' field",
    "--host": "dgxctl always binds the container's own interface",
    "--port": "dgxctl allocates the port",
    "--tensor-parallel-size": "the tensor-parallel field",
    "--gpu-memory-utilization": (
        "the memory share is derived from the reservation, so several models "
        "can share a GPU; setting it by hand is how a card gets oversubscribed"
    ),
    "--max-model-len": "the max model len field",
    "--quantization": "the quantization field",
    "--revision": "the catalog entry's revision",
}

# name -> value kind. "flag" takes no value.
KNOWN: dict[str, str] = {
    # scheduling and batching
    "--max-num-seqs": "int",
    "--max-num-batched-tokens": "int",
    "--max-seq-len-to-capture": "int",
    "--scheduling-policy": "str",
    "--enable-chunked-prefill": "flag",
    "--num-scheduler-steps": "int",
    "--preemption-mode": "str",
    # memory and cache
    "--swap-space": "float",
    "--cpu-offload-gb": "float",
    "--block-size": "int",
    "--kv-cache-dtype": "str",
    "--enable-prefix-caching": "flag",
    "--no-enable-prefix-caching": "flag",
    "--num-gpu-blocks-override": "int",
    # parallelism beyond TP
    "--pipeline-parallel-size": "int",
    "--distributed-executor-backend": "str",
    "--enable-expert-parallel": "flag",
    # model loading
    "--dtype": "str",
    "--load-format": "str",
    "--tokenizer": "str",
    "--tokenizer-mode": "str",
    "--trust-remote-code": "flag",
    "--download-dir": "str",
    "--config-format": "str",
    "--code-revision": "str",
    "--tokenizer-revision": "str",
    "--hf-overrides": "str",
    "--rope-scaling": "str",
    "--rope-theta": "float",
    "--seed": "int",
    # task and serving surface
    "--task": "str",
    "--chat-template": "str",
    "--response-role": "str",
    "--max-logprobs": "int",
    "--disable-log-requests": "flag",
    "--disable-log-stats": "flag",
    "--served-model-alias": "str",
    "--api-key": "str",
    "--allowed-origins": "str",
    "--tool-call-parser": "str",
    "--enable-auto-tool-choice": "flag",
    "--guided-decoding-backend": "str",
    # quantisation detail and LoRA
    "--quantization-param-path": "str",
    "--enable-lora": "flag",
    "--max-loras": "int",
    "--max-lora-rank": "int",
    "--lora-modules": "str",
    "--max-cpu-loras": "int",
    # speculative decoding
    "--speculative-model": "str",
    "--num-speculative-tokens": "int",
    "--speculative-draft-tensor-parallel-size": "int",
    # multimodal
    "--limit-mm-per-prompt": "str",
    "--mm-processor-kwargs": "str",
    # observability
    "--otlp-traces-endpoint": "str",
    "--collect-detailed-traces": "str",
}


@dataclass
class ArgIssue:
    severity: str          # "error" blocks a deploy, "warning" does not
    title: str
    detail: str
    fix: str


def normalise(name: str) -> str:
    """Accept `max_num_seqs`, `max-num-seqs` and `--max-num-seqs` alike."""
    name = name.strip()
    if not name.startswith("--"):
        name = "--" + name.lstrip("-")
    return name.replace("_", "-")


def _type_ok(kind: str, value) -> bool:
    if kind == "flag":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False               # a bool where a number belongs is a mistake
    if kind == "int":
        return isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit())
    if kind == "float":
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    return value is not None


def validate(extra_args: dict) -> list[ArgIssue]:
    """Check flags an operator supplied by hand."""
    issues: list[ArgIssue] = []
    if not isinstance(extra_args, dict):
        return [ArgIssue(
            "error", "Extra vLLM flags must be an object",
            f"Got {type(extra_args).__name__}.",
            'Use a JSON object, for example {"--max-num-seqs": 128}.',
        )]

    for raw, value in extra_args.items():
        name = normalise(str(raw))

        if name in MANAGED:
            issues.append(ArgIssue(
                "error", f"{name} is set by dgxctl",
                f"Passing it here conflicts with the value dgxctl computes.",
                f"Remove it and use {MANAGED[name]}.",
            ))
            continue

        kind = KNOWN.get(name)
        if kind is None:
            issues.append(ArgIssue(
                "warning", f"{name} is not a flag dgxctl recognises",
                "It may be valid in your vLLM version, or it may be a typo. "
                "vLLM exits on an unknown argument, several minutes into loading.",
                f"Check it against `vllm serve --help` for {'' or 'your image'}. "
                f"Leave it if you are sure.",
            ))
            continue

        if not _type_ok(kind, value):
            expected = {"flag": "true or false", "int": "a whole number",
                        "float": "a number"}.get(kind, "a string")
            issues.append(ArgIssue(
                "error", f"{name} expects {expected}",
                f"Got {value!r}.",
                f"Give {name} {expected}.",
            ))

    return issues
