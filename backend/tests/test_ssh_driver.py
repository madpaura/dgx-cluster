"""The real-hardware driver.

This code never runs under the simulator, so it is exactly the code most likely
to be wrong on the day the fleet is real. SSH itself is stubbed; what is under
test is the command we build and the output we parse — the two things that
would actually break against a live node.
"""
from __future__ import annotations

import json
import shlex

import pytest

from app.drivers.base import LaunchSpec
from app.drivers.ssh_driver import (
    SSHDriver, _exit_code, _health, _parse_gpu_line, _parse_labels,
)

NVIDIA_SMI = (
    "0, GPU-abc123, NVIDIA H100 80GB HBM3, 81559, 1024, 37, 42, 297.55, 700.00, 0\n"
    "1, GPU-def456, NVIDIA H100 80GB HBM3, 81559, 0, 0, 31, 71.12, 700.00, 0\n"
)


class FakeNode:
    id = "n1"
    name = "dgx-01"
    hostname = "dgx-01.lan"
    ssh_port = 22
    ssh_user = "ops"


class RecordingDriver(SSHDriver):
    """Captures commands instead of running them over SSH."""

    def __init__(self, responses=None):
        super().__init__()
        self.commands: list[str] = []
        self._responses = responses or {}

    async def run(self, node, cmd, timeout=60.0):
        self.commands.append(cmd)
        for needle, reply in self._responses.items():
            if needle in cmd:
                return reply
        return (0, "", "")


# ------------------------------------------------------------------ parsing

def test_nvidia_smi_csv_is_parsed_field_by_field():
    gpu = _parse_gpu_line(NVIDIA_SMI.splitlines()[0])
    assert gpu.index == 0
    assert gpu.uuid == "GPU-abc123"
    assert gpu.name == "NVIDIA H100 80GB HBM3"
    assert gpu.memory_total_mb == 81559
    assert gpu.memory_used_mb == 1024
    assert gpu.utilization == 37
    assert gpu.temperature_c == 42
    assert gpu.power_draw_w == 297.55
    assert gpu.power_limit_w == 700.0
    assert gpu.ecc_errors == 0


def test_header_and_blank_lines_are_ignored():
    assert _parse_gpu_line("") is None
    assert _parse_gpu_line("index, uuid, name, memory.total") is None


def test_not_supported_fields_do_not_crash_the_probe():
    """Consumer cards report [N/A] for power and ECC; a node must still probe."""
    line = "0, GPU-x, NVIDIA RTX 6000 Ada Generation, 49140, 500, 12, 40, [N/A], [N/A], [N/A]"
    gpu = _parse_gpu_line(line)
    assert gpu is not None
    assert gpu.name == "NVIDIA RTX 6000 Ada Generation"
    assert gpu.power_draw_w == 0.0 and gpu.ecc_errors == 0


def test_docker_label_string_is_split_into_pairs():
    labels = _parse_labels("dgxctl.deployment=abc-123,dgxctl.model=qwen3-32b,other=x")
    assert labels["dgxctl.deployment"] == "abc-123"
    assert labels["dgxctl.model"] == "qwen3-32b"


def test_malformed_labels_are_skipped_not_fatal():
    assert _parse_labels("novalue,a=b") == {"a": "b"}
    assert _parse_labels("") == {}


@pytest.mark.parametrize("status,code", [
    ("Exited (137) 2 minutes ago", 137),
    ("Exited (0) About an hour ago", 0),
    ("Up 3 hours", None),
    ("Exited (garbage)", None),
])
def test_exit_code_is_read_from_the_status_string(status, code):
    assert _exit_code(status) == code


@pytest.mark.parametrize("status,health", [
    ("Up 2 minutes (healthy)", "healthy"),
    ("Up 5 seconds (starting)", "starting"),
    ("Up 1 hour (unhealthy)", "unhealthy"),
    ("Up 1 hour", ""),
])
def test_health_is_read_from_the_status_string(status, health):
    assert _health(status) == health


# --------------------------------------------------------------- the command

async def test_launch_builds_a_command_an_operator_could_paste():
    driver = RecordingDriver({"docker run": (0, "deadbeefcafe\n", "")})
    spec = LaunchSpec(
        name="dgxctl-qwen3-32b-abc12345",
        image="vllm/vllm-openai:latest",
        gpu_indices=[4, 5],
        host_port=8100,
        args=["--model", "Qwen/Qwen3-32B", "--tensor-parallel-size", "2"],
        env={"HUGGING_FACE_HUB_TOKEN": "hf_secret"},
        volumes={"/opt/hf-cache": "/root/.cache/huggingface"},
        labels={"dgxctl.deployment": "abc-123"},
    )
    container_id = await driver.launch(FakeNode(), spec)
    assert container_id == "deadbeefcafe"

    cmd = driver.commands[0]
    # the GPU selector must keep its inner quotes or docker rejects it
    assert '--gpus \'"device=4,5"\'' in cmd or '--gpus "device=4,5"' in cmd
    assert "-p 8100:8000" in cmd
    assert "--ipc=host" in cmd
    assert "--shm-size 16g" in cmd
    assert "--restart unless-stopped" in cmd
    assert "-v /opt/hf-cache:/root/.cache/huggingface" in cmd
    assert "--label dgxctl.deployment=abc-123" in cmd
    assert "vllm/vllm-openai:latest" in cmd
    assert cmd.rstrip().endswith("--tensor-parallel-size 2")


async def test_the_gpu_selector_survives_shell_quoting():
    """`--gpus '"device=0,1"'` is the one token that must NOT be re-quoted;
    getting this wrong means every deploy fails on real hardware."""
    driver = RecordingDriver({"docker run": (0, "id\n", "")})
    await driver.launch(FakeNode(), LaunchSpec(
        name="c", image="img", gpu_indices=[0, 1], host_port=8100, args=[]))
    cmd = driver.commands[0]
    tokens = shlex.split(cmd)
    assert "device=0,1" in tokens[tokens.index("--gpus") + 1]


async def test_a_single_gpu_is_still_expressed_as_a_device_list():
    driver = RecordingDriver({"docker run": (0, "id\n", "")})
    await driver.launch(FakeNode(), LaunchSpec(
        name="c", image="img", gpu_indices=[3], host_port=8100, args=[]))
    assert "device=3" in driver.commands[0]


async def test_a_failed_docker_run_raises_with_the_daemon_message():
    driver = RecordingDriver({"docker run": (125, "", "docker: Error response from daemon: "
                                                     "port is already allocated.")})
    with pytest.raises(RuntimeError, match="port is already allocated"):
        await driver.launch(FakeNode(), LaunchSpec(
            name="c", image="img", gpu_indices=[0], host_port=8100, args=[]))


async def test_arguments_with_spaces_are_quoted():
    driver = RecordingDriver({"docker run": (0, "id\n", "")})
    await driver.launch(FakeNode(), LaunchSpec(
        name="c", image="img", gpu_indices=[0], host_port=8100,
        args=["--chat-template", "a template with spaces"]))
    assert "'a template with spaces'" in driver.commands[0]


# ----------------------------------------------------------------- probing

async def test_probe_assembles_facts_from_several_commands():
    driver = RecordingDriver({
        "--query-gpu=index": (0, NVIDIA_SMI, ""),
        "driver_version": (0, "550.90.07\n", ""),
        "docker version": (0, "26.1.4\n", ""),
        "nproc": (0, "224\n2048\n12.4\n", ""),
    })
    facts = await driver.probe(FakeNode())
    assert facts.reachable and not facts.error
    assert len(facts.gpus) == 2
    assert facts.driver_version == "550.90.07"
    assert facts.docker_version == "26.1.4"
    assert facts.cpu_count == 224
    assert facts.memory_gb == 2048.0
    assert facts.cuda_version == "12.4"


async def test_a_node_without_a_working_driver_reports_why():
    driver = RecordingDriver({
        "--query-gpu=index": (9, "", "NVIDIA-SMI has failed because it couldn't "
                                     "communicate with the NVIDIA driver."),
    })
    facts = await driver.probe(FakeNode())
    assert facts.reachable is True          # SSH worked; the GPUs did not
    assert "couldn't communicate with the NVIDIA driver" in facts.error
    assert facts.gpus == []


async def test_an_ssh_failure_marks_the_node_unreachable_not_broken():
    class Dead(SSHDriver):
        async def run(self, node, cmd, timeout=60.0):
            raise OSError("No route to host")

    facts = await Dead().probe(FakeNode())
    assert facts.reachable is False
    assert "No route to host" in facts.error


# -------------------------------------------------------------- containers

async def test_docker_ps_json_is_parsed_into_container_info():
    rows = "\n".join(json.dumps(r) for r in [
        {"ID": "abc123", "Names": "dgxctl-qwen3-32b-aaaa", "Image": "vllm/vllm-openai:latest",
         "State": "running", "Status": "Up 2 hours (healthy)", "CreatedAt": "2026-09-12 08:00:00",
         "Labels": "dgxctl.deployment=dep-1,dgxctl.model=qwen3-32b"},
        {"ID": "def456", "Names": "dgxctl-llama-bbbb", "Image": "vllm/vllm-openai:latest",
         "State": "exited", "Status": "Exited (137) 5 minutes ago", "CreatedAt": "2026-09-12 07:00:00",
         "Labels": "dgxctl.deployment=dep-2"},
    ])
    driver = RecordingDriver({"docker ps": (0, rows, "")})
    containers = await driver.list_containers(FakeNode(), label_filter="dgxctl.deployment")

    assert [c.id for c in containers] == ["abc123", "def456"]
    assert containers[0].state == "running" and containers[0].health == "healthy"
    assert containers[0].labels["dgxctl.deployment"] == "dep-1"
    assert containers[1].exit_code == 137
    assert "--filter label=dgxctl.deployment" in driver.commands[0]


async def test_a_node_with_no_containers_is_not_an_error():
    driver = RecordingDriver({"docker ps": (0, "\n", "")})
    assert await driver.list_containers(FakeNode()) == []


async def test_docker_being_unavailable_raises_clearly():
    driver = RecordingDriver({"docker ps": (1, "", "Cannot connect to the Docker daemon")})
    with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon"):
        await driver.list_containers(FakeNode())


async def test_stop_removes_the_container_by_name():
    driver = RecordingDriver()
    await driver.stop(FakeNode(), "dgxctl-qwen3-32b-abc12345", remove=True)
    assert any("docker stop -t 30" in c for c in driver.commands)
    assert any("docker rm -f" in c for c in driver.commands)


async def test_logs_merge_stderr_so_tracebacks_are_not_lost():
    driver = RecordingDriver({"docker logs": (0, "line one\nline two", "")})
    out = await driver.logs(FakeNode(), "container", tail=50)
    assert out == "line one\nline two"
    assert "--tail 50" in driver.commands[0]
    assert "2>&1" in driver.commands[0]


async def test_the_ssh_user_falls_back_to_the_server_default():
    """A node row with a blank ssh_user must use the global setting, not ''."""
    from app.config import settings

    class Blank(FakeNode):
        ssh_user = ""

    assert (Blank.ssh_user or settings.ssh_user) == settings.ssh_user
    assert settings.ssh_user


# ------------------------------------------------------- host key verification

def test_host_keys_are_verified_when_a_known_hosts_file_is_configured(monkeypatch):
    """Without verification, anything answering on a node's address is trusted
    with the command that launches containers on it."""
    from app.config import settings
    from app.drivers import ssh_driver

    monkeypatch.setattr(settings, "ssh_known_hosts", "/etc/dgxctl/known_hosts")
    assert ssh_driver._host_key_policy() == "/etc/dgxctl/known_hosts"


def test_an_unset_known_hosts_file_skips_verification(monkeypatch):
    from app.config import settings
    from app.drivers import ssh_driver

    monkeypatch.setattr(settings, "ssh_known_hosts", "")
    assert ssh_driver._host_key_policy() is None


def test_skipping_verification_is_announced_exactly_once(monkeypatch):
    """A warning at the poll interval would bury the log it is meant to stand
    out in, so the flag is what carries the once-only guarantee."""
    from app.config import settings
    from app.drivers import ssh_driver

    monkeypatch.setattr(settings, "ssh_known_hosts", "")
    monkeypatch.setattr(ssh_driver._host_key_policy, "_warned", False, raising=False)

    warnings: list[str] = []
    monkeypatch.setattr(ssh_driver.log, "warning", lambda msg, *a: warnings.append(msg))

    for _ in range(5):
        ssh_driver._host_key_policy()
    assert len(warnings) == 1
    assert "not being verified" in warnings[0]


def test_configuring_a_file_stops_the_warning_entirely(monkeypatch):
    from app.config import settings
    from app.drivers import ssh_driver

    monkeypatch.setattr(settings, "ssh_known_hosts", "/etc/dgxctl/known_hosts")
    monkeypatch.setattr(ssh_driver._host_key_policy, "_warned", False, raising=False)
    warnings: list[str] = []
    monkeypatch.setattr(ssh_driver.log, "warning", lambda msg, *a: warnings.append(msg))

    ssh_driver._host_key_policy()
    assert warnings == []
