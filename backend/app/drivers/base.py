"""Node driver interface.

Everything the control server does to a GPU box goes through this. Two
implementations exist: SSHDriver (real fleet) and SimDriver (fake fleet, so the
whole product runs and demos with no hardware attached).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GpuProbe:
    index: int
    uuid: str = ""
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    utilization: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    ecc_errors: int = 0


@dataclass
class NodeFacts:
    reachable: bool
    error: str = ""
    gpus: list[GpuProbe] = field(default_factory=list)
    driver_version: str = ""
    cuda_version: str = ""
    docker_version: str = ""
    cpu_count: int = 0
    memory_gb: float = 0.0


@dataclass
class ContainerInfo:
    id: str
    name: str
    image: str
    state: str          # running | exited | created | restarting
    exit_code: int | None
    started_at: str
    labels: dict[str, str] = field(default_factory=dict)
    health: str = ""


@dataclass
class LaunchSpec:
    name: str
    image: str
    gpu_indices: list[int]
    host_port: int
    args: list[str]                                   # vLLM CLI args
    env: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)   # host -> container
    labels: dict[str, str] = field(default_factory=dict)
    shm_size: str = "16g"


class NodeDriver(ABC):
    """`node` is the ORM Node; drivers only read .hostname/.ssh_* from it."""

    @abstractmethod
    async def probe(self, node) -> NodeFacts: ...

    @abstractmethod
    async def list_containers(self, node, label_filter: str | None = None) -> list[ContainerInfo]: ...

    @abstractmethod
    async def launch(self, node, spec: LaunchSpec) -> str:
        """Start the container detached and return its id."""

    @abstractmethod
    async def stop(self, node, name: str, remove: bool = True) -> None: ...

    @abstractmethod
    async def logs(self, node, name: str, tail: int = 200) -> str: ...

    @abstractmethod
    async def image_present(self, node, image: str) -> bool:
        """Is this image already on the node?

        Worth asking separately from pulling it: the first deployment of a given
        vLLM image moves several gigabytes, and an operator watching a model sit
        at "starting" deserves to know that is what is happening.
        """

    @abstractmethod
    async def pull_image(self, node, image: str) -> None:
        """Fetch an image, raising with the registry's own message on failure."""

    async def install_authorized_key(self, node, password: str, public_key: str) -> None:
        """Append the control server's public key to the node's authorized_keys,
        authenticating with a password this once.

        The password is used and discarded: it is never stored, logged, or put in
        the audit trail. What is recorded is that a key was installed.
        """
        raise NotImplementedError

    @abstractmethod
    async def http_get(self, node, port: int, path: str, timeout: float = 5.0) -> tuple[int, str]:
        """HTTP GET against a container port. Goes direct if the control server can
        reach the node's ports, which is the normal case on a private fleet."""

    async def close(self) -> None:
        return None
