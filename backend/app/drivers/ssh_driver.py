"""Real fleet driver: asyncssh + docker CLI on the node.

Deliberately uses the docker CLI rather than the Docker HTTP API — nothing to
expose on the nodes, and every command we run is one an operator can paste into
a terminal to reproduce what the dashboard did.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex

import asyncssh
import httpx

from ..config import settings
from .base import ContainerInfo, GpuProbe, LaunchSpec, NodeDriver, NodeFacts

log = logging.getLogger(__name__)

GPU_QUERY = (
    "index,uuid,name,memory.total,memory.used,utilization.gpu,"
    "temperature.gpu,power.draw,power.limit,ecc.errors.uncorrected.volatile.total"
)


class SSHDriver(NodeDriver):
    def __init__(self) -> None:
        self._conns: dict[str, asyncssh.SSHClientConnection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._http = httpx.AsyncClient(timeout=10.0)

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    async def _conn(self, node) -> asyncssh.SSHClientConnection:
        key = node.id
        async with self._lock(key):
            conn = self._conns.get(key)
            if conn is not None and not conn.is_closed():
                return conn
            conn = await asyncio.wait_for(
                asyncssh.connect(
                    node.hostname,
                    port=node.ssh_port,
                    username=node.ssh_user or settings.ssh_user,
                    client_keys=[settings.ssh_key_path],
                    known_hosts=None,  # private fleet; pin via known_hosts file if you prefer
                ),
                timeout=settings.ssh_connect_timeout,
            )
            self._conns[key] = conn
            return conn

    async def run(self, node, cmd: str, timeout: float = 60.0) -> tuple[int, str, str]:
        conn = await self._conn(node)
        try:
            r = await asyncio.wait_for(conn.run(cmd, check=False), timeout=timeout)
        except (asyncssh.Error, OSError):
            self._conns.pop(node.id, None)  # force reconnect next call
            raise
        return r.exit_status or 0, (r.stdout or ""), (r.stderr or "")

    # ---------------------------------------------------------------- probe

    async def probe(self, node) -> NodeFacts:
        try:
            rc, out, err = await self.run(
                node,
                f"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits",
                timeout=20,
            )
            if rc != 0:
                return NodeFacts(reachable=True, error=f"nvidia-smi failed: {err.strip()[:300]}")
            gpus = [g for g in (_parse_gpu_line(line) for line in out.splitlines()) if g]

            rc2, ver, _ = await self.run(
                node, "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1", timeout=15
            )
            rc3, dver, _ = await self.run(node, "docker version --format '{{.Server.Version}}'", timeout=15)
            rc4, misc, _ = await self.run(
                node, "nproc; free -g | awk '/^Mem:/{print $2}'; nvidia-smi | grep -oP 'CUDA Version: \\K[0-9.]+' | head -1",
                timeout=15,
            )
            parts = misc.splitlines()
            return NodeFacts(
                reachable=True,
                gpus=gpus,
                driver_version=ver.strip() if rc2 == 0 else "",
                docker_version=dver.strip() if rc3 == 0 else "",
                cpu_count=int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else 0,
                memory_gb=float(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 0.0,
                cuda_version=parts[2].strip() if len(parts) > 2 else "",
            )
        except Exception as exc:  # unreachable, auth failure, timeout
            return NodeFacts(reachable=False, error=f"{type(exc).__name__}: {exc}"[:300])

    # ------------------------------------------------------------ containers

    async def list_containers(self, node, label_filter: str | None = None) -> list[ContainerInfo]:
        flt = f"--filter label={shlex.quote(label_filter)}" if label_filter else ""
        rc, out, err = await self.run(node, f"docker ps -a {flt} --format '{{{{json .}}}}'", timeout=30)
        if rc != 0:
            raise RuntimeError(f"docker ps failed: {err.strip()[:200]}")
        infos: list[ContainerInfo] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            state = (d.get("State") or "").lower()
            infos.append(
                ContainerInfo(
                    id=d.get("ID", ""),
                    name=d.get("Names", ""),
                    image=d.get("Image", ""),
                    state=state,
                    exit_code=_exit_code(d.get("Status", "")),
                    started_at=d.get("CreatedAt", ""),
                    labels=_parse_labels(d.get("Labels", "")),
                    health=_health(d.get("Status", "")),
                )
            )
        return infos

    async def launch(self, node, spec: LaunchSpec) -> str:
        devices = ",".join(str(i) for i in spec.gpu_indices)
        cmd = [
            "docker", "run", "-d",
            "--name", spec.name,
            "--restart", "unless-stopped",
            "--gpus", f'"device={devices}"',
            "--ipc=host",
            "--shm-size", spec.shm_size,
            "-p", f"{spec.host_port}:8000",
        ]
        for host, cont in spec.volumes.items():
            cmd += ["-v", f"{host}:{cont}"]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        for k, v in spec.labels.items():
            cmd += ["--label", f"{k}={v}"]
        cmd += [spec.image, *spec.args]

        # --gpus needs its inner quotes preserved, so quote every *other* token.
        rendered = " ".join(t if t.startswith('"device=') else shlex.quote(t) for t in cmd)
        rc, out, err = await self.run(node, rendered, timeout=180)
        if rc != 0:
            raise RuntimeError(err.strip()[:500] or out.strip()[:500] or "docker run failed")
        return out.strip()[:64]

    async def stop(self, node, name: str, remove: bool = True) -> None:
        await self.run(node, f"docker stop -t 30 {shlex.quote(name)}", timeout=60)
        if remove:
            await self.run(node, f"docker rm -f {shlex.quote(name)}", timeout=60)

    async def logs(self, node, name: str, tail: int = 200) -> str:
        rc, out, err = await self.run(
            node, f"docker logs --tail {int(tail)} {shlex.quote(name)} 2>&1", timeout=30
        )
        return out or err

    async def http_get(self, node, port: int, path: str, timeout: float = 5.0) -> tuple[int, str]:
        url = f"http://{node.hostname}:{port}{path}"
        try:
            r = await self._http.get(url, timeout=timeout)
            return r.status_code, r.text
        except httpx.HTTPError as exc:
            return 0, str(exc)

    async def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
        await self._http.aclose()


def _parse_gpu_line(line: str) -> GpuProbe | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 9 or not parts[0].isdigit():
        return None

    def num(v: str, cast=float):
        try:
            return cast(v)
        except (ValueError, TypeError):
            return cast(0)

    return GpuProbe(
        index=int(parts[0]),
        uuid=parts[1],
        name=parts[2],
        memory_total_mb=num(parts[3], int),
        memory_used_mb=num(parts[4], int),
        utilization=num(parts[5]),
        temperature_c=num(parts[6]),
        power_draw_w=num(parts[7]),
        power_limit_w=num(parts[8]),
        ecc_errors=num(parts[9], int) if len(parts) > 9 else 0,
    )


def _parse_labels(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _exit_code(status: str) -> int | None:
    # "Exited (137) 2 minutes ago"
    if status.startswith("Exited ("):
        try:
            return int(status.split("(", 1)[1].split(")", 1)[0])
        except (IndexError, ValueError):
            return None
    return None


def _health(status: str) -> str:
    for h in ("healthy", "unhealthy", "starting"):
        if f"({h})" in status:
            return h
    return ""
