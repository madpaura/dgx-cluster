from ..config import settings
from .base import ContainerInfo, GpuProbe, LaunchSpec, NodeDriver, NodeFacts
from .sim_driver import SimDriver
from .ssh_driver import SSHDriver

_driver: NodeDriver | None = None


def get_driver() -> NodeDriver:
    global _driver
    if _driver is None:
        _driver = SSHDriver() if settings.driver == "ssh" else SimDriver()
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


__all__ = [
    "ContainerInfo", "GpuProbe", "LaunchSpec", "NodeDriver", "NodeFacts",
    "get_driver", "close_driver",
]
