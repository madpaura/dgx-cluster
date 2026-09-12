from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DGXCTL_", extra="ignore")

    # --- core ---
    database_url: str = "postgresql+asyncpg://dgxctl:dgxctl@postgres:5432/dgxctl"
    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:5173"

    # --- driver: how we talk to GPU nodes ---
    # "ssh" = real fleet. "sim" = fake fleet, no hardware needed.
    driver: Literal["ssh", "sim"] = "sim"
    ssh_user: str = "root"
    ssh_key_path: str = "/run/secrets/fleet_key"
    ssh_connect_timeout: int = 10

    # --- vLLM defaults ---
    vllm_image: str = "vllm/vllm-openai:latest"
    vllm_port_range_start: int = 8100
    vllm_port_range_end: int = 8399
    hf_cache_dir: str = "/opt/hf-cache"
    hf_token: str = ""
    container_prefix: str = "dgxctl"

    # --- litellm ---
    litellm_base_url: str = "http://litellm:4000"
    litellm_master_key: str = "sk-dgxctl-master"
    litellm_auto_register: bool = True

    # --- auth ---
    # "dev" bypasses OIDC and logs everyone in as admin. Use only locally.
    auth_mode: Literal["dev", "oidc"] = "dev"
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_admin_groups: str = "gpu-admins"
    oidc_deployer_groups: str = "gpu-deployers"
    public_url: str = "http://localhost:8000"

    # --- mcp ---
    # Agent-facing control endpoint at /mcp. It can stop every model in the
    # fleet, so it refuses to mount unauthenticated outside dev mode.
    mcp_enabled: bool = True
    mcp_token: str = ""
    # The MCP transport validates the Host header as DNS-rebinding protection,
    # which rejects every request once you reach the server by its real
    # hostname. Listed hosts are allowed; "*" turns the check off and leans on
    # the bearer token, which is the control that actually matters here.
    mcp_allowed_hosts: str = "*"

    # --- security ---
    # Verify GPU node host keys against this file. Empty disables the check,
    # which is a MITM risk on a shared network.
    ssh_known_hosts: str = ""
    # Set when the dashboard is served over HTTPS, so session cookies are not
    # sent in the clear.
    session_https_only: bool = False
    # What an MCP agent may do: admin, deployer or viewer.
    mcp_role: str = "admin"

    # --- retention ---
    event_retention_days: int = 30
    audit_retention_days: int = 365
    # The fleet summary asks LiteLLM whether it is alive; the dashboard polls
    # the summary every few seconds per open tab, so the answer is cached.
    summary_cache_seconds: int = 15

    # --- polling ---
    gpu_poll_seconds: int = 10
    vllm_poll_seconds: int = 15
    health_poll_seconds: int = 20
    metric_retention_hours: int = 48

    @property
    def mcp_host_list(self) -> list[str]:
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
