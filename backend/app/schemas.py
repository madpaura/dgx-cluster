from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .models import DeployStatus, NodeKind, NodeStatus, Role


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------- identity

class UserOut(ORM):
    id: str
    email: str
    name: str
    role: Role
    team_id: str | None = None


class TeamOut(ORM):
    id: str
    name: str
    max_gpus: int


# ------------------------------------------------------------------- clusters

class ClusterOut(ORM):
    id: str
    name: str
    description: str = ""
    color: str = ""
    sort_order: int = 0
    node_count: int = 0


class ClusterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    color: str = ""
    sort_order: int | None = None
    node_ids: list[str] = []


class ClusterUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    sort_order: int | None = None


class NodeMove(BaseModel):
    node_ids: list[str]


# ---------------------------------------------------------------------- nodes

class GpuOut(ORM):
    id: str
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    ecc_errors: int
    updated_at: datetime | None = None
    # filled by the API layer
    deployment_id: str | None = None
    model_name: str | None = None


class NodeOut(ORM):
    id: str
    name: str
    hostname: str
    ssh_port: int
    kind: NodeKind
    status: NodeStatus
    cluster_id: str | None = None
    cluster_name: str = ""
    labels: dict = {}
    driver_version: str = ""
    cuda_version: str = ""
    docker_version: str = ""
    cpu_count: int = 0
    memory_gb: float = 0
    last_error: str = ""
    last_seen: datetime | None = None
    gpus: list[GpuOut] = []


class NodeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    hostname: str
    ssh_port: int = 22
    ssh_user: str = ""
    kind: NodeKind = NodeKind.dgx
    cluster_id: str | None = None
    labels: dict = {}


class NodeUpdate(BaseModel):
    hostname: str | None = None
    ssh_port: int | None = None
    ssh_user: str | None = None
    kind: NodeKind | None = None
    status: NodeStatus | None = None
    cluster_id: str | None = None
    labels: dict | None = None


# -------------------------------------------------------------------- catalog

class ModelSpecOut(ORM):
    id: str
    key: str
    display_name: str
    hf_repo: str
    revision: str = ""
    params_b: float
    quantization: str = ""
    min_gpu_memory_gb: float
    recommended_tp: int
    max_model_len: int
    extra_args: dict = {}
    vllm_image: str = ""
    tags: list = []
    notes: str = ""


class ModelSpecIn(BaseModel):
    key: str
    display_name: str
    hf_repo: str
    revision: str = ""
    params_b: float = 0
    quantization: str = ""
    min_gpu_memory_gb: float = 0
    recommended_tp: int = 1
    max_model_len: int = 0
    extra_args: dict = {}
    vllm_image: str = ""
    tags: list = []
    notes: str = ""


# ---------------------------------------------------------------- deployments

class TargetIn(BaseModel):
    node_id: str
    gpu_indices: list[int]


class DeployRequest(BaseModel):
    """Either pick a catalog entry (spec_key) or give hf_repo directly."""
    spec_key: str | None = None
    hf_repo: str | None = None
    served_model_name: str | None = None

    # placement: explicit targets, or auto with replicas (+ optional node shortlist)
    targets: list[TargetIn] = []
    replicas: int = 1
    node_ids: list[str] = []

    tensor_parallel_size: int | None = None
    max_model_len: int = 0
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    extra_args: dict = {}
    image: str = ""
    team_id: str | None = None
    dry_run: bool = False


class DeploymentOut(ORM):
    id: str
    served_model_name: str
    hf_repo: str
    node_id: str
    node_name: str = ""
    gpu_indices: list = []
    port: int
    endpoint: str = ""
    status: DeployStatus
    status_reason: str = ""
    image: str = ""
    tensor_parallel_size: int
    container_name: str = ""
    litellm_registered: bool = False
    created_by: str = ""
    team_id: str | None = None
    last_metrics: dict = {}
    vllm_args: dict = {}
    healthy_since: datetime | None = None
    created_at: datetime


class PlacementOut(BaseModel):
    node_id: str
    node_name: str
    gpu_indices: list[int]
    gpu_model: str
    free_gb_per_gpu: float
    note: str = ""


class RejectionOut(BaseModel):
    node_name: str
    reason: str


class PlanOut(BaseModel):
    placements: list[PlacementOut]
    rejections: list[RejectionOut]
    per_gpu_gb: float
    tensor_parallel_size: int
    argv: list[str]


class FindingOut(BaseModel):
    code: str
    severity: str
    title: str
    detail: str
    fix: str
    evidence: str = ""
    actions: list[str] = []


class LogsOut(BaseModel):
    deployment_id: str
    text: str
    findings: list[FindingOut] = []


class BulkAction(BaseModel):
    deployment_ids: list[str]


# --------------------------------------------------------------------- fleet

class FleetSummary(BaseModel):
    nodes_total: int
    nodes_online: int
    nodes_unreachable: int
    gpus_total: int
    gpus_busy: int
    gpus_free: int
    vram_total_gb: float
    vram_used_gb: float
    deployments_healthy: int
    deployments_degraded: int
    deployments_failed: int
    models_served: int
    tokens_per_second: float
    requests_running: int
    requests_waiting: int
    litellm_reachable: bool


class EventOut(ORM):
    id: int
    ts: datetime
    severity: str
    source: str
    source_id: str
    message: str


class AuditOut(ORM):
    id: int
    ts: datetime
    actor: str
    action: str
    target_type: str
    target_id: str
    summary: str
    ok: bool


class SeriesPoint(BaseModel):
    ts: datetime
    values: dict


class SeriesOut(BaseModel):
    scope: str
    scope_id: str
    points: list[SeriesPoint]
