"""Database schema.

One deployment == one vLLM container pinned to a set of GPUs on one node.
Several deployments may serve the same public model name; LiteLLM then
load-balances across them as one model group.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    admin = "admin"        # everything, incl. node registration
    deployer = "deployer"  # deploy/stop within own team
    viewer = "viewer"      # read-only


class NodeKind(str, enum.Enum):
    dgx = "dgx"
    workstation = "workstation"


class NodeStatus(str, enum.Enum):
    unknown = "unknown"
    online = "online"
    unreachable = "unreachable"
    draining = "draining"   # no new deployments, existing left alone
    maintenance = "maintenance"


class DeployStatus(str, enum.Enum):
    pending = "pending"
    pulling = "pulling"
    starting = "starting"
    healthy = "healthy"
    degraded = "degraded"
    failed = "failed"
    stopping = "stopping"
    stopped = "stopped"


ACTIVE_STATUSES = (
    DeployStatus.pending, DeployStatus.pulling, DeployStatus.starting,
    DeployStatus.healthy, DeployStatus.degraded,
)


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    # soft quota, enforced at deploy time
    max_gpus: Mapped[int] = mapped_column(Integer, default=0)  # 0 = unlimited
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.viewer)
    team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    team: Mapped[Team | None] = relationship(lazy="selectin")
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Cluster(Base):
    """An operator-defined grouping of nodes.

    Purely organisational — it does not constrain scheduling. It exists because
    a fleet of thirty boxes is unmanageable as one flat list, and how you want
    to carve it up (by room, by owner, by generation) is your call, not ours.
    """
    __tablename__ = "clusters"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    color: Mapped[str] = mapped_column(String(16), default="")  # UI accent
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # No `nodes` relationship on purpose: membership is always read through the
    # node list, which the fleet view needs in full anyway.


class Node(Base):
    __tablename__ = "nodes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hostname: Mapped[str] = mapped_column(String(255))
    ssh_port: Mapped[int] = mapped_column(Integer, default=22)
    ssh_user: Mapped[str] = mapped_column(String(64), default="")  # blank -> global default
    kind: Mapped[NodeKind] = mapped_column(Enum(NodeKind), default=NodeKind.dgx)
    status: Mapped[NodeStatus] = mapped_column(Enum(NodeStatus), default=NodeStatus.unknown)
    # null == unassigned; deleting a cluster releases its nodes rather than them
    cluster_id: Mapped[str | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cluster: Mapped["Cluster | None"] = relationship(lazy="joined")
    # free-form labels used for placement constraints, e.g. {"rack":"r1","net":"ib"}
    labels: Mapped[dict] = mapped_column(JSONType, default=dict)
    driver_version: Mapped[str] = mapped_column(String(32), default="")
    cuda_version: Mapped[str] = mapped_column(String(32), default="")
    docker_version: Mapped[str] = mapped_column(String(64), default="")
    cpu_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_gb: Mapped[float] = mapped_column(Float, default=0.0)
    last_error: Mapped[str] = mapped_column(Text, default="")
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    gpus: Mapped[list[Gpu]] = relationship(
        back_populates="node", cascade="all, delete-orphan", lazy="selectin",
        order_by="Gpu.index",
    )
    # passive_deletes stops the ORM nullifying deployments.node_id (which is
    # NOT NULL) when a node goes; the rows are removed explicitly instead.
    deployments: Mapped[list[Deployment]] = relationship(
        back_populates="node", lazy="selectin", passive_deletes=True
    )

    @property
    def schedulable(self) -> bool:
        return self.status == NodeStatus.online


class Gpu(Base):
    """Latest known state of one physical GPU. Overwritten by the poller."""
    __tablename__ = "gpus"
    __table_args__ = (UniqueConstraint("node_id", "index", name="uq_gpu_node_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    uuid: Mapped[str] = mapped_column(String(64), default="")
    name: Mapped[str] = mapped_column(String(128), default="")
    memory_total_mb: Mapped[int] = mapped_column(Integer, default=0)
    memory_used_mb: Mapped[int] = mapped_column(Integer, default=0)
    utilization: Mapped[float] = mapped_column(Float, default=0.0)
    temperature_c: Mapped[float] = mapped_column(Float, default=0.0)
    power_draw_w: Mapped[float] = mapped_column(Float, default=0.0)
    power_limit_w: Mapped[float] = mapped_column(Float, default=0.0)
    ecc_errors: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    node: Mapped[Node] = relationship(back_populates="gpus")


class ModelSpec(Base):
    """Catalog entry: a model you might want to serve, plus how to serve it."""
    __tablename__ = "model_specs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # short handle
    display_name: Mapped[str] = mapped_column(String(128))
    hf_repo: Mapped[str] = mapped_column(String(255))
    revision: Mapped[str] = mapped_column(String(64), default="")
    params_b: Mapped[float] = mapped_column(Float, default=0.0)
    quantization: Mapped[str] = mapped_column(String(32), default="")  # "", awq, gptq, fp8
    # per-GPU VRAM needed at the recommended TP size
    min_gpu_memory_gb: Mapped[float] = mapped_column(Float, default=0.0)
    recommended_tp: Mapped[int] = mapped_column(Integer, default=1)
    max_model_len: Mapped[int] = mapped_column(Integer, default=0)  # 0 = model default
    # extra vLLM CLI flags, e.g. {"--enable-prefix-caching": true, "--dtype": "bfloat16"}
    extra_args: Mapped[dict] = mapped_column(JSONType, default=dict)
    vllm_image: Mapped[str] = mapped_column(String(255), default="")  # override global
    tags: Mapped[list] = mapped_column(JSONType, default=list)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # public name clients use through LiteLLM; shared by replicas
    served_model_name: Mapped[str] = mapped_column(String(128), index=True)
    spec_id: Mapped[str | None] = mapped_column(ForeignKey("model_specs.id"), nullable=True)
    spec: Mapped[ModelSpec | None] = relationship(lazy="selectin")
    hf_repo: Mapped[str] = mapped_column(String(255))

    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    node: Mapped[Node] = relationship(back_populates="deployments", lazy="selectin")
    gpu_indices: Mapped[list] = mapped_column(JSONType, default=list)
    # VRAM this deployment holds on EACH of its GPUs. A GPU is a pool, not a
    # slot: several models share one card as long as the reservations fit, and
    # this is the ledger that decides whether they do. vLLM is told the matching
    # --gpu-memory-utilization so it reserves exactly this much and no more.
    reserved_mb_per_gpu: Mapped[int] = mapped_column(Integer, default=0)
    port: Mapped[int] = mapped_column(Integer)

    status: Mapped[DeployStatus] = mapped_column(Enum(DeployStatus), default=DeployStatus.pending, index=True)
    status_reason: Mapped[str] = mapped_column(Text, default="")
    container_id: Mapped[str] = mapped_column(String(64), default="")
    container_name: Mapped[str] = mapped_column(String(128), default="")
    image: Mapped[str] = mapped_column(String(255), default="")
    vllm_args: Mapped[dict] = mapped_column(JSONType, default=dict)
    tensor_parallel_size: Mapped[int] = mapped_column(Integer, default=1)

    litellm_model_id: Mapped[str] = mapped_column(String(64), default="")
    litellm_registered: Mapped[bool] = mapped_column(Boolean, default=False)

    team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    team: Mapped[Team | None] = relationship(lazy="selectin")
    created_by: Mapped[str] = mapped_column(String(255), default="")

    # rolling health / metric snapshot, refreshed by the poller
    last_metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    healthy_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    @property
    def endpoint(self) -> str:
        return f"http://{self.node.hostname}:{self.port}"


class MetricSample(Base):
    """Time series, trimmed to settings.metric_retention_hours."""
    __tablename__ = "metric_samples"
    __table_args__ = (Index("ix_metric_scope_ts", "scope", "scope_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(16))       # "gpu" | "deployment"
    scope_id: Mapped[str] = mapped_column(String(36))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    values: Mapped[dict] = mapped_column(JSONType, default=dict)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(255), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str] = mapped_column(String(64), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[dict] = mapped_column(JSONType, default=dict)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)


class Event(Base):
    """Operator-facing feed: what happened on the fleet, newest first."""
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")  # info|warning|error
    source: Mapped[str] = mapped_column(String(32), default="")        # node|deployment|litellm
    source_id: Mapped[str] = mapped_column(String(36), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[dict] = mapped_column(JSONType, default=dict)


class AppSetting(Base):
    """Portal settings an operator edits in the browser, rather than a redeploy.

    One row per setting group, value as JSON. Env vars stay the place for things
    needed before the database is up (its own URL, the driver); this is for what
    an admin should be able to change at runtime — currently the LLM used to
    draft catalog entries.
    """
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    updated_by: Mapped[str] = mapped_column(String(255), default="")
