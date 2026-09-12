export type NodeStatus = "unknown" | "online" | "unreachable" | "draining" | "maintenance";
export type DeployStatus =
  | "pending" | "pulling" | "starting" | "healthy" | "degraded" | "failed" | "stopping" | "stopped";

export interface Gpu {
  id: string;
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  utilization: number;
  temperature_c: number;
  power_draw_w: number;
  power_limit_w: number;
  ecc_errors: number;
  deployment_id: string | null;
  model_name: string | null;
}

export interface Cluster {
  id: string;
  name: string;
  description: string;
  color: string;
  sort_order: number;
  node_count: number;
}

export interface Node {
  id: string;
  name: string;
  hostname: string;
  ssh_port: number;
  kind: "dgx" | "workstation";
  status: NodeStatus;
  cluster_id: string | null;
  cluster_name: string;
  labels: Record<string, string>;
  driver_version: string;
  cuda_version: string;
  docker_version: string;
  cpu_count: number;
  memory_gb: number;
  last_error: string;
  last_seen: string | null;
  gpus: Gpu[];
}

export interface Metrics {
  running?: number;
  waiting?: number;
  kv_cache_pct?: number;
  gen_tps?: number;
  prompt_tps?: number;
  ttft_avg_ms?: number;
  e2e_avg_ms?: number;
  req_per_min?: number;
  preemptions?: number;
}

export interface Deployment {
  id: string;
  served_model_name: string;
  hf_repo: string;
  node_id: string;
  node_name: string;
  gpu_indices: number[];
  port: number;
  endpoint: string;
  status: DeployStatus;
  status_reason: string;
  image: string;
  tensor_parallel_size: number;
  container_name: string;
  litellm_registered: boolean;
  created_by: string;
  team_id: string | null;
  last_metrics: Metrics;
  vllm_args: { argv?: string[] };
  healthy_since: string | null;
  created_at: string;
}

export interface ModelSpec {
  id: string;
  key: string;
  display_name: string;
  hf_repo: string;
  revision: string;
  params_b: number;
  quantization: string;
  min_gpu_memory_gb: number;
  recommended_tp: number;
  max_model_len: number;
  extra_args: Record<string, unknown>;
  vllm_image: string;
  tags: string[];
  notes: string;
}

export interface Summary {
  nodes_total: number;
  nodes_online: number;
  nodes_unreachable: number;
  gpus_total: number;
  gpus_busy: number;
  gpus_free: number;
  vram_total_gb: number;
  vram_used_gb: number;
  deployments_healthy: number;
  deployments_degraded: number;
  deployments_failed: number;
  models_served: number;
  tokens_per_second: number;
  requests_running: number;
  requests_waiting: number;
  litellm_reachable: boolean;
}

export interface Placement {
  node_id: string;
  node_name: string;
  gpu_indices: number[];
  gpu_model: string;
  free_gb_per_gpu: number;
  note: string;
}

export interface Plan {
  placements: Placement[];
  rejections: { node_name: string; reason: string }[];
  per_gpu_gb: number;
  tensor_parallel_size: number;
  argv: string[];
}

export interface Finding {
  code: string;
  severity: "error" | "warning" | "info";
  title: string;
  detail: string;
  fix: string;
  evidence: string;
  actions: string[];
}

export interface EventRow {
  id: number;
  ts: string;
  severity: string;
  source: string;
  source_id: string;
  message: string;
}

export interface AuditRow {
  id: number;
  ts: string;
  actor: string;
  action: string;
  target_type: string;
  target_id: string;
  summary: string;
  ok: boolean;
}

export interface LiteLLMMember {
  id: string;
  api_base: string;
  upstream_model: string;
  managed_by_dgxctl: boolean;
  deployment_id: string;
}

export interface LiteLLMStatus {
  base_url: string;
  auto_register: boolean;
  reachable: boolean;
  detail: unknown;
  groups: { model_name: string; members: LiteLLMMember[] }[];
}

export interface Me {
  id: string;
  email: string;
  name: string;
  role: "admin" | "deployer" | "viewer";
  team_id: string | null;
}

export interface RuntimeConfig {
  driver: string;
  simulated: boolean;
  auth_mode: string;
  vllm_image: string;
  litellm_base_url: string;
  poll_seconds: number;
}
