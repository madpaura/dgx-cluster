# Architecture

## Why this shape

The control server holds no GPUs and installs nothing on the nodes. It reaches
each machine over SSH and drives the docker CLI there. Three consequences worth
stating, because they drove most of the design:

- **Every action is reproducible by hand.** The dashboard shows the exact
  `docker run` it issued. When something is wrong at 2am you can paste it into
  a terminal and get the same result.
- **Nodes need no agent, no cluster membership, no Kubernetes.** Adding a
  workstation is one `authorized_keys` line. A DGX and an RTX 6000 box are the
  same kind of thing to dgxctl.
- **The control server is not in the data path.** If it dies, every model keeps
  serving and LiteLLM keeps routing. You lose the dashboard, not the cluster.

## Components

| Piece | Responsibility |
|---|---|
| `drivers/` | The only code that touches a node. `SSHDriver` runs docker over asyncssh; `SimDriver` fakes a whole fleet in memory. One interface, so everything above is identical in both modes. |
| `services/placement.py` | Given per-GPU memory need and tensor-parallel size, which GPUs are free and big enough — and why each other node was skipped. |
| `services/deployments.py` | Writes intended state, launches the container, hands off to the reconciler. |
| `services/diagnostics.py` | Log signature → cause → fix. Also hardware-level findings (ECC, thermals, mixed GPUs). |
| `services/litellm.py` | Register/deregister/resync/test against the proxy admin API. |
| `worker.py` | Three idempotent loops that converge observed state onto recorded state. |
| `api/` | Thin FastAPI routers. Auth, audit, serialisation. |
| `frontend/` | React + TypeScript, no chart or UI framework. One websocket pushes change notifications; components refetch. |

## State model

```
Cluster 1─* Node             operator-defined grouping; membership is a label
Node 1─* Gpu                 latest hardware snapshot, overwritten by the poller
Node 1─* Deployment          one vLLM container, pinned to GPU indices
Deployment *─1 ModelSpec     optional catalog entry (sizing + default flags)
MetricSample                 time series for gpu and deployment scopes, trimmed
Event / AuditLog             what happened, and who did it
```

A deployment's `served_model_name` is the public name. Several deployments may
share it; that is how a model scales horizontally, and LiteLLM turns them into
one load-balanced model group without extra configuration.

## Clusters are labels, not boundaries

A cluster groups nodes in the UI and nothing more. It does not partition
scheduling, own quota, or gate access. That is deliberate: the moment grouping
constrains placement, every reorganisation becomes a risky operation, and people
stop reorganising. Deleting a cluster releases its nodes and stops nothing.

Scoping a deploy to a cluster is an explicit, per-deploy choice — the dialog
passes the cluster's node ids as a shortlist to the same placement engine.

## The reconcile loop

dgxctl never assumes a command worked. `create()` records intent and fires
`docker run`; the reconciler decides what is actually true:

```
for each active deployment:
    container missing            → failed ("container not present on node")
    container exited             → failed + last error line from the logs
    container restart-looping    → degraded
    running, /health 200         → healthy → register with LiteLLM
    running, no /health, young   → starting  (weights still loading)
    running, no /health, old     → degraded  (grace period is 15 min)
    node unreachable             → degraded  (the model may well be fine)
```

This is why a node rebooting behind your back shows up correctly instead of the
UI insisting everything is green.

## Placement

`plan()` returns viable placements **best-first** plus a rejection list. The
rejection list is the part that matters operationally: "no capacity" is a dead
end, whereas "dgx-03: only 2 of 8 GPUs free, needs 4" is something you can act
on. It is surfaced live in the deploy dialog before anything runs.

Scoring is **best-fit**: prefer the node with the fewest spare GPUs left over
after placing, so large contiguous blocks stay available for models that
genuinely need them. A small model therefore lands on a workstation rather than
fragmenting a DGX. Groups must be homogeneous (never tensor-parallel across
different GPU models) and an aligned contiguous run (0-3, 4-7) is preferred
because that is how NVLink/NVSwitch domains are laid out.

## Metrics

The poller scrapes each container's `/metrics` and keeps the handful of numbers
operators actually read: generation and prompt tokens/s, requests running and
queued, KV cache utilisation, TTFT and end-to-end latency, preemptions. Rates
are derived from counter deltas between samples. Samples land in Postgres and
are trimmed to `DGXCTL_METRIC_RETENTION_HOURS` (48 by default).

No Prometheus to run. If you already have one, scrape the vLLM containers
directly — dgxctl does not get in the way.

## Simulator

`SimDriver` exists so the product is demonstrable and developable with zero
hardware. It models GPU VRAM consumption per container, a ~25 s startup, vLLM's
log format including the safetensors shard progress, a Prometheus endpoint with
moving counters, and failure injection — a model that does not fit produces a
real `torch.OutOfMemoryError` traceback, which the diagnostics engine then
classifies exactly as it would in production.

One environment variable switches to the real fleet. Nothing above the driver
layer knows the difference.

## What is deliberately not here

- **No scheduler queue.** Deployments are long-lived services, not jobs. If you
  need job scheduling for training, that is Slurm's problem, not this tool's.
- **No autoscaling.** Adding a replica is one click; deciding when to is a
  judgement call that needs your traffic, not a heuristic.
- **No in-app model downloads.** vLLM pulls from the Hugging Face cache on the
  node. Pre-warming that cache is the single biggest win for deploy latency.
