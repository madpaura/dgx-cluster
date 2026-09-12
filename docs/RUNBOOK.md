# Runbook

## A model will not start

Open it from the Fleet page or Models page and read the **Overview** tab — the
diagnostics panel names the cause and the fix. The common ones:

| What you see | What it means | What to do |
|---|---|---|
| `CUDA out of memory` | Weights + KV cache do not fit on the assigned GPUs | Raise tensor-parallel size, lower `--max-model-len`, drop `--gpu-memory-utilization` to ~0.85, or use an AWQ/FP8 build |
| `max seq len … larger than KV cache` | Model loaded, context window does not fit | Lower `--max-model-len` |
| `401/403` or `gated repo` | Hugging Face token missing or licence not accepted | Set `DGXCTL_HF_TOKEN`, accept the licence with that account |
| `NCCL error` | Multi-GPU transport failed | Use an aligned GPU group (0-3, 4-7); on workstations without NVLink try `NCCL_P2P_DISABLE=1` |
| `port is already allocated` | Orphaned container holding the port | Node drawer → **Reconcile containers**, then redeploy |
| `Exited (137)` | Host OOM killer, not VRAM | Check host RAM during weight loading; fewer concurrent deployments on that box |

Stuck in `starting` for a long time is usually normal on first deploy — a 70B
model is ~140 GB of downloads. The **Logs** tab shows shard progress.

## A node went unreachable

The card turns red and floats to the top with the SSH error. Existing
deployments are marked `degraded`, not failed — they are probably still serving.

1. Can you `ssh` to it manually from the control server?
2. Node drawer → **Probe now** to retry immediately.
3. If the box rebooted, containers with `--restart unless-stopped` come back by
   themselves; the reconciler will return them to `healthy`. If they do not,
   **Reconcile containers** shows what is actually running, then redeploy.

## Taking a node out of service

Node drawer → **Drain**. No new deployments land there; what is running keeps
running. Move each model with **+ replica** on the Models page (a new replica
lands elsewhere, LiteLLM starts routing to it), then stop the old one. Zero
downtime, because both are in the model group during the overlap.

**Undrain** to put it back.

## Users report errors but the models look healthy

Check the **Proxy** page first.

- Proxy unreachable → the litellm container is down; models are still serving
  directly on their node ports.
- Model shows "not routed" → it is healthy but not registered. Click **Resync
  with fleet**; that reconciles LiteLLM against reality in both directions.
- Everything looks right → hit **Test** on the model group. It round-trips a
  real completion through the proxy and tells you exactly where it breaks.

## Queue depth is climbing

On the Models page, **Queued** going above zero persistently means the model is
saturated. In order of preference:

1. **+ replica** — the fastest fix, and LiteLLM load-balances immediately.
2. Lower `--max-model-len` so more sequences fit in the KV cache.
3. Check the Metrics tab for preemptions. Rising preemptions with high KV cache
   usage means requests are being evicted and recomputed; a replica fixes it.

## GPU reporting ECC errors

Treat it as hardware failure. Drain the node, stop deployments using that GPU,
run a field diagnostic, open an RMA. dgxctl will keep scheduling onto a GPU that
merely reports errors — it does not know the card is dying, only that you have
not drained it.

## Backup

Everything durable is in Postgres: node inventory, deployments, catalog, audit
log, metric samples.

```bash
docker compose exec postgres pg_dump -U dgxctl dgxctl | gzip > dgxctl-$(date +%F).sql.gz
```

The vLLM containers themselves are disposable — dgxctl can recreate any
deployment from its recorded arguments.

## Upgrading vLLM

Set `DGXCTL_VLLM_IMAGE` to the new tag (or override it per model in the
Catalog), then restart deployments one at a time. With a replica of each model
on another node, the upgrade is invisible to callers.
