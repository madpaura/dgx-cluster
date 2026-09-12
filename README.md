# dgxctl — one screen for the whole GPU fleet

Deploy and manage vLLM models across DGX servers and RTX workstations from a
single control server. Every healthy model is registered with LiteLLM
automatically, so your users get one OpenAI-compatible endpoint and never need
to know which machine is serving what.

```
 control server (no GPUs)                        your fleet
 ┌──────────────────────────────┐              ┌────────────────────┐
 │ React UI   ← websocket       │              │ dgx-01   8× H100   │
 │ FastAPI    ─── asyncssh ─────┼─────────────▶│  docker run vllm…  │
 │ Postgres   (state + metrics) │              ├────────────────────┤
 │ reconciler ─── HTTP /metrics ┼─────────────▶│ rtx-ws-03 4× 6000  │
 │ LiteLLM    ◀── auto-register │              └────────────────────┘
 └──────────────────────────────┘
        ▲
   your users: POST http://dgxctl:4000/v1/chat/completions
```

## What it does

**Fleet** — the whole estate on one page, grouped into **clusters you define**.
Each node is a compact tile: alive or not, how many GPUs are taken, what it is
serving, how fast. Click tiles to select them, drag them between clusters to
reorganise, and hit **Deploy here** on a cluster to confine a model to it.
Per-GPU detail is one click away in the node drawer. Unreachable boxes float to
the top of their cluster.

Clusters are labels, not fences — create, rename, move nodes across and delete
freely. Deleting one releases its nodes to *Unassigned*; nothing stops.

**Deploy** — pick a model, and before anything starts you see which nodes can
host it, **why each other node cannot**, and the exact `docker run` that will
execute. Deploy to one GPU, a tensor-parallel group, or several nodes at once.

**Diagnose** — when a deployment fails, you get a plain-English cause and a fix,
not a 4000-line traceback. CUDA OOM, gated Hugging Face repos, NCCL transport
failures, KV-cache-too-small, host OOM, ECC errors — each has a rule that names
the problem and what to change.

**Proxy** — new models appear in LiteLLM by themselves. Deploy the same model
twice and the two replicas become one load-balanced model group; nothing else
to configure.

## Themes

Three colour schemes ship, switched from the swatches in the top bar. The choice
is stored per browser and applied before first paint, so there is no flash on
reload.

| Theme | Looks like |
|---|---|
| **Nexus** (default) | Cool grey canvas, white cards, indigo primary, teal secondary |
| **Crextio** | Warm cream canvas, ink and yellow, large soft radii |
| **Midnight** | Dark, for a wall display or a night shift |

Every colour, radius and font in the UI is a CSS custom property defined in one
block per theme in `frontend/src/styles.css`. No component names a colour, so
adding your own palette is that one block plus an entry in
`frontend/src/lib/theme.ts`. (The stylesheet uses `color-mix()` to derive badge
and hover shades — needs Chrome/Edge 111+, Firefox 113+, Safari 16.2+.)

## Run it right now, with no GPUs

The simulator gives you seven fake nodes — three DGX (H100/A100), three RTX 6000
Ada workstations, and one deliberately unreachable box — with realistic startup
delays, VRAM accounting, vLLM-shaped metrics and injected failures.

```bash
python3 -m venv .venv
.venv/bin/pip install fastapi "uvicorn[standard]" "sqlalchemy[asyncio]" aiosqlite \
  asyncpg pydantic pydantic-settings asyncssh httpx authlib itsdangerous greenlet
cd frontend && npm install && npm run build && cd ..
make dev-api          # http://localhost:8000
```

Everything works: deploy, watch weights load, see a 70B model OOM on a 48 GB
card and get told why, stop, restart, bulk actions. Nothing touches hardware.

## Deploy it at the office

**1. Prepare each GPU node** (once per DGX / workstation):

```bash
# NVIDIA driver + container toolkit must already work:
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# let the control server in
mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys <<< "ssh-ed25519 AAAA… dgxctl"

# shared model cache on fast local disk (or an NFS mount shared fleet-wide)
sudo mkdir -p /opt/hf-cache && sudo chmod 777 /opt/hf-cache
```

The SSH user must be able to run `docker` without a password prompt.

**2. Bring up the control server:**

```bash
ssh-keygen -t ed25519 -f secrets/fleet_key -N ''   # public key goes on the nodes
cp .env.example .env
# set DGXCTL_SECRET_KEY, LITELLM_MASTER_KEY, POSTGRES_PASSWORD, and DGXCTL_DRIVER=ssh
docker compose up -d --build
```

Open `http://<control-server>:8080`, make a cluster or two (**+ Cluster**), then
**+ Node** for each machine. dgxctl probes each one immediately — you either see
its GPUs or the exact SSH error.

**3. Point your users at the proxy:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://dgxctl:4000/v1", api_key="<a LiteLLM virtual key>")
client.chat.completions.create(model="qwen3-32b", messages=[...])
```

## Ports

| Port | What |
|---|---|
| 8080 | dgxctl UI + API |
| 4000 | LiteLLM — this is the only one your users need |
| 8100–8399 | vLLM containers on the nodes (control server reaches these directly) |

## Roles

`viewer` reads everything · `deployer` deploys and stops within their team ·
`admin` also registers, drains and removes nodes. Set `DGXCTL_AUTH_MODE=oidc`
and map your IdP groups with `DGXCTL_OIDC_ADMIN_GROUPS` /
`DGXCTL_OIDC_DEPLOYER_GROUPS`. Every mutating action lands in the audit log.

## Layout

```
backend/app/
  drivers/        ssh_driver.py (real) and sim_driver.py (fake) behind one interface
  services/
    placement.py    where does this model fit, and why not there
    deployments.py  create / stop / restart, port + container naming
    diagnostics.py  log signature -> cause -> fix
    litellm.py      register, deregister, resync, test
    vllm_metrics.py Prometheus text -> the numbers operators read
  worker.py       inventory, reconcile and metrics loops
frontend/src/
  pages/          Fleet, Models, Proxy, Catalog, Activity
  components/     ClusterSection (grouping + drag-drop), NodeTile,
                  DeployDialog, node and deployment drawers
```

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/RUNBOOK.md](docs/RUNBOOK.md).
