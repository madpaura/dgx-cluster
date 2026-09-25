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

**Deploy** — pick a model, and before anything starts you see what it will
need, which nodes can host it, **why each other node cannot**, and the exact
`docker run` that will execute.

The size is read from the model's own `config.json`, not guessed from its name:
parameter count from the architecture, KV cache from the layer and KV-head
shape at your context length. Settings vLLM would reject — a context longer
than the model supports, a tensor-parallel size that does not divide its
attention heads, FP8 on a pre-Hopper card, a quantization that contradicts the
checkpoint — are refused up front rather than discovered by a container dying
several minutes into a weight download.

**GPUs are pools, not slots.** Several models share a card as long as their
reservations fit, and each is told the matching `--gpu-memory-utilization`, so
an 8B model no longer occupies a whole 80 GB H100. Placement packs onto
partly-used cards first, keeping whole GPUs free for models that need them.

**Diagnose** — when a deployment fails, you get a plain-English cause and a fix,
not a 4000-line traceback. CUDA OOM, gated Hugging Face repos, NCCL transport
failures, KV-cache-too-small, host OOM, ECC errors — each has a rule that names
the problem and what to change.

**Proxy** — new models appear in LiteLLM by themselves. Deploy the same model
twice and the two replicas become one load-balanced model group; nothing else
to configure.

**Catalog** — paste a Hugging Face or GitHub URL and the entry fills itself in.
The model card is read, an entry is drafted, and you get the normal form to
check and correct before anything is saved. Configure which model does the
drafting under **Settings**; by default it is the fleet's own LiteLLM proxy, so
nothing leaves the building and there is no external account to set up.

Two limits are deliberate. Only `huggingface.co` and `github.com` can be
fetched — the control server can reach machines a browser cannot, and an
endpoint that fetches any URL you hand it is an SSRF hole aimed at your own
network. And the LLM never sets the VRAM figure: that stays measured from the
model's `config.json`, because it is what refuses a deploy that would OOM a box.

## Letting an agent run it

The dashboard exposes its own capabilities as an MCP server at `/mcp`
(streamable HTTP, stateless), so an agent can operate the fleet with the same
21 tools the UI is built on — and the same validation, since every tool calls
the very endpoint function the browser calls.

### Connecting

`./setup.sh mcp` prints everything below, filled in with your generated token.

**Claude Code**

```bash
claude mcp add --transport http dgxctl http://localhost:8080/mcp/ \
  --header "Authorization: Bearer $DGXCTL_MCP_TOKEN"
```

**Claude Desktop / any `mcp.json`**

```jsonc
{
  "mcpServers": {
    "dgxctl": {
      "type": "http",
      "url": "http://dgxctl.your-office.lan:8080/mcp/",
      "headers": { "Authorization": "Bearer <DGXCTL_MCP_TOKEN>" }
    }
  }
}
```

Note the **trailing slash** on `/mcp/`, and use the host's real name rather than
`localhost` when connecting from another machine.

**Checking it by hand**

```bash
curl -s http://localhost:8080/mcp/ \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -H "Authorization: Bearer $DGXCTL_MCP_TOKEN" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### The tools

| | |
|---|---|
| **Look** | `fleet_summary` · `list_nodes` · `list_models` · `get_deployment` · `list_clusters` · `list_catalog` · `list_events` · `litellm_status` |
| **Catalog** | `draft_catalog_entry` · `add_catalog_entry` |
| **Diagnose** | `deployment_logs` · `diagnose_node` · `reconcile_node` |
| **Serve** | `plan_deployment` · `deploy_model` · `stop_deployment` · `restart_deployment` |
| **Fleet** | `register_node` · `probe_node` · `drain_node` · `create_cluster` · `move_nodes` · `resync_litellm` |

What makes them useful to an agent rather than just callable:

- **`plan_deployment` is a dry run** that reports which nodes can host a model
  and *why each other one cannot*. `deploy_model` refuses with the same
  reasoning rather than half-placing — `"dgx-01: needs 149 GiB per GPU, largest
  free GPU has 79 GiB"` is something an agent can act on.
- **`deployment_logs` returns a diagnosis, not a log.** Each finding names the
  cause, quotes the evidence, and states the fix, so the agent never has to
  parse a vLLM traceback.
- **`fleet_summary` leads with `needs_attention`** — a list of what is wrong, or
  empty.
- **`draft_catalog_entry` returns a draft and saves nothing**, with
  `"saved": false` and the next step spelled out. Keeping it is a separate
  `add_catalog_entry` call, so a model card full of instructions cannot write
  itself into your catalog.
- **Agent actions are attributable.** They run as `agent@mcp`, so the audit log
  distinguishes what an agent did from what a person did.

### Securing it

Anyone who can reach `/mcp` can stop every model in the fleet.
`./setup.sh check` generates `DGXCTL_MCP_TOKEN` for you; without a token the
endpoint **refuses to mount** unless `auth_mode=dev`. `DGXCTL_MCP_ENABLED=false`
removes it entirely.

The MCP transport also validates the `Host` header as DNS-rebinding protection,
which rejects requests once you reach the server by its real hostname.
`DGXCTL_MCP_ALLOWED_HOSTS` defaults to `*`, turning that check off and relying
on the bearer token; set it to a comma-separated host list to turn it back on.

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

## Run it

Everything is Docker. One script drives it.

```bash
./setup.sh check     # prerequisites, generates .env and its secrets
./setup.sh start     # build and start; prints the URLs when healthy
```

That is a complete, working system: dashboard, API, MCP endpoint, Postgres and
a LiteLLM proxy. It starts against a **simulated fleet** — seven fake nodes
(three DGX, three RTX workstations, one deliberately unreachable), with
realistic startup delays, VRAM accounting, vLLM-shaped metrics and injected
failures. Deploy models, watch weights load, see a 70B refuse to fit on a 48 GB
card and be told why. No hardware is touched.

| Command | |
|---|---|
| `./setup.sh check` | Prerequisites; creates `.env` and generates secrets |
| `./setup.sh start` | Build and start everything |
| `./setup.sh down` | Stop |
| `./setup.sh restart` | Stop, then start |
| `./setup.sh status` | Container state and a fleet summary |
| `./setup.sh logs [api\|litellm\|postgres]` | Tail logs |
| `./setup.sh test` | Run the verification suite **inside the built image** |
| `./setup.sh mcp` | Print MCP connection details for an agent |
| `./setup.sh keygen` | Generate the SSH key to install on the GPU nodes |
| `./setup.sh clean` | Stop and delete all data (asks first) |
| `./setup.sh backup [file]` | Dump both databases, `.env` and `secrets/` into one archive |
| `./setup.sh restore <file>` | Restore that archive — how you move to a new machine |

### Without Docker

```bash
python3 -m venv .venv
.venv/bin/pip install -e backend
cd frontend && npm install && npm run build && cd ..
make dev-api          # http://localhost:8000
```

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
./setup.sh keygen     # prints the public key to install in step 1
./setup.sh check      # writes .env with generated secrets
# edit .env: set DGXCTL_DRIVER=ssh (and DGXCTL_SSH_USER if not root)
./setup.sh start
./setup.sh test       # optional: prove this build works before trusting it
```

**3. Point your users at the proxy:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://dgxctl:4000/v1", api_key="<a LiteLLM virtual key>")
client.chat.completions.create(model="qwen3-32b", messages=[...])
```

## Ports

| Port | What |
|---|---|
| 8080 | dgxctl UI + API, and the MCP endpoint at `/mcp` |
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
  mcp_server.py   the same capabilities as MCP tools for an agent
  worker.py       inventory, reconcile and metrics loops
frontend/src/
  pages/          Fleet, Models, Proxy, Catalog, Activity
  components/     ClusterSection (grouping + drag-drop), NodeTile,
                  DeployDialog, node and deployment drawers
```

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/RUNBOOK.md](docs/RUNBOOK.md).
