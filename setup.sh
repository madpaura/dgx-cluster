#!/bin/bash
# dgxctl — Docker Setup & Prerequisite Checker
#
# Control plane for a GPU fleet. Runs on a management server with no GPUs of
# its own; it reaches the DGX boxes and workstations over SSH.
#
# Usage: ./setup.sh [check|start|down|restart|logs|status|test|mcp|keygen|clean]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()     { echo -e "  ${GREEN}✔${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()   { echo -e "  ${RED}✘${NC} $1"; }
header() { echo -e "\n${BLUE}── $1 ──${NC}"; }

ERRORS=0
KEY_PATH="./secrets/fleet_key"

# ── Compose helpers ──────────────────────────────────────────────

compose_cmd() {
    if docker compose version &>/dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

compose_files() {
    echo "-f docker-compose.yml"
}

compose() {
    local COMPOSE FILES
    COMPOSE=$(compose_cmd)
    FILES=$(compose_files)
    $COMPOSE $FILES "$@"
}

env_value() {
    [ -f .env ] || return
    grep "^${1}=" .env 2>/dev/null | tail -1 | cut -d= -f2-
}

ensure_env_var() {
    local key="$1"
    local value="$2"
    [ ! -f .env ] && return
    grep -q "^${key}=" .env 2>/dev/null && [ -n "$(env_value "$key")" ] && return
    # replace a present-but-empty key, otherwise append
    if grep -q "^${key}=" .env 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        echo "${key}=${value}" >> .env
    fi
}

rand_hex() { openssl rand -hex 32 2>/dev/null || date +%s%N | sha256sum | cut -c1-64; }

ensure_runtime_env() {
    if [ ! -f .env ]; then
        cp .env.example .env
        ok "Created .env from .env.example"
    fi
    # Secrets the stack refuses to start without. Generated once, then left alone.
    ensure_env_var "DGXCTL_SECRET_KEY"   "$(rand_hex)"
    ensure_env_var "LITELLM_MASTER_KEY"  "sk-$(rand_hex | cut -c1-32)"
    ensure_env_var "LITELLM_SALT_KEY"    "sk-$(rand_hex | cut -c1-32)"
    ensure_env_var "POSTGRES_PASSWORD"   "$(rand_hex | cut -c1-24)"
    ensure_env_var "DGXCTL_MCP_TOKEN"    "$(rand_hex)"
    ensure_env_var "DGXCTL_PORT"         "8080"
    ensure_env_var "LITELLM_PORT"        "4000"
    ensure_env_var "DGXCTL_DRIVER"       "sim"
    ensure_env_var "FLEET_SSH_KEY"       "$KEY_PATH"
}

# ── Key generation ───────────────────────────────────────────────

keygen() {
    header "Fleet SSH key"
    mkdir -p secrets
    if [ -f "$KEY_PATH" ]; then
        ok "Key already exists at $KEY_PATH"
    else
        ssh-keygen -t ed25519 -f "$KEY_PATH" -N '' -C 'dgxctl' >/dev/null
        chmod 600 "$KEY_PATH"
        ok "Generated $KEY_PATH"
    fi
    echo ""
    echo "  Add this public key to ~/.ssh/authorized_keys on every GPU node,"
    echo "  for a user that can run docker:"
    echo ""
    echo -e "  ${GREEN}$(cat "${KEY_PATH}.pub")${NC}"
    echo ""
}

# ── Prerequisites ────────────────────────────────────────────────

check_prereqs() {
    header "Docker"
    if command -v docker &>/dev/null; then
        ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
    else
        fail "docker not found — https://docs.docker.com/engine/install/"
        ERRORS=$((ERRORS + 1))
    fi
    if docker info &>/dev/null; then
        ok "Docker daemon reachable"
    else
        fail "Cannot talk to the Docker daemon (is it running? are you in the docker group?)"
        ERRORS=$((ERRORS + 1))
    fi
    if docker compose version &>/dev/null 2>&1 || command -v docker-compose &>/dev/null; then
        ok "compose available ($(compose_cmd))"
    else
        fail "docker compose not found"
        ERRORS=$((ERRORS + 1))
    fi

    header "Configuration"
    ensure_runtime_env
    for key in DGXCTL_SECRET_KEY LITELLM_MASTER_KEY POSTGRES_PASSWORD DGXCTL_MCP_TOKEN; do
        if [ -n "$(env_value "$key")" ]; then ok "$key set"
        else fail "$key missing from .env"; ERRORS=$((ERRORS + 1)); fi
    done

    local driver
    driver=$(env_value DGXCTL_DRIVER)
    if [ "$driver" = "sim" ]; then
        warn "DGXCTL_DRIVER=sim — a simulated 7-node fleet, no hardware touched"
        echo "      Set DGXCTL_DRIVER=ssh in .env to drive real nodes."
    else
        ok "DGXCTL_DRIVER=$driver — driving real hardware"
        if [ -f "$KEY_PATH" ]; then ok "Fleet SSH key present"
        else fail "No SSH key at $KEY_PATH — run ./setup.sh keygen"; ERRORS=$((ERRORS + 1)); fi
    fi

    header "Ports"
    for entry in "DGXCTL_PORT:dashboard" "LITELLM_PORT:litellm"; do
        local var=${entry%%:*} what=${entry##*:} port
        port=$(env_value "$var")
        if ss -ltn 2>/dev/null | grep -q ":${port} "; then
            warn "Port $port ($what) is already in use — change $var in .env"
        else
            ok "Port $port free ($what)"
        fi
    done

    echo ""
    if [ $ERRORS -eq 0 ]; then
        echo -e "  ${GREEN}Ready.${NC} Run ${GREEN}./setup.sh start${NC}"
    else
        echo -e "  ${RED}$ERRORS problem(s) to fix first.${NC}"
        exit 1
    fi
}

# ── Start ────────────────────────────────────────────────────────

start() {
    header "Building and starting"
    compose up -d --build
    ok "Containers started"

    header "Waiting for the dashboard"
    local port
    port=$(env_value DGXCTL_PORT)
    for _ in $(seq 1 60); do
        if curl -fsS "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            ok "Dashboard healthy"
            show_endpoints
            return 0
        fi
        sleep 2
    done
    fail "Dashboard did not come up within 120s"
    echo "  Check the logs:  ./setup.sh logs api"
    exit 1
}

show_endpoints() {
    local port litellm_port token
    port=$(env_value DGXCTL_PORT)
    litellm_port=$(env_value LITELLM_PORT)
    token=$(env_value DGXCTL_MCP_TOKEN)
    echo ""
    echo -e "  Dashboard   ${GREEN}http://localhost:${port}${NC}"
    echo -e "  LiteLLM     ${GREEN}http://localhost:${litellm_port}/v1${NC}  (what your users call)"
    echo -e "  MCP         ${GREEN}http://localhost:${port}/mcp/${NC}  (for agents — ./setup.sh mcp)"
    echo ""
    [ "$(env_value DGXCTL_DRIVER)" = "sim" ] && \
        echo -e "  ${YELLOW}Simulated fleet.${NC} Set DGXCTL_DRIVER=ssh in .env and restart to go live."
    echo ""
}

# ── Down / restart / clean ───────────────────────────────────────

down() {
    header "Stopping all containers"
    compose down
    ok "All containers stopped"
}

restart() {
    down
    start
}

clean() {
    header "Removing containers AND data"
    warn "This deletes the Postgres volume: node inventory, deployments,"
    warn "clusters, the catalog and the audit log. Models on the GPU nodes"
    warn "keep running; dgxctl just forgets about them."
    read -r -p "  Type 'yes' to continue: " reply
    [ "$reply" = "yes" ] || { echo "  Cancelled."; exit 0; }
    compose down -v
    ok "Containers and volumes removed"
}

# ── Logs / status ────────────────────────────────────────────────

show_logs() {
    local svc="${1:-}"
    if [ -z "$svc" ]; then
        compose logs -f --tail 100
    else
        compose logs -f --tail 100 "$svc"
    fi
}

status() {
    header "Containers"
    compose ps
    header "Fleet"
    local port
    port=$(env_value DGXCTL_PORT)
    if ! curl -fsS "http://localhost:${port}/healthz" >/dev/null 2>&1; then
        fail "Dashboard not responding on port ${port}"
        exit 1
    fi
    local tmp
    tmp=$(mktemp)
    if ! curl -fsS "http://localhost:${port}/api/summary" -o "$tmp"; then
        warn "Could not read the fleet summary"
        rm -f "$tmp"
        return
    fi
    python3 - "$tmp" <<'SUMMARY'
import json, sys

d = json.load(open(sys.argv[1]))
row = "  {:<11} {}".format
print(row("nodes", "{}/{} online".format(d["nodes_online"], d["nodes_total"])
          + (", {} unreachable".format(d["nodes_unreachable"]) if d["nodes_unreachable"] else "")))
print(row("gpus", "{}/{} in use".format(d["gpus_busy"], d["gpus_total"])))
print(row("vram", "{:.0f}/{:.0f} GB".format(d["vram_used_gb"], d["vram_total_gb"])))
print(row("models", "{} serving, {} healthy".format(d["models_served"], d["deployments_healthy"])
          + (", {} failed".format(d["deployments_failed"]) if d["deployments_failed"] else "")))
print(row("throughput", "{:.0f} tok/s, {} running / {} queued".format(
    d["tokens_per_second"], d["requests_running"], d["requests_waiting"])))
print(row("litellm", "reachable" if d["litellm_reachable"] else "UNREACHABLE"))
SUMMARY
    rm -f "$tmp"
    echo ""
}

# ── Test ─────────────────────────────────────────────────────────

run_tests() {
    header "Verification suite (simulated fleet, no hardware)"
    compose run --rm --no-deps \
        -e DGXCTL_DATABASE_URL="sqlite+aiosqlite:///./test.db" \
        -e DGXCTL_DRIVER=sim \
        -e DGXCTL_AUTH_MODE=dev \
        api python -m pytest -q
    ok "All tests passed"
}

# ── MCP ──────────────────────────────────────────────────────────

mcp_details() {
    local port token url
    port=$(env_value DGXCTL_PORT)
    token=$(env_value DGXCTL_MCP_TOKEN)
    url="http://localhost:${port}/mcp/"

    header "MCP endpoint"
    if [ -z "$token" ]; then
        fail "DGXCTL_MCP_TOKEN is not set in .env"
        exit 1
    fi
    echo -e "  URL    ${GREEN}${url}${NC}"
    echo -e "  Token  ${GREEN}${token}${NC}"
    echo ""
    echo "  Claude Code:"
    echo -e "    ${BLUE}claude mcp add --transport http dgxctl ${url} \\
      --header \"Authorization: Bearer ${token}\"${NC}"
    echo ""
    echo "  Or in an mcp.json / claude_desktop_config.json:"
    cat <<JSON
    {
      "mcpServers": {
        "dgxctl": {
          "type": "http",
          "url": "${url}",
          "headers": { "Authorization": "Bearer ${token}" }
        }
      }
    }
JSON
    echo ""
    echo "  Reachable from another machine? Replace localhost with this host's"
    echo "  name and make sure DGXCTL_PORT is open."
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────

case "${1:-check}" in
    check)            check_prereqs ;;
    start|up|deploy)  check_prereqs && start ;;
    down|stop)        down ;;
    restart)          restart ;;
    logs)             show_logs "$2" ;;
    status|ps)        status ;;
    test)             run_tests ;;
    mcp)              mcp_details ;;
    keygen)           keygen ;;
    clean)            clean ;;
    *)
        echo "dgxctl — GPU fleet control plane"
        echo ""
        echo "Usage: ./setup.sh [command]"
        echo ""
        echo "  check      Check prerequisites and seed .env (default)"
        echo "  start      Build and start everything"
        echo "  down       Stop all containers"
        echo "  restart    Stop, then start"
        echo "  status     Container state and a fleet summary"
        echo "  logs       Tail logs: ./setup.sh logs [api|litellm|postgres]"
        echo "  test       Run the verification suite against the simulated fleet"
        echo "  mcp        Print MCP connection details for an agent"
        echo "  keygen     Generate the SSH key to install on the GPU nodes"
        echo "  clean      Stop and DELETE all data (asks first)"
        echo ""
        echo "First run:  ./setup.sh check && ./setup.sh start"
        ;;
esac
