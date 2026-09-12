import { useEffect, useState } from "react";
import { api, ApiError, usePolled } from "../lib/api";
import { DEPLOY_TONE, ago, fmtNum } from "../lib/format";
import type { Deployment, Finding } from "../types";
import { Findings } from "./Findings";
import { Confirm, Pill, Sparkline, useToast } from "./ui";

interface Series {
  points: { ts: string; values: Record<string, number> }[];
}

const CHARTS: { key: string; label: string; unit: string; color: string }[] = [
  { key: "gen_tps", label: "Generation throughput", unit: "tok/s", color: "var(--ok)" },
  { key: "prompt_tps", label: "Prompt throughput", unit: "tok/s", color: "var(--accent)" },
  { key: "running", label: "Requests running", unit: "", color: "var(--busy)" },
  { key: "waiting", label: "Queue depth", unit: "", color: "var(--warn)" },
  { key: "kv_cache_pct", label: "KV cache used", unit: "%", color: "var(--busy)" },
  { key: "ttft_ms", label: "Time to first token", unit: "ms", color: "var(--accent)" },
];

export function DeploymentDrawer({
  deploymentId,
  onClose,
  onChanged,
}: {
  deploymentId: string;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [tab, setTab] = useState<"overview" | "metrics" | "logs" | "config">("overview");
  const [confirm, setConfirm] = useState<null | "stop" | "restart">(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const { data: dep, refresh } = usePolled<Deployment>(`/api/deployments/${deploymentId}`, 5000);
  const { data: logs } = usePolled<{ text: string; findings: Finding[] }>(
    `/api/deployments/${deploymentId}/logs?tail=400`,
    tab === "logs" || tab === "overview" ? 8000 : 60000
  );
  const { data: series } = usePolled<Series>(
    `/api/deployments/${deploymentId}/series?minutes=60`,
    tab === "metrics" ? 10000 : 60000
  );

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  if (!dep) return null;
  const m = dep.last_metrics ?? {};
  const points = series?.points ?? [];

  async function act(kind: "stop" | "restart") {
    setBusy(true);
    try {
      await api.post(`/api/deployments/${deploymentId}/${kind}`);
      toast(kind === "stop" ? "Stopped" : "Restarting — it will reload weights");
      onChanged();
      refresh();
      if (kind === "stop") onClose();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
      setConfirm(null);
    }
  }

  async function testThroughProxy() {
    setBusy(true);
    try {
      const r = await api.post<{ ok: boolean; reply?: string; error?: string }>("/api/litellm/test", {
        model_name: dep!.served_model_name,
      });
      toast(r.ok ? `LiteLLM answered: "${(r.reply ?? "").slice(0, 60)}"` : `LiteLLM: ${r.error}`, !r.ok);
    } catch (e) {
      toast(String(e), true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="backdrop" style={{ padding: 0 }} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
        <aside className="drawer" onMouseDown={(e) => e.stopPropagation()}>
          <header>
            <div>
              <div style={{ fontWeight: 650, fontSize: 15 }}>{dep.served_model_name}</div>
              <div className="hint mono">
                {dep.node_name} · GPU {dep.gpu_indices.join(",")} · TP{dep.tensor_parallel_size} · :{dep.port}
              </div>
            </div>
            <div className="right flex">
              <Pill tone={DEPLOY_TONE[dep.status]}>{dep.status}</Pill>
              <button className="btn sm" disabled={busy} onClick={() => setConfirm("restart")}>
                Restart
              </button>
              <button className="btn sm danger" disabled={busy} onClick={() => setConfirm("stop")}>
                Stop
              </button>
              <button className="btn ghost sm" onClick={onClose}>
                ✕
              </button>
            </div>
          </header>

          <div className="tabs">
            {(["overview", "metrics", "logs", "config"] as const).map((t) => (
              <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                {t[0].toUpperCase() + t.slice(1)}
                {t === "logs" && logs?.findings.some((f) => f.severity === "error") ? " ⚠" : ""}
              </button>
            ))}
          </div>

          <div className="body">
            {tab === "overview" && (
              <>
                {dep.status !== "healthy" && dep.status_reason && (
                  <div className={`finding ${dep.status === "failed" ? "error" : "warning"}`}>
                    <div className="t">{dep.status === "failed" ? "This deployment failed" : "Not serving yet"}</div>
                    <div className="d">{dep.status_reason}</div>
                  </div>
                )}
                {logs?.findings?.length ? <Findings items={logs.findings} /> : null}

                <div className="strip" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                  <Stat k="Generating" v={fmtNum(m.gen_tps, 1)} unit="tok/s" />
                  <Stat k="Prompt" v={fmtNum(m.prompt_tps, 1)} unit="tok/s" />
                  <Stat k="In flight" v={fmtNum(m.running)} />
                  <Stat k="Queued" v={fmtNum(m.waiting)} tone={(m.waiting ?? 0) > 0 ? "warn" : undefined} />
                  <Stat k="TTFT" v={fmtNum(m.ttft_avg_ms)} unit="ms" />
                  <Stat
                    k="KV cache"
                    v={fmtNum(m.kv_cache_pct, 1)}
                    unit="%"
                    tone={(m.kv_cache_pct ?? 0) > 90 ? "warn" : undefined}
                  />
                </div>

                <dl className="kv mt">
                  <dt>Endpoint</dt>
                  <dd>{dep.endpoint}/v1</dd>
                  <dt>Model repo</dt>
                  <dd>{dep.hf_repo}</dd>
                  <dt>In LiteLLM</dt>
                  <dd>
                    {dep.litellm_registered ? (
                      <span style={{ color: "var(--ok)" }}>
                        yes — clients call model <code>{dep.served_model_name}</code>
                      </span>
                    ) : (
                      <span style={{ color: "var(--warn)" }}>not registered</span>
                    )}
                  </dd>
                  <dt>Healthy since</dt>
                  <dd>{dep.healthy_since ? ago(dep.healthy_since) : "–"}</dd>
                  <dt>Deployed by</dt>
                  <dd>
                    {dep.created_by} · {ago(dep.created_at)}
                  </dd>
                </dl>

                <div className="mt">
                  <button className="btn" disabled={busy || !dep.litellm_registered} onClick={testThroughProxy}>
                    Send a test request through LiteLLM
                  </button>
                </div>
              </>
            )}

            {tab === "metrics" && (
              <>
                {points.length < 2 ? (
                  <div className="empty">Collecting samples… charts appear within a minute of going healthy.</div>
                ) : (
                  <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
                    {CHARTS.map((c) => {
                      const vals = points.map((p) => Number(p.values[c.key] ?? 0));
                      const last = vals[vals.length - 1] ?? 0;
                      return (
                        <section className="card" key={c.key}>
                          <div className="body">
                            <div className="flex">
                              <span className="hint">{c.label}</span>
                              <strong className="right mono">
                                {fmtNum(last, 1)} {c.unit}
                              </strong>
                            </div>
                            <div className="mt">
                              <Sparkline values={vals} width={320} height={56} color={c.color} />
                            </div>
                          </div>
                        </section>
                      );
                    })}
                  </div>
                )}
                <div className="hint mt">Last 60 minutes, sampled from the container's /metrics endpoint.</div>
              </>
            )}

            {tab === "logs" && (
              <>
                {logs?.findings?.length ? <Findings items={logs.findings} /> : null}
                <pre className="log" style={{ maxHeight: "60vh" }}>
                  {logs?.text ?? "loading…"}
                </pre>
              </>
            )}

            {tab === "config" && (
              <>
                <div className="hint mb">Exactly what is running on {dep.node_name}:</div>
                <pre className="log">
                  {[
                    "docker run -d \\",
                    `  --name ${dep.container_name} \\`,
                    `  --gpus '"device=${dep.gpu_indices.join(",")}"' --ipc=host --shm-size 16g \\`,
                    `  -p ${dep.port}:8000 \\`,
                    "  -v /opt/hf-cache:/root/.cache/huggingface \\",
                    `  ${dep.image} \\`,
                    `  ${(dep.vllm_args.argv ?? []).join(" ")}`,
                  ].join("\n")}
                </pre>
                <dl className="kv mt">
                  <dt>Container</dt>
                  <dd>{dep.container_name}</dd>
                  <dt>Image</dt>
                  <dd>{dep.image}</dd>
                  <dt>Deployment id</dt>
                  <dd>{dep.id}</dd>
                </dl>
              </>
            )}
          </div>
        </aside>
      </div>

      {confirm === "stop" && (
        <Confirm
          title="Stop this deployment?"
          danger
          confirmLabel="Stop it"
          body={
            <p>
              <strong>{dep.served_model_name}</strong> on <strong>{dep.node_name}</strong> will be removed from
              LiteLLM first, then the container is stopped. In-flight requests will fail.
            </p>
          }
          onCancel={() => setConfirm(null)}
          onConfirm={() => act("stop")}
        />
      )}
      {confirm === "restart" && (
        <Confirm
          title="Restart this deployment?"
          confirmLabel="Restart"
          body={
            <p>
              The container is recreated with identical arguments on the same GPUs. Expect a few minutes of
              downtime while weights reload.
            </p>
          }
          onCancel={() => setConfirm(null)}
          onConfirm={() => act("restart")}
        />
      )}
    </>
  );
}

function Stat({ k, v, unit, tone }: { k: string; v: string; unit?: string; tone?: string }) {
  return (
    <div className={`stat ${tone ?? ""}`}>
      <div className="k">{k}</div>
      <div className="v">
        {v} {unit && <small>{unit}</small>}
      </div>
    </div>
  );
}
