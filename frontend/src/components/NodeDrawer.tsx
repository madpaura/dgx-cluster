import { useEffect, useState } from "react";
import { api, ApiError, usePolled } from "../lib/api";
import { DEPLOY_TONE, NODE_TONE, ago, fmtGb } from "../lib/format";
import type { Deployment, Finding, Node } from "../types";
import { Findings } from "./Findings";
import { Confirm, Pill, useToast } from "./ui";

interface Reconcile {
  error: string;
  orphans: { name: string; image: string; state: string; model: string }[];
  missing: string[];
}

export function NodeDrawer({
  nodeId,
  isAdmin,
  onClose,
  onChanged,
  onOpenDeployment,
}: {
  nodeId: string;
  isAdmin: boolean;
  onClose: () => void;
  onChanged: () => void;
  onOpenDeployment: (id: string) => void;
}) {
  const { data: node, refresh } = usePolled<Node>(`/api/nodes/${nodeId}`, 6000);
  const { data: findings } = usePolled<Finding[]>(`/api/nodes/${nodeId}/diagnostics`, 15000);
  const { data: deps } = usePolled<Deployment[]>(`/api/deployments?active_only=false&node_id=${nodeId}`, 8000);
  const [recon, setRecon] = useState<Reconcile | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const toast = useToast();

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  if (!node) return null;
  const active = (deps ?? []).filter((d) => !["stopped", "failed"].includes(d.status));

  async function run<T>(fn: () => Promise<T>, ok: string) {
    setBusy(true);
    try {
      const r = await fn();
      toast(ok);
      refresh();
      onChanged();
      return r;
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
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
              <div style={{ fontWeight: 650, fontSize: 15 }}>{node.name}</div>
              <div className="hint mono">
                {node.hostname}:{node.ssh_port} · {node.kind}
              </div>
            </div>
            <div className="right flex">
              <Pill tone={NODE_TONE[node.status]}>{node.status}</Pill>
              <button className="btn ghost sm" onClick={onClose}>
                ✕
              </button>
            </div>
          </header>

          <div className="body">
            {findings?.length ? <Findings items={findings} /> : null}

            <div className="flex wrap mb">
              <button className="btn sm" disabled={busy} onClick={() => run(() => api.post(`/api/nodes/${nodeId}/probe`), "Probed")}>
                Probe now
              </button>
              <button
                className="btn sm"
                disabled={busy}
                onClick={async () => {
                  const r = await run(() => api.post<Reconcile>(`/api/nodes/${nodeId}/reconcile`), "Reconciled");
                  if (r) setRecon(r);
                }}
              >
                Reconcile containers
              </button>
              {isAdmin && node.status !== "draining" && (
                <button className="btn sm" disabled={busy} onClick={() => run(() => api.post(`/api/nodes/${nodeId}/drain`), "Draining — no new deployments here")}>
                  Drain
                </button>
              )}
              {isAdmin && node.status === "draining" && (
                <button className="btn sm" disabled={busy} onClick={() => run(() => api.post(`/api/nodes/${nodeId}/drain?undo=true`), "Back in rotation")}>
                  Undrain
                </button>
              )}
              {isAdmin && (
                <button className="btn sm danger right" disabled={busy} onClick={() => setConfirmDelete(true)}>
                  Remove node
                </button>
              )}
            </div>

            {recon && (
              <section className="card mb">
                <header>
                  <h3>Reconcile result</h3>
                </header>
                <div className="body">
                  {recon.error && <div style={{ color: "var(--err)" }}>{recon.error}</div>}
                  {!recon.error && !recon.orphans.length && !recon.missing.length && (
                    <div className="hint">The node matches our records exactly.</div>
                  )}
                  {recon.missing.length > 0 && (
                    <div className="hint mb" style={{ color: "var(--warn)" }}>
                      {recon.missing.length} deployment(s) had no container on the node and were marked failed.
                    </div>
                  )}
                  {recon.orphans.map((o) => (
                    <div key={o.name} className="hint">
                      Orphan container <code>{o.name}</code> ({o.state}) serving {o.model || "?"} — clean it up on the
                      node with <code>docker rm -f {o.name}</code>.
                    </div>
                  ))}
                </div>
              </section>
            )}

            <dl className="kv mb">
              <dt>Cluster</dt>
              <dd>{node.cluster_name || "Unassigned"}</dd>
              <dt>GPUs</dt>
              <dd>
                {node.gpus.length}× {node.gpus[0]?.name ?? "–"}
              </dd>
              <dt>VRAM</dt>
              <dd>
                {fmtGb(node.gpus.reduce((a, g) => a + g.memory_used_mb, 0), 1)}B used /{" "}
                {fmtGb(node.gpus.reduce((a, g) => a + g.memory_total_mb, 0))}B
              </dd>
              <dt>Driver / CUDA</dt>
              <dd>
                {node.driver_version || "–"} / {node.cuda_version || "–"}
              </dd>
              <dt>Docker</dt>
              <dd>{node.docker_version || "–"}</dd>
              <dt>Host</dt>
              <dd>
                {node.cpu_count} vCPU · {node.memory_gb} GB RAM
              </dd>
              <dt>Labels</dt>
              <dd>
                {Object.entries(node.labels)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(" ") || "–"}
              </dd>
              <dt>Last seen</dt>
              <dd>{ago(node.last_seen)}</dd>
            </dl>

            <section className="card">
              <header>
                <h3>GPU detail</h3>
              </header>
              <table className="t">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Serving</th>
                    <th style={{ textAlign: "right" }}>Reserved</th>
                    <th style={{ textAlign: "right" }}>Util</th>
                    <th style={{ textAlign: "right" }}>VRAM</th>
                    <th style={{ textAlign: "right" }}>Temp</th>
                    <th style={{ textAlign: "right" }}>Power</th>
                  </tr>
                </thead>
                <tbody>
                  {node.gpus.map((g) => (
                    <tr
                      key={g.index}
                      className={g.tenants.length ? "clickable" : ""}
                      onClick={() => g.tenants[0] && onOpenDeployment(g.tenants[0].deployment_id)}
                    >
                      <td className="mono">{g.index}</td>
                      <td>
                        {g.tenants.length === 0 ? (
                          <span className="muted">free</span>
                        ) : (
                          <div className="flex wrap" style={{ gap: 5 }}>
                            {g.tenants.map((t) => (
                              <button
                                key={t.deployment_id}
                                className="btn ghost sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onOpenDeployment(t.deployment_id);
                                }}
                              >
                                {t.model_name}
                              </button>
                            ))}
                          </div>
                        )}
                      </td>
                      <td className="num">
                        {g.reserved_mb ? `${fmtGb(g.reserved_mb, 0)}B` : "–"}
                      </td>
                      <td className="num">{g.utilization.toFixed(0)}%</td>
                      <td className="num">
                        {fmtGb(g.memory_used_mb, 1)}/{fmtGb(g.memory_total_mb)}B
                      </td>
                      <td className="num" style={{ color: g.temperature_c >= 85 ? "var(--err)" : undefined }}>
                        {g.temperature_c.toFixed(0)}°
                      </td>
                      <td className="num">{g.power_draw_w.toFixed(0)}W</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            {active.length > 0 && (
              <section className="card mt">
                <header>
                  <h3>Running here</h3>
                </header>
                <table className="t">
                  <tbody>
                    {active.map((d) => (
                      <tr key={d.id} className="clickable" onClick={() => onOpenDeployment(d.id)}>
                        <td>{d.served_model_name}</td>
                        <td className="sub">GPU {d.gpu_indices.join(",")}</td>
                        <td>
                          <Pill tone={DEPLOY_TONE[d.status]}>{d.status}</Pill>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </div>
        </aside>
      </div>

      {confirmDelete && (
        <Confirm
          title={`Remove ${node.name}?`}
          danger
          confirmLabel="Remove"
          body={
            <p>
              This only removes the node from dgxctl's inventory — nothing on the machine is touched. Any active
              deployments must be stopped first.
            </p>
          }
          onCancel={() => setConfirmDelete(false)}
          onConfirm={async () => {
            setConfirmDelete(false);
            await run(() => api.del(`/api/nodes/${nodeId}`), "Node removed");
            onClose();
          }}
        />
      )}
    </>
  );
}
