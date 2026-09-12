import { useMemo, useState } from "react";
import { DeployDialog } from "../components/DeployDialog";
import { DeploymentDrawer } from "../components/DeploymentDrawer";
import { Confirm, Pill, useToast } from "../components/ui";
import { api, ApiError, useLiveEvents, usePolled } from "../lib/api";
import { DEPLOY_TONE, ago, fmtNum } from "../lib/format";
import type { Deployment, Me, ModelSpec } from "../types";

/** Models, not containers. Replicas of one served name collapse into a single
 *  row — because to everyone calling the proxy they are one model. */
export function Models({ me }: { me: Me | null }) {
  const { data: deps, refresh } = usePolled<Deployment[]>("/api/deployments?active_only=false", 6000);
  const { data: specs } = usePolled<ModelSpec[]>("/api/catalog", 60000);
  const [open, setOpen] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deploying, setDeploying] = useState(false);
  const [showStopped, setShowStopped] = useState(false);
  const [confirmBulk, setConfirmBulk] = useState<null | "stop" | "restart">(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  useLiveEvents(["deployment", "metrics"], refresh);
  const canDeploy = me?.role === "admin" || me?.role === "deployer";

  const groups = useMemo(() => {
    const rows = (deps ?? []).filter((d) => showStopped || d.status !== "stopped");
    const map = new Map<string, Deployment[]>();
    rows.forEach((d) => map.set(d.served_model_name, [...(map.get(d.served_model_name) ?? []), d]));
    return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [deps, showStopped]);

  function toggleSel(ids: string[], on: boolean) {
    setSelected((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
      return next;
    });
  }

  async function bulk(kind: "stop" | "restart") {
    setBusy(true);
    try {
      const done = await api.post<Deployment[]>(`/api/deployments/bulk/${kind}`, {
        deployment_ids: [...selected],
      });
      toast(`${kind === "stop" ? "Stopped" : "Restarted"} ${done.length} deployment(s)`);
      setSelected(new Set());
      refresh();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
      setConfirmBulk(null);
    }
  }

  return (
    <div className="page">
      <div className="flex mb">
        <label className="flex hint" style={{ gap: 6, cursor: "pointer" }}>
          <input
            type="checkbox"
            style={{ width: "auto" }}
            checked={showStopped}
            onChange={(e) => setShowStopped(e.target.checked)}
          />
          Show stopped
        </label>
        <div className="right">
          <button className="btn primary" disabled={!canDeploy} onClick={() => setDeploying(true)}>
            Deploy a model
          </button>
        </div>
      </div>

      {groups.length === 0 ? (
        <div className="empty">
          Nothing deployed yet.
          {canDeploy && <div className="mt">Pick a model from the catalog and dgxctl will find it a home.</div>}
        </div>
      ) : (
        <section className="card">
          <table className="t">
            <thead>
              <tr>
                <th style={{ width: 28 }}></th>
                <th>Model</th>
                <th>Replicas</th>
                <th>Proxy</th>
                <th style={{ textAlign: "right" }}>tok/s</th>
                <th style={{ textAlign: "right" }}>In flight</th>
                <th style={{ textAlign: "right" }}>Queued</th>
                <th style={{ textAlign: "right" }}>TTFT</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {groups.map(([name, members]) => {
                const healthy = members.filter((d) => d.status === "healthy");
                const bad = members.filter((d) => ["failed", "degraded"].includes(d.status));
                const tps = healthy.reduce((a, d) => a + (d.last_metrics.gen_tps ?? 0), 0);
                const running = healthy.reduce((a, d) => a + (d.last_metrics.running ?? 0), 0);
                const waiting = healthy.reduce((a, d) => a + (d.last_metrics.waiting ?? 0), 0);
                const ttft = healthy.length
                  ? healthy.reduce((a, d) => a + (d.last_metrics.ttft_avg_ms ?? 0), 0) / healthy.length
                  : undefined;
                const isOpen = expanded.has(name);
                const allSel = members.every((d) => selected.has(d.id));
                const inProxy = members.filter((d) => d.litellm_registered).length;

                return (
                  <>
                    <tr
                      key={name}
                      className="clickable"
                      onClick={() =>
                        setExpanded((p) => {
                          const n = new Set(p);
                          n.has(name) ? n.delete(name) : n.add(name);
                          return n;
                        })
                      }
                    >
                      <td onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          style={{ width: "auto" }}
                          checked={allSel}
                          onChange={(e) => toggleSel(members.map((d) => d.id), e.target.checked)}
                        />
                      </td>
                      <td>
                        <span className="muted mono">{isOpen ? "▾" : "▸"}</span>{" "}
                        <strong>{name}</strong>
                        <div className="sub">{members[0].hf_repo}</div>
                      </td>
                      <td>
                        <div className="flex" style={{ gap: 5 }}>
                          <Pill tone={bad.length ? "warn" : healthy.length ? "ok" : "muted"}>
                            {healthy.length}/{members.length} healthy
                          </Pill>
                        </div>
                        <div className="sub">{[...new Set(members.map((d) => d.node_name))].join(", ")}</div>
                      </td>
                      <td>
                        {inProxy > 0 ? (
                          <Pill tone="info">{inProxy === members.length ? "routed" : `${inProxy}/${members.length}`}</Pill>
                        ) : (
                          <span className="muted sub">not routed</span>
                        )}
                      </td>
                      <td className="num">{fmtNum(tps, 1)}</td>
                      <td className="num">{fmtNum(running)}</td>
                      <td className="num" style={{ color: waiting > 0 ? "var(--warn)" : undefined }}>
                        {fmtNum(waiting)}
                      </td>
                      <td className="num">{ttft === undefined ? "–" : `${fmtNum(ttft)}ms`}</td>
                      <td onClick={(e) => e.stopPropagation()}>
                        <button
                          className="btn sm"
                          disabled={!canDeploy}
                          title="Add another replica of this model"
                          onClick={async () => {
                            setBusy(true);
                            try {
                              await api.post("/api/deployments", {
                                hf_repo: members[0].hf_repo,
                                served_model_name: name,
                                tensor_parallel_size: members[0].tensor_parallel_size,
                                replicas: 1,
                              });
                              toast(`Adding a replica of ${name}`);
                              refresh();
                            } catch (e) {
                              toast(e instanceof ApiError ? e.message : String(e), true);
                            } finally {
                              setBusy(false);
                            }
                          }}
                        >
                          + replica
                        </button>
                      </td>
                    </tr>
                    {isOpen &&
                      members.map((d) => (
                        <tr key={d.id} className="clickable" onClick={() => setOpen(d.id)}>
                          <td onClick={(e) => e.stopPropagation()}>
                            <input
                              type="checkbox"
                              style={{ width: "auto" }}
                              checked={selected.has(d.id)}
                              onChange={(e) => toggleSel([d.id], e.target.checked)}
                            />
                          </td>
                          <td style={{ paddingLeft: 30 }}>
                            <span className="sub">
                              {d.node_name} · GPU {d.gpu_indices.join(",")} · TP{d.tensor_parallel_size}
                            </span>
                          </td>
                          <td>
                            <Pill tone={DEPLOY_TONE[d.status]}>{d.status}</Pill>
                          </td>
                          <td className="sub">{d.litellm_registered ? "yes" : "—"}</td>
                          <td className="num">{fmtNum(d.last_metrics.gen_tps, 1)}</td>
                          <td className="num">{fmtNum(d.last_metrics.running)}</td>
                          <td className="num">{fmtNum(d.last_metrics.waiting)}</td>
                          <td className="num">
                            {d.last_metrics.ttft_avg_ms ? `${fmtNum(d.last_metrics.ttft_avg_ms)}ms` : "–"}
                          </td>
                          <td className="sub">{ago(d.created_at)}</td>
                        </tr>
                      ))}
                    {isOpen && bad.length > 0 && (
                      <tr key={`${name}-why`}>
                        <td></td>
                        <td colSpan={8} className="sub" style={{ color: "var(--warn)" }}>
                          {bad[0].node_name}: {bad[0].status_reason}
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {selected.size > 0 && (
        <div className="selbar">
          <strong>{selected.size} selected</strong>
          <div className="right flex">
            <button className="btn ghost" onClick={() => setSelected(new Set())}>
              Clear
            </button>
            <button className="btn" disabled={busy} onClick={() => setConfirmBulk("restart")}>
              Restart all
            </button>
            <button className="btn danger" disabled={busy} onClick={() => setConfirmBulk("stop")}>
              Stop all
            </button>
          </div>
        </div>
      )}

      {confirmBulk && (
        <Confirm
          title={`${confirmBulk === "stop" ? "Stop" : "Restart"} ${selected.size} deployment(s)?`}
          danger={confirmBulk === "stop"}
          confirmLabel={confirmBulk === "stop" ? "Stop them" : "Restart them"}
          body={
            <p>
              {confirmBulk === "stop"
                ? "They are removed from LiteLLM first, then their containers stop. In-flight requests will fail."
                : "Each container is recreated with identical arguments. Expect downtime while weights reload."}
            </p>
          }
          onCancel={() => setConfirmBulk(null)}
          onConfirm={() => bulk(confirmBulk)}
        />
      )}

      {deploying && (
        <DeployDialog specs={specs ?? []} scope={[]} onClose={() => setDeploying(false)} onDeployed={refresh} />
      )}
      {open && <DeploymentDrawer deploymentId={open} onClose={() => setOpen(null)} onChanged={refresh} />}
    </div>
  );
}
