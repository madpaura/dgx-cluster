import { useState } from "react";
import { Pill, useToast } from "../components/ui";
import { api, ApiError, usePolled } from "../lib/api";
import type { LiteLLMStatus, Me } from "../types";

/** LiteLLM view. The question this page answers: can my users actually call
 *  the models I think I am serving, and by what name? */
export function Proxy({ me }: { me: Me | null }) {
  const { data, refresh } = usePolled<LiteLLMStatus>("/api/litellm/status", 10000);
  const [busy, setBusy] = useState(false);
  const [tested, setTested] = useState<Record<string, string>>({});
  const toast = useToast();
  const canWrite = me?.role === "admin" || me?.role === "deployer";

  async function resync() {
    setBusy(true);
    try {
      const r = await api.post<{ added: string[]; removed: string[]; errors: string[] }>("/api/litellm/resync");
      toast(
        `Registered ${r.added.length}, removed ${r.removed.length}` +
          (r.errors.length ? `, ${r.errors.length} error(s)` : ""),
        r.errors.length > 0
      );
      refresh();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  }

  async function test(name: string) {
    setTested((t) => ({ ...t, [name]: "…" }));
    try {
      const r = await api.post<{ ok: boolean; reply?: string; error?: string }>("/api/litellm/test", {
        model_name: name,
      });
      setTested((t) => ({ ...t, [name]: r.ok ? `✓ ${(r.reply ?? "").slice(0, 40)}` : `✗ ${r.error?.slice(0, 80)}` }));
    } catch (e) {
      setTested((t) => ({ ...t, [name]: `✗ ${String(e).slice(0, 80)}` }));
    }
  }

  if (!data) return <div className="empty">Loading proxy status…</div>;

  return (
    <div className="page">
      <section className="card mb">
        <header>
          <h3>LiteLLM proxy</h3>
          <div className="right flex">
            <Pill tone={data.reachable ? "ok" : "err"}>{data.reachable ? "reachable" : "unreachable"}</Pill>
            <button className="btn sm" disabled={busy || !canWrite} onClick={resync}>
              Resync with fleet
            </button>
          </div>
        </header>
        <div className="body">
          <dl className="kv">
            <dt>Base URL</dt>
            <dd>{data.base_url}</dd>
            <dt>Auto-register</dt>
            <dd>{data.auto_register ? "on — every healthy deployment is registered automatically" : "off"}</dd>
            <dt>Model groups</dt>
            <dd>{data.groups.length}</dd>
          </dl>
          {!data.reachable && (
            <div className="finding error mt">
              <div className="t">The proxy is not answering</div>
              <div className="d">{String(data.detail).slice(0, 300)}</div>
              <div className="fix">
                <b>Fix →</b> Check the litellm container is up on the control server and that{" "}
                <code>DGXCTL_LITELLM_MASTER_KEY</code> matches its master key. Models keep serving directly
                meanwhile; only proxy routing is affected.
              </div>
            </div>
          )}
          <p className="hint mt">
            Clients point at <code>{data.base_url}/v1</code> and use the model names below — routing, retries and
            load-balancing across replicas happen inside LiteLLM.
          </p>
        </div>
      </section>

      {data.groups.length === 0 ? (
        <div className="empty">No models registered with the proxy yet.</div>
      ) : (
        <section className="card">
          <header>
            <h3>Model groups</h3>
          </header>
          <table className="t">
            <thead>
              <tr>
                <th>Model name (what clients call)</th>
                <th>Replicas</th>
                <th>Backends</th>
                <th>Managed</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.groups.map((g) => (
                <tr key={g.model_name}>
                  <td>
                    <strong className="mono">{g.model_name}</strong>
                  </td>
                  <td className="num">{g.members.length}</td>
                  <td className="sub">{g.members.map((m) => m.api_base).join("  ")}</td>
                  <td>
                    {g.members.every((m) => m.managed_by_dgxctl) ? (
                      <Pill tone="info">dgxctl</Pill>
                    ) : (
                      <Pill tone="muted">external</Pill>
                    )}
                  </td>
                  <td>
                    <div className="flex">
                      <button className="btn sm" onClick={() => test(g.model_name)}>
                        Test
                      </button>
                      <span className="sub">{tested[g.model_name]}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
