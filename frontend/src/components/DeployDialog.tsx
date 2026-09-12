import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Deployment, ModelSpec, Plan } from "../types";
import { Modal, useToast } from "./ui";

export interface ScopeTarget {
  node_id: string;
  node_name: string;
}

/** The deploy flow. Everything that decides whether this will work is on one
 *  screen: what fits, what does not and why, and the exact command we will run.
 *
 *  Placement within a node is always automatic — picking GPU indices by hand is
 *  a job for the machine. What you choose is *which boxes are eligible*. */
export function DeployDialog({
  specs,
  scope,
  scopeLabel,
  onClose,
  onDeployed,
}: {
  specs: ModelSpec[];
  scope: ScopeTarget[];
  scopeLabel?: string;
  onClose: () => void;
  onDeployed: (deps: Deployment[]) => void;
}) {
  const [specKey, setSpecKey] = useState(specs[0]?.key ?? "");
  const [customRepo, setCustomRepo] = useState("");
  const [servedName, setServedName] = useState("");
  const [mode, setMode] = useState<"auto" | "scoped">(scope.length ? "scoped" : "auto");
  const [replicas, setReplicas] = useState(scope.length ? Math.min(scope.length, 8) : 1);
  const [tp, setTp] = useState<number | "">("");
  const [maxLen, setMaxLen] = useState<number | "">("");
  const [gpuUtil, setGpuUtil] = useState(0.9);
  const [image, setImage] = useState("");
  const [extra, setExtra] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const spec = useMemo(() => specs.find((s) => s.key === specKey), [specs, specKey]);
  const usingCustom = specKey === "__custom__";

  const body = useMemo(() => {
    let parsedExtra: Record<string, unknown> = {};
    if (extra.trim()) {
      try {
        parsedExtra = JSON.parse(extra);
      } catch {
        parsedExtra = {};
      }
    }
    return {
      spec_key: usingCustom ? null : specKey || null,
      hf_repo: usingCustom ? customRepo.trim() || null : null,
      served_model_name: servedName.trim() || null,
      tensor_parallel_size: tp === "" ? null : Number(tp),
      max_model_len: maxLen === "" ? 0 : Number(maxLen),
      gpu_memory_utilization: gpuUtil,
      extra_args: parsedExtra,
      image: image.trim(),
      replicas,
      node_ids: mode === "scoped" ? scope.map((t) => t.node_id) : [],
      targets: [],
    };
  }, [usingCustom, specKey, customRepo, servedName, tp, maxLen, gpuUtil, extra, image, mode, replicas, scope]);

  // Re-plan on every change, debounced. Free, and it means nobody deploys blind.
  useEffect(() => {
    if (!specKey || (usingCustom && !customRepo.trim())) {
      setPlan(null);
      return;
    }
    let cancelled = false;
    setPlanning(true);
    const t = window.setTimeout(async () => {
      try {
        const p = await api.post<Plan>("/api/deployments/plan", body);
        if (!cancelled) setPlan(p);
      } catch {
        if (!cancelled) setPlan(null);
      } finally {
        if (!cancelled) setPlanning(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [body, specKey, usingCustom, customRepo]);

  const canDeploy =
    !busy && !!(usingCustom ? customRepo.trim() : specKey) && (plan?.placements.length ?? 0) > 0;

  async function deploy() {
    setBusy(true);
    try {
      const created = await api.post<Deployment[]>("/api/deployments", body);
      const failed = created.filter((d) => d.status === "failed");
      toast(
        failed.length
          ? `${created.length - failed.length} started, ${failed.length} failed to launch`
          : `Deploying ${created[0]?.served_model_name} to ${created.length} target${created.length > 1 ? "s" : ""}`,
        failed.length > 0
      );
      onDeployed(created);
      onClose();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title="Deploy a model"
      onClose={onClose}
      footer={
        <>
          <span className="hint right" style={{ marginRight: "auto" }}>
            {replicas} replica{replicas > 1 ? "s" : ""}
            {mode === "scoped" ? ` across ${scope.length} chosen node${scope.length === 1 ? "" : "s"}` : ", anywhere in the fleet"}
          </span>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={!canDeploy} onClick={deploy}>
            {busy ? "Starting…" : "Deploy"}
          </button>
        </>
      }
    >
      <div className="row">
        <label className="f">
          <span>Model</span>
          <select value={specKey} onChange={(e) => setSpecKey(e.target.value)}>
            {specs.map((s) => (
              <option key={s.key} value={s.key}>
                {s.display_name} — {s.params_b}B{s.quantization ? ` ${s.quantization.toUpperCase()}` : ""}
              </option>
            ))}
            <option value="__custom__">Custom Hugging Face repo…</option>
          </select>
        </label>
        <label className="f">
          <span>Served as (LiteLLM model name)</span>
          <input
            value={servedName}
            placeholder={usingCustom ? customRepo.split("/").pop() ?? "" : specKey}
            onChange={(e) => setServedName(e.target.value)}
          />
        </label>
      </div>

      {usingCustom && (
        <label className="f">
          <span>Hugging Face repo</span>
          <input
            value={customRepo}
            placeholder="org/model-name"
            onChange={(e) => setCustomRepo(e.target.value)}
          />
        </label>
      )}

      {spec?.notes && <div className="hint mb">ℹ {spec.notes}</div>}

      <div className="flex mb wrap" style={{ gap: 16 }}>
        <label className="flex" style={{ gap: 6, cursor: "pointer" }}>
          <input
            type="radio"
            style={{ width: "auto" }}
            checked={mode === "auto"}
            onChange={() => setMode("auto")}
          />
          <span>Anywhere in the fleet</span>
        </label>
        <label
          className="flex"
          style={{ gap: 6, cursor: scope.length ? "pointer" : "not-allowed", opacity: scope.length ? 1 : 0.5 }}
        >
          <input
            type="radio"
            style={{ width: "auto" }}
            disabled={!scope.length}
            checked={mode === "scoped"}
            onChange={() => setMode("scoped")}
          />
          <span>
            Only {scopeLabel ?? `the ${scope.length} node${scope.length === 1 ? "" : "s"} I chose`}
          </span>
        </label>
        <label className="flex" style={{ gap: 6 }}>
          <span className="hint">Replicas</span>
          <input
            type="number"
            min={1}
            max={16}
            value={replicas}
            style={{ width: 64 }}
            onChange={(e) => setReplicas(Math.max(1, Number(e.target.value)))}
          />
        </label>
      </div>

      {/* ---------------------------------------------------- plan preview */}
      <section className="card mb">
        <header>
          <h3>Where it will land</h3>
          <span className="hint right">
            {planning
              ? "checking…"
              : plan
                ? `needs ~${plan.per_gpu_gb} GB per GPU × TP${plan.tensor_parallel_size}`
                : ""}
          </span>
        </header>
        <div className="body">
          {!plan ? (
            <div className="hint">Pick a model to see placement.</div>
          ) : plan.placements.length === 0 ? (
            <div>
              <div className="mb" style={{ color: "var(--err)" }}>
                {mode === "scoped"
                  ? `Nothing in ${scopeLabel ?? "the chosen nodes"} can host this right now.`
                  : "Nothing in the fleet can host this right now."}
              </div>
              {plan.rejections.map((r) => (
                <div key={r.node_name} className="hint">
                  <strong style={{ color: "var(--fg-2)" }}>{r.node_name}</strong> — {r.reason}
                </div>
              ))}
            </div>
          ) : (
            <>
              <table className="t">
                <tbody>
                  {plan.placements.slice(0, replicas).map((p) => (
                    <tr key={p.node_id}>
                      <td>
                        <strong>{p.node_name}</strong>
                      </td>
                      <td className="mono sub">GPU {p.gpu_indices.join(",")}</td>
                      <td className="sub">{p.gpu_model.replace("NVIDIA ", "")}</td>
                      <td className="num sub">{p.free_gb_per_gpu} GB free</td>
                      <td className="sub">{p.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {plan.placements.length < replicas && (
                <div className="hint mt" style={{ color: "var(--warn)" }}>
                  ⚠ Only {plan.placements.length} of {replicas} replicas can be placed right now.
                </div>
              )}
              {plan.rejections.length > 0 && (
                <details className="mt">
                  <summary className="hint" style={{ cursor: "pointer" }}>
                    {plan.rejections.length} node(s) skipped
                  </summary>
                  <div className="mt">
                    {plan.rejections.map((r) => (
                      <div key={r.node_name} className="hint">
                        <strong style={{ color: "var(--fg-2)" }}>{r.node_name}</strong> — {r.reason}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </>
          )}
        </div>
      </section>

      <button className="btn ghost sm mb" onClick={() => setAdvanced((v) => !v)}>
        {advanced ? "▾" : "▸"} vLLM settings
      </button>

      {advanced && (
        <>
          <div className="row">
            <label className="f">
              <span>Tensor parallel</span>
              <input
                type="number"
                min={1}
                max={16}
                value={tp}
                placeholder={String(spec?.recommended_tp ?? 1)}
                onChange={(e) => setTp(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </label>
            <label className="f">
              <span>Max model len</span>
              <input
                type="number"
                value={maxLen}
                placeholder={String(spec?.max_model_len || "model default")}
                onChange={(e) => setMaxLen(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </label>
            <label className="f">
              <span>GPU memory util</span>
              <input
                type="number"
                step={0.01}
                min={0.5}
                max={0.99}
                value={gpuUtil}
                onChange={(e) => setGpuUtil(Number(e.target.value))}
              />
            </label>
          </div>
          <label className="f">
            <span>vLLM image override</span>
            <input value={image} placeholder="vllm/vllm-openai:latest" onChange={(e) => setImage(e.target.value)} />
          </label>
          <label className="f">
            <span>Extra vLLM flags (JSON)</span>
            <textarea
              rows={2}
              value={extra}
              placeholder='{"--enable-prefix-caching": true, "--max-num-seqs": 128}'
              onChange={(e) => setExtra(e.target.value)}
            />
          </label>
        </>
      )}

      {plan && (
        <div>
          <div className="hint mb">This is the command that will run on each target:</div>
          <pre className="log" style={{ maxHeight: 110 }}>
            docker run --gpus … {plan.argv.join(" ")}
          </pre>
        </div>
      )}
    </Modal>
  );
}
