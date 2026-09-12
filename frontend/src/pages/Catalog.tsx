import { useState } from "react";
import { Confirm, Modal, Pill, useToast } from "../components/ui";
import { api, ApiError, usePolled } from "../lib/api";
import type { Me, ModelSpec } from "../types";

const BLANK = {
  key: "",
  display_name: "",
  hf_repo: "",
  revision: "",
  params_b: 0,
  quantization: "",
  min_gpu_memory_gb: 0,
  recommended_tp: 1,
  max_model_len: 0,
  extra_args: {},
  vllm_image: "",
  tags: [] as string[],
  notes: "",
};

/** The catalog is what makes deploying a one-click operation: sizing, tensor
 *  parallel and flags are decided once, here, instead of in everyone's head. */
export function Catalog({ me }: { me: Me | null }) {
  const { data: specs, refresh } = usePolled<ModelSpec[]>("/api/catalog", 30000);
  const [editing, setEditing] = useState<(typeof BLANK & { id?: string }) | null>(null);
  const [deleting, setDeleting] = useState<ModelSpec | null>(null);
  const toast = useToast();
  const canWrite = me?.role === "admin" || me?.role === "deployer";

  async function save() {
    if (!editing) return;
    try {
      const { id, ...body } = editing;
      if (id) await api.put(`/api/catalog/${id}`, body);
      else await api.post("/api/catalog", body);
      toast(id ? "Catalog entry updated" : "Catalog entry added");
      setEditing(null);
      refresh();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    }
  }

  return (
    <div className="page">
      <div className="flex mb">
        <p className="hint" style={{ margin: 0 }}>
          Per-GPU memory is what dgxctl uses to decide where a model fits. Tune it once against your own hardware.
        </p>
        {canWrite && (
          <button className="btn primary right" onClick={() => setEditing({ ...BLANK })}>
            + Add model
          </button>
        )}
      </div>

      <section className="card">
        <table className="t">
          <thead>
            <tr>
              <th>Model</th>
              <th>Repo</th>
              <th style={{ textAlign: "right" }}>Params</th>
              <th style={{ textAlign: "right" }}>VRAM / GPU</th>
              <th style={{ textAlign: "right" }}>TP</th>
              <th>Tags</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(specs ?? []).map((s) => (
              <tr key={s.id}>
                <td>
                  <strong>{s.display_name}</strong>
                  <div className="sub">{s.key}</div>
                </td>
                <td className="sub">{s.hf_repo}</td>
                <td className="num">{s.params_b}B</td>
                <td className="num">{s.min_gpu_memory_gb} GB</td>
                <td className="num">{s.recommended_tp}</td>
                <td>
                  <div className="flex wrap" style={{ gap: 4 }}>
                    {s.quantization && <Pill tone="busy">{s.quantization}</Pill>}
                    {s.tags.map((t) => (
                      <Pill key={t} tone="muted">
                        {t}
                      </Pill>
                    ))}
                  </div>
                </td>
                <td>
                  {canWrite && (
                    <div className="flex">
                      <button className="btn sm ghost" onClick={() => setEditing({ ...s })}>
                        Edit
                      </button>
                      <button className="btn sm ghost" onClick={() => setDeleting(s)}>
                        ✕
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {editing && (
        <Modal
          title={editing.id ? `Edit ${editing.display_name}` : "Add a model to the catalog"}
          width={720}
          onClose={() => setEditing(null)}
          footer={
            <>
              <button className="btn" onClick={() => setEditing(null)}>
                Cancel
              </button>
              <button className="btn primary" disabled={!editing.key || !editing.hf_repo} onClick={save}>
                Save
              </button>
            </>
          }
        >
          <div className="row">
            <label className="f">
              <span>Key (short handle, used as the served model name)</span>
              <input value={editing.key} onChange={(e) => setEditing({ ...editing, key: e.target.value })} />
            </label>
            <label className="f">
              <span>Display name</span>
              <input
                value={editing.display_name}
                onChange={(e) => setEditing({ ...editing, display_name: e.target.value })}
              />
            </label>
          </div>
          <div className="row">
            <label className="f">
              <span>Hugging Face repo</span>
              <input value={editing.hf_repo} onChange={(e) => setEditing({ ...editing, hf_repo: e.target.value })} />
            </label>
            <label className="f">
              <span>Revision</span>
              <input
                value={editing.revision}
                placeholder="main"
                onChange={(e) => setEditing({ ...editing, revision: e.target.value })}
              />
            </label>
          </div>
          <div className="row">
            <label className="f">
              <span>Params (B)</span>
              <input
                type="number"
                value={editing.params_b}
                onChange={(e) => setEditing({ ...editing, params_b: Number(e.target.value) })}
              />
            </label>
            <label className="f">
              <span>VRAM per GPU (GB)</span>
              <input
                type="number"
                value={editing.min_gpu_memory_gb}
                onChange={(e) => setEditing({ ...editing, min_gpu_memory_gb: Number(e.target.value) })}
              />
            </label>
            <label className="f">
              <span>Tensor parallel</span>
              <input
                type="number"
                min={1}
                value={editing.recommended_tp}
                onChange={(e) => setEditing({ ...editing, recommended_tp: Number(e.target.value) })}
              />
            </label>
            <label className="f">
              <span>Max model len</span>
              <input
                type="number"
                value={editing.max_model_len}
                onChange={(e) => setEditing({ ...editing, max_model_len: Number(e.target.value) })}
              />
            </label>
          </div>
          <div className="row">
            <label className="f">
              <span>Quantization</span>
              <select
                value={editing.quantization}
                onChange={(e) => setEditing({ ...editing, quantization: e.target.value })}
              >
                <option value="">from checkpoint</option>
                <option value="fp8">fp8</option>
                <option value="awq">awq</option>
                <option value="gptq">gptq</option>
                <option value="bitsandbytes">bitsandbytes</option>
              </select>
            </label>
            <label className="f">
              <span>vLLM image override</span>
              <input
                value={editing.vllm_image}
                placeholder="(global default)"
                onChange={(e) => setEditing({ ...editing, vllm_image: e.target.value })}
              />
            </label>
            <label className="f">
              <span>Tags (comma separated)</span>
              <input
                value={editing.tags.join(", ")}
                onChange={(e) =>
                  setEditing({ ...editing, tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean) })
                }
              />
            </label>
          </div>
          <label className="f">
            <span>Extra vLLM flags (JSON)</span>
            <textarea
              rows={2}
              value={JSON.stringify(editing.extra_args)}
              onChange={(e) => {
                try {
                  setEditing({ ...editing, extra_args: JSON.parse(e.target.value || "{}") });
                } catch {
                  /* keep typing */
                }
              }}
            />
          </label>
          <label className="f">
            <span>Notes shown at deploy time</span>
            <textarea rows={2} value={editing.notes} onChange={(e) => setEditing({ ...editing, notes: e.target.value })} />
          </label>
        </Modal>
      )}

      {deleting && (
        <Confirm
          title={`Remove ${deleting.display_name} from the catalog?`}
          danger
          confirmLabel="Remove"
          body={<p>Running deployments of this model are not affected.</p>}
          onCancel={() => setDeleting(null)}
          onConfirm={async () => {
            try {
              await api.del(`/api/catalog/${deleting.id}`);
              toast("Removed from catalog");
              refresh();
            } catch (e) {
              toast(String(e), true);
            } finally {
              setDeleting(null);
            }
          }}
        />
      )}
    </div>
  );
}
