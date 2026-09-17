import { useEffect, useState } from "react";
import { api, ApiError, usePolled } from "../lib/api";
import { useToast } from "../components/ui";
import type { LLMSettings, LLMTestResult, Me } from "../types";

/** Portal settings. Currently one thing: the model that drafts catalog entries
 *  from a pasted URL.
 *
 *  The default is the fleet's own LiteLLM proxy — whatever you already serve is
 *  good enough to read a model card, and it means nothing leaves the building
 *  and there is no external account to set up. */
export function Settings({ me }: { me: Me | null }) {
  const { data, refresh } = usePolled<LLMSettings>("/api/settings/llm", 0);
  const [form, setForm] = useState<LLMSettings | null>(null);
  const [key, setKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<LLMTestResult | null>(null);
  const toast = useToast();

  const isAdmin = me?.role === "admin";

  useEffect(() => {
    if (data && !form) setForm(data);
  }, [data, form]);

  if (!isAdmin) {
    return (
      <div className="page">
        <section className="card">
          <div className="body">
            <p className="hint">Portal settings are admin-only. Ask an admin to change these.</p>
          </div>
        </section>
      </div>
    );
  }

  if (!form) {
    return (
      <div className="page">
        <section className="card">
          <div className="body hint">Loading…</div>
        </section>
      </div>
    );
  }

  async function save() {
    if (!form) return;
    setBusy(true);
    setTest(null);
    try {
      const saved = await api.put<LLMSettings>("/api/settings/llm", {
        enabled: form.enabled,
        base_url: form.base_url.trim(),
        model: form.model.trim(),
        // "" keeps the stored key (we were never shown it); null clears it.
        api_key: clearKey ? null : key,
        timeout_s: form.timeout_s,
        max_input_chars: form.max_input_chars,
      });
      setForm(saved);
      setKey("");
      setClearKey(false);
      toast("Settings saved");
      refresh();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  }

  async function runTest() {
    setBusy(true);
    try {
      setTest(await api.post<LLMTestResult>("/api/settings/llm/test", {}));
    } catch (e) {
      setTest({ ok: false, detail: e instanceof ApiError ? e.message : String(e), model: "" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <section className="card mb">
        <header>
          <h3>Drafting model</h3>
          <span className="hint right">used by Catalog → Import from URL</span>
        </header>
        <div className="body">
          <p className="hint mb">
            Paste a Hugging Face or GitHub link into the catalog and this model reads the page
            and fills the form in for you. You review every field before anything is saved. It
            never decides how much VRAM a model needs — that stays measured from the model's own
            config, because it is what blocks a deploy that would OOM the box.
          </p>

          <label className="flex mb" style={{ gap: 8, cursor: "pointer" }}>
            <input
              type="checkbox"
              style={{ width: "auto" }}
              checked={form.enabled}
              onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
            />
            <span>Enable LLM-assisted catalog import</span>
          </label>

          <div className="row">
            <label className="f">
              <span>Endpoint (OpenAI-compatible)</span>
              <input
                value={form.base_url}
                placeholder={`leave blank for your own proxy — ${form.effective_base_url}`}
                onChange={(e) => setForm({ ...form, base_url: e.target.value })}
              />
            </label>
            <label className="f">
              <span>Model</span>
              <input
                value={form.model}
                placeholder="a model name your endpoint serves"
                onChange={(e) => setForm({ ...form, model: e.target.value })}
              />
            </label>
          </div>
          <p className="hint mb">
            Blank endpoint means <code>{form.effective_base_url}</code> — the proxy this portal
            already manages, authenticated with its master key. Any model you have deployed will
            do; a small instruct model is plenty for reading a model card.
          </p>

          <div className="row">
            <label className="f">
              <span>
                API key{" "}
                {form.has_api_key && !clearKey && (
                  <span className="hint">— stored, ending {form.api_key_hint}</span>
                )}
              </span>
              <input
                type="password"
                autoComplete="new-password"
                value={clearKey ? "" : key}
                disabled={clearKey}
                placeholder={form.has_api_key ? "leave blank to keep the stored key" : "not needed for your own proxy"}
                onChange={(e) => setKey(e.target.value)}
              />
            </label>
            <label className="f">
              <span>Timeout (seconds)</span>
              <input
                type="number"
                min={5}
                max={600}
                value={form.timeout_s}
                onChange={(e) => setForm({ ...form, timeout_s: Number(e.target.value) })}
              />
            </label>
            <label className="f">
              <span>Max page characters</span>
              <input
                type="number"
                min={1000}
                max={200000}
                value={form.max_input_chars}
                onChange={(e) => setForm({ ...form, max_input_chars: Number(e.target.value) })}
              />
            </label>
          </div>

          {form.has_api_key && (
            <label className="flex mb" style={{ gap: 8, cursor: "pointer" }}>
              <input
                type="checkbox"
                style={{ width: "auto" }}
                checked={clearKey}
                onChange={(e) => setClearKey(e.target.checked)}
              />
              <span className="hint">Remove the stored API key</span>
            </label>
          )}

          <div className="flex mt" style={{ gap: 8 }}>
            <button className="btn primary" disabled={busy} onClick={save}>
              {busy ? "Working…" : "Save"}
            </button>
            <button className="btn" disabled={busy || !data?.enabled} onClick={runTest} title={
              data?.enabled ? "Send one short prompt and report what comes back" : "Save an enabled configuration first"
            }>
              Test connection
            </button>
            {data?.updated_by && (
              <span className="hint right">last changed by {data.updated_by}</span>
            )}
          </div>

          {test && (
            <div className={`finding ${test.ok ? "info" : "error"} mt`}>
              <div className="t">{test.ok ? "The model answered" : "That did not work"}</div>
              <div className="d">
                {test.detail}
                {test.model ? ` (${test.model})` : ""}
              </div>
              {!test.ok && (
                <div className="fix">
                  <b>Check →</b> the endpoint is reachable from the control server (not just from
                  your laptop), the model name is one it actually serves, and the key is right.
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section className="card">
        <header>
          <h3>What this model can and cannot do</h3>
        </header>
        <div className="body">
          <ul className="hint" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
            <li>It only reads pages on huggingface.co and github.com. Other URLs are refused —
              this server can reach machines inside your network that a browser cannot.</li>
            <li>It never writes to the catalog. It produces a draft you edit and save yourself.</li>
            <li>It never sets the VRAM figure. That is measured from the model's config and is
              what stops a deploy that would hang a DGX.</li>
            <li>Model cards are untrusted text. Anything odd in a draft is a page being strange,
              not a decision the portal has made.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
