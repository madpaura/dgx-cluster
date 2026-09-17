import { useState } from "react";
import { api, ApiError } from "../lib/api";
import type { CatalogDraft } from "../types";
import { Modal } from "./ui";

const EXAMPLES = [
  "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
  "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
];

/** Paste a model page, get a filled-in catalog form.
 *
 *  The draft never reaches the database from here — it opens the normal editor,
 *  so every field is something a person looked at before it was saved. */
export function ImportDialog({
  onClose,
  onDrafted,
}: {
  onClose: () => void;
  onDrafted: (draft: CatalogDraft) => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setBusy(true);
    setError("");
    try {
      onDrafted(await api.post<CatalogDraft>("/api/catalog/import", { url: url.trim() }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title="Import a model from a URL"
      onClose={onClose}
      footer={
        <>
          <span className="hint" style={{ marginRight: "auto" }}>
            Nothing is saved — you get the form to check first.
          </span>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={busy || !url.trim()} onClick={run}>
            {busy ? "Reading the page…" : "Read page"}
          </button>
        </>
      }
    >
      <label className="f">
        <span>Hugging Face or GitHub URL</span>
        <input
          value={url}
          autoFocus
          placeholder="https://huggingface.co/org/model"
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && url.trim() && !busy) void run();
          }}
        />
      </label>

      <div className="hint mb">
        Try:{" "}
        {EXAMPLES.map((e, i) => (
          <span key={e}>
            {i > 0 && " · "}
            <a
              href="#"
              onClick={(ev) => {
                ev.preventDefault();
                setUrl(e);
              }}
            >
              {e.split("/").slice(-1)[0]}
            </a>
          </span>
        ))}
      </div>

      {busy && (
        <div className="hint">
          Fetching the model card and its config, then drafting an entry. This takes a few seconds.
        </div>
      )}

      {error && (
        <div className="finding error">
          <div className="t">Could not import that</div>
          <div className="d">{error}</div>
          <div className="fix">
            <b>Fix →</b> only huggingface.co and github.com model pages can be read. If the repo is
            private or gated, or no drafting model is set up under Settings, add the entry by hand
            instead — nothing here is required to deploy.
          </div>
        </div>
      )}

      <p className="hint" style={{ marginBottom: 0 }}>
        The VRAM figure is measured from the model's own config, not guessed by the LLM — it is what
        decides whether a deploy is allowed to start.
      </p>
    </Modal>
  );
}
