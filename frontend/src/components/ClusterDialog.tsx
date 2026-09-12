import { useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Cluster } from "../types";
import { Modal, useToast } from "./ui";

/** Create or rename a cluster. Deliberately tiny — a cluster is a label, and
 *  nothing about it should feel like a commitment. */
export function ClusterDialog({
  existing,
  nodeIds,
  onClose,
  onSaved,
}: {
  existing: Cluster | null;
  nodeIds?: string[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(existing?.name ?? "");
  const [description, setDescription] = useState(existing?.description ?? "");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  async function save() {
    setBusy(true);
    try {
      if (existing) {
        await api.patch(`/api/clusters/${existing.id}`, { name: name.trim(), description });
        toast(`Renamed to ${name.trim()}`);
      } else {
        await api.post("/api/clusters", {
          name: name.trim(),
          description,
          node_ids: nodeIds ?? [],
        });
        toast(
          nodeIds?.length
            ? `Created ${name.trim()} with ${nodeIds.length} node${nodeIds.length === 1 ? "" : "s"}`
            : `Created ${name.trim()}`
        );
      }
      onSaved();
      onClose();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title={existing ? `Rename ${existing.name}` : "New cluster"}
      width={520}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={busy || !name.trim()} onClick={save}>
            {existing ? "Save" : "Create"}
          </button>
        </>
      }
    >
      <label className="f">
        <span>Name</span>
        <input
          autoFocus
          value={name}
          placeholder="Machine room A"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && name.trim() && save()}
        />
      </label>
      <label className="f">
        <span>Description</span>
        <input
          value={description}
          placeholder="What is in here, or who owns it"
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>
      {!existing && nodeIds && nodeIds.length > 0 && (
        <p className="hint">
          The {nodeIds.length} selected node{nodeIds.length === 1 ? "" : "s"} will be moved into it.
        </p>
      )}
      <p className="hint">
        Clusters are how you organise the fleet in this dashboard. They do not constrain scheduling —
        a model can still be placed anywhere unless you scope the deploy.
      </p>
    </Modal>
  );
}
