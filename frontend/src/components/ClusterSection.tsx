import { useState } from "react";
import type { Cluster, Deployment, Node } from "../types";
import { fmtNum } from "../lib/format";
import { NodeTile } from "./NodeTile";

export const UNASSIGNED = "__unassigned__";

/** A cluster is a drop target. Dragging nodes between sections is the fastest
 *  way to reorganise a fleet; the selection bar's move menu does the same thing
 *  for anyone not using a mouse. */
export function ClusterSection({
  cluster,
  nodes,
  deploymentsByNode,
  selection,
  isAdmin,
  onToggleNode,
  onOpenNode,
  onSelectAll,
  onDropNodes,
  onRename,
  onDelete,
  onDeployHere,
}: {
  cluster: Cluster | null; // null = the Unassigned bucket
  nodes: Node[];
  deploymentsByNode: Map<string, Deployment[]>;
  selection: Set<string>;
  isAdmin: boolean;
  onToggleNode: (id: string) => void;
  onOpenNode: (id: string) => void;
  onSelectAll: (ids: string[], on: boolean) => void;
  onDropNodes: (nodeIds: string[], clusterId: string | null) => void;
  onRename: (cluster: Cluster) => void;
  onDelete: (cluster: Cluster) => void;
  onDeployHere: (nodeIds: string[]) => void;
}) {
  const [over, setOver] = useState(false);
  const id = cluster?.id ?? null;

  const online = nodes.filter((n) => n.status === "online").length;
  const down = nodes.filter((n) => n.status === "unreachable").length;
  const gpus = nodes.reduce((a, n) => a + n.gpus.length, 0);
  const busy = nodes.reduce((a, n) => a + n.gpus.filter((g) => g.deployment_id).length, 0);
  const deps = nodes
    .flatMap((n) => deploymentsByNode.get(n.id) ?? [])
    .filter((d) => !["stopped", "failed"].includes(d.status));
  const models = new Set(deps.map((d) => d.served_model_name)).size;
  const tps = deps.reduce((a, d) => a + (d.last_metrics?.gen_tps ?? 0), 0);
  const ids = nodes.map((n) => n.id);
  const allSelected = ids.length > 0 && ids.every((i) => selection.has(i));

  return (
    <section
      className={`cluster ${over ? "dropping" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        if (!over) setOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node2)) setOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        try {
          const payload = JSON.parse(e.dataTransfer.getData("text/plain")) as { nodeIds?: string[] };
          const moving = (payload.nodeIds ?? []).filter(
            (n) => !ids.includes(n) // already here: nothing to do
          );
          if (moving.length) onDropNodes(moving, id);
        } catch {
          /* something else was dropped */
        }
      }}
    >
      <header className="cluster-head">
        <div className="cluster-title">
          <label className="flex" style={{ gap: 7, cursor: "pointer", minWidth: 0 }} onClick={(e) => e.stopPropagation()}>
            <input
              type="checkbox"
              style={{ width: "auto" }}
              checked={allSelected}
              disabled={!ids.length}
              onChange={(e) => onSelectAll(ids, e.target.checked)}
            />
            <span className="cluster-name">{cluster ? cluster.name : "Unassigned"}</span>
          </label>

          <div className="right flex" style={{ gap: 2 }}>
            <button
              className="btn ghost sm"
              disabled={!ids.length}
              title="Deploy a model onto this cluster"
              onClick={() => onDeployHere(ids)}
            >
              Deploy
            </button>
            {cluster && isAdmin && (
              <>
                <button className="btn ghost sm" title="Rename" onClick={() => onRename(cluster)}>
                  ✎
                </button>
                <button className="btn ghost sm" title="Delete cluster" onClick={() => onDelete(cluster)}>
                  ✕
                </button>
              </>
            )}
          </div>
        </div>

        {cluster?.description && <div className="cluster-desc">{cluster.description}</div>}

        <div className="cluster-stats mono">
          {nodes.length} node{nodes.length === 1 ? "" : "s"}
          {down > 0 && <b className="err-text"> · {down} down</b>}
          {gpus > 0 && ` · ${busy}/${gpus} GPUs`}
          {models > 0 && ` · ${models} models`}
          {tps > 0 && ` · ${fmtNum(tps, 0)} tok/s`}
          {online === 0 && nodes.length > 0 && " · none online"}
        </div>
      </header>

      {nodes.length === 0 ? (
        <div className="cluster-empty">
          {cluster ? "Drag nodes here, or use the move menu." : "Every node belongs to a cluster."}
        </div>
      ) : (
        <div className="tiles">
          {nodes.map((n) => (
            <NodeTile
              key={n.id}
              node={n}
              deployments={deploymentsByNode.get(n.id) ?? []}
              selected={selection.has(n.id)}
              draggable={isAdmin}
              onToggle={() => onToggleNode(n.id)}
              onOpen={() => onOpenNode(n.id)}
              onDragStart={(e) => {
                // Dragging a selected node moves the whole selection.
                const payload = selection.has(n.id) ? [...selection] : [n.id];
                e.dataTransfer.setData("text/plain", JSON.stringify({ nodeIds: payload }));
                e.dataTransfer.effectAllowed = "move";
              }}
            />
          ))}
        </div>
      )}
    </section>
  );
}

// `Node` is our domain type in this file, so reach for the DOM one explicitly.
type Node2 = globalThis.Node;
