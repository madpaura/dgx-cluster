import { useMemo, useState } from "react";
import { AddNodeDialog } from "../components/AddNodeDialog";
import { ClusterDialog } from "../components/ClusterDialog";
import { ClusterSection } from "../components/ClusterSection";
import { DeployDialog, type ScopeTarget } from "../components/DeployDialog";
import { DeploymentDrawer } from "../components/DeploymentDrawer";
import { NodeDrawer } from "../components/NodeDrawer";
import { Confirm, useToast } from "../components/ui";
import { api, ApiError, useLiveEvents, usePolled } from "../lib/api";
import type { Cluster, Deployment, Me, ModelSpec, Node } from "../types";

/** The fleet map, organised the way you organise it. Nodes are minimal tiles —
 *  alive, how full, what it serves — grouped into clusters you define. Select
 *  tiles to deploy onto them; drag them between clusters to reorganise. */
export function Fleet({ me }: { me: Me | null }) {
  const { data: nodes, refresh } = usePolled<Node[]>("/api/nodes", 6000);
  const { data: clusters, refresh: refreshClusters } = usePolled<Cluster[]>("/api/clusters", 20000);
  // Everything, not just active: a tile must be able to say "one of these
  // failed" — which it cannot do if failures are filtered out server-side.
  const { data: deployments } = usePolled<Deployment[]>("/api/deployments?active_only=false", 6000);
  const { data: specs } = usePolled<ModelSpec[]>("/api/catalog", 60000);

  const [selection, setSelection] = useState<Set<string>>(new Set());
  // A deploy is either scoped to an explicit set of nodes (a cluster, or the
  // current selection) or unscoped. `null` means "use the selection".
  const [deploy, setDeploy] = useState<{ ids: string[] | null; label?: string } | null>(null);
  const [openDeployment, setOpenDeployment] = useState<string | null>(null);
  const [openNode, setOpenNode] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [editingCluster, setEditingCluster] = useState<Cluster | null | undefined>(undefined);
  const [deletingCluster, setDeletingCluster] = useState<Cluster | null>(null);
  const [filter, setFilter] = useState("");
  const toast = useToast();

  useLiveEvents(["node", "deployment", "cluster"], refresh);

  const isAdmin = me?.role === "admin";
  const canDeploy = isAdmin || me?.role === "deployer";

  const deploymentsByNode = useMemo(() => {
    const map = new Map<string, Deployment[]>();
    (deployments ?? []).forEach((d) => map.set(d.node_id, [...(map.get(d.node_id) ?? []), d]));
    return map;
  }, [deployments]);

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return nodes ?? [];
    return (nodes ?? []).filter(
      (n) =>
        n.name.toLowerCase().includes(q) ||
        n.cluster_name.toLowerCase().includes(q) ||
        n.gpus.some((g) => (g.model_name ?? "").toLowerCase().includes(q)) ||
        (deploymentsByNode.get(n.id) ?? []).some((d) =>
          d.served_model_name.toLowerCase().includes(q)
        ) ||
        Object.values(n.labels).some((v) => String(v).toLowerCase().includes(q))
    );
  }, [nodes, filter, deploymentsByNode]);

  /** Sections in display order, unassigned last — and only when it has nodes. */
  const sections = useMemo(() => {
    const byCluster = new Map<string | null, Node[]>();
    visible.forEach((n) => {
      const key = n.cluster_id ?? null;
      byCluster.set(key, [...(byCluster.get(key) ?? []), n]);
    });
    // Troubled boxes first inside each cluster, then by name.
    const rank = (n: Node) => (n.status === "unreachable" ? 0 : n.status === "online" ? 2 : 1);
    byCluster.forEach((list) => list.sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name)));

    const out: { cluster: Cluster | null; nodes: Node[] }[] = (clusters ?? []).map((c) => ({
      cluster: c,
      nodes: byCluster.get(c.id) ?? [],
    }));
    const loose = byCluster.get(null) ?? [];
    if (loose.length || !(clusters ?? []).length) out.push({ cluster: null, nodes: loose });
    return out;
  }, [visible, clusters]);

  const selectedNodes = useMemo(
    () => (nodes ?? []).filter((n) => selection.has(n.id)),
    [nodes, selection]
  );

  const scope: ScopeTarget[] = useMemo(() => {
    if (!deploy) return [];
    const ids = new Set(deploy.ids ?? [...selection]);
    return (nodes ?? [])
      .filter((n) => ids.has(n.id))
      .map((n) => ({ node_id: n.id, node_name: n.name }));
  }, [deploy, selection, nodes]);

  function toggleNode(id: string) {
    setSelection((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function selectAll(ids: string[], on: boolean) {
    setSelection((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
      return next;
    });
  }

  async function moveNodes(nodeIds: string[], clusterId: string | null) {
    try {
      if (clusterId) await api.post(`/api/clusters/${clusterId}/nodes`, { node_ids: nodeIds });
      else await api.post("/api/clusters/unassign", { node_ids: nodeIds });
      const target = clusterId ? (clusters ?? []).find((c) => c.id === clusterId)?.name : "Unassigned";
      toast(`Moved ${nodeIds.length} node${nodeIds.length === 1 ? "" : "s"} to ${target}`);
      refresh();
      refreshClusters();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    }
  }

  return (
    <div className="page">
      <div className="flex mb wrap">
        <input
          placeholder="Filter nodes, clusters, models, labels…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ maxWidth: 300 }}
        />
        <span className="hint">
          {visible.length} node{visible.length === 1 ? "" : "s"}
          {clusters?.length ? ` · ${clusters.length} cluster${clusters.length === 1 ? "" : "s"}` : ""}
        </span>
        <div className="right flex">
          {isAdmin && (
            <>
              <button className="btn" onClick={() => setEditingCluster(null)}>
                + Cluster
              </button>
              <button className="btn" onClick={() => setAdding(true)}>
                + Node
              </button>
            </>
          )}
          <button
            className="btn primary"
            disabled={!canDeploy}
            onClick={() => setDeploy({ ids: [] })}
          >
            Deploy a model
          </button>
        </div>
      </div>

      {!nodes ? (
        <div className="empty">Loading fleet…</div>
      ) : visible.length === 0 && filter ? (
        <div className="empty">No nodes match “{filter}”.</div>
      ) : (
        <div className="clusters-rail">
          {sections.map(({ cluster, nodes: list }) => (
            <ClusterSection
              key={cluster?.id ?? "unassigned"}
              cluster={cluster}
              nodes={list}
              deploymentsByNode={deploymentsByNode}
              selection={selection}
              isAdmin={!!isAdmin}
              onToggleNode={toggleNode}
              onOpenNode={setOpenNode}
              onSelectAll={selectAll}
              onDropNodes={moveNodes}
              onRename={(c) => setEditingCluster(c)}
              onDelete={(c) => setDeletingCluster(c)}
              onDeployHere={(ids) =>
                setDeploy({ ids, label: cluster ? `cluster ${cluster.name}` : "unassigned nodes" })
              }
            />
          ))}
        </div>
      )}

      {selection.size > 0 && (
        <div className="selbar">
          <strong>
            {selection.size} node{selection.size === 1 ? "" : "s"}
          </strong>
          <span className="hint">{selectedNodes.map((n) => n.name).join(", ")}</span>
          <div className="right flex">
            {isAdmin && (
              <select
                value=""
                style={{ width: 190 }}
                onChange={(e) => {
                  const v = e.target.value;
                  if (!v) return;
                  if (v === "__new__") setEditingCluster(null);
                  else moveNodes([...selection], v === "__none__" ? null : v);
                  e.currentTarget.value = "";
                }}
              >
                <option value="">Move to cluster…</option>
                {(clusters ?? []).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
                <option value="__none__">Unassigned</option>
                <option value="__new__">New cluster…</option>
              </select>
            )}
            <button className="btn ghost" onClick={() => setSelection(new Set())}>
              Clear
            </button>
            <button
              className="btn primary"
              disabled={!canDeploy}
              onClick={() => setDeploy({ ids: null })}
            >
              Deploy here
            </button>
          </div>
        </div>
      )}

      {deploy && (
        <DeployDialog
          specs={specs ?? []}
          scope={scope}
          scopeLabel={deploy.label}
          onClose={() => setDeploy(null)}
          onDeployed={() => {
            setSelection(new Set());
            refresh();
          }}
        />
      )}
      {adding && (
        <AddNodeDialog
          clusters={clusters ?? []}
          onClose={() => setAdding(false)}
          onAdded={(n) => {
            refresh();
            refreshClusters();
            toast(
              n.status === "online"
                ? `${n.name} registered — ${n.gpus.length} GPUs found`
                : `${n.name} registered but unreachable: ${n.last_error.slice(0, 90)}`,
              n.status !== "online"
            );
          }}
        />
      )}
      {editingCluster !== undefined && (
        <ClusterDialog
          existing={editingCluster}
          nodeIds={editingCluster === null ? [...selection] : undefined}
          onClose={() => setEditingCluster(undefined)}
          onSaved={() => {
            setSelection(new Set());
            refresh();
            refreshClusters();
          }}
        />
      )}
      {deletingCluster && (
        <Confirm
          title={`Delete cluster “${deletingCluster.name}”?`}
          danger
          confirmLabel="Delete cluster"
          body={
            <p>
              Its {deletingCluster.node_count} node{deletingCluster.node_count === 1 ? "" : "s"} move to
              Unassigned. Nothing is stopped and no machine is touched — a cluster is only a label.
            </p>
          }
          onCancel={() => setDeletingCluster(null)}
          onConfirm={async () => {
            try {
              await api.del(`/api/clusters/${deletingCluster.id}`);
              toast(`Deleted ${deletingCluster.name}`);
              refresh();
              refreshClusters();
            } catch (e) {
              toast(e instanceof ApiError ? e.message : String(e), true);
            } finally {
              setDeletingCluster(null);
            }
          }}
        />
      )}
      {openDeployment && (
        <DeploymentDrawer
          deploymentId={openDeployment}
          onClose={() => setOpenDeployment(null)}
          onChanged={refresh}
        />
      )}
      {openNode && (
        <NodeDrawer
          nodeId={openNode}
          isAdmin={!!isAdmin}
          onClose={() => setOpenNode(null)}
          onChanged={refresh}
          onOpenDeployment={(id) => {
            setOpenNode(null);
            setOpenDeployment(id);
          }}
        />
      )}
    </div>
  );
}
