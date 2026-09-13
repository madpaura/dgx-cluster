import { useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Cluster, Node } from "../types";
import { Modal, useToast } from "./ui";

export function AddNodeDialog({
  clusters,
  onClose,
  onAdded,
}: {
  clusters: Cluster[];
  onClose: () => void;
  onAdded: (n: Node) => void;
}) {
  const [name, setName] = useState("");
  const [hostname, setHostname] = useState("");
  const [port, setPort] = useState(22);
  const [user, setUser] = useState("");
  const [kind, setKind] = useState<"dgx" | "workstation">("dgx");
  const [clusterId, setClusterId] = useState(clusters[0]?.id ?? "");
  const [labels, setLabels] = useState("");
  const [busy, setBusy] = useState(false);
  // A node that has never met dgxctl refuses its key. Rather than sending the
  // operator away to run ssh-copy-id, ask for the password once and install it.
  const [needsKey, setNeedsKey] = useState<Node | null>(null);
  const [password, setPassword] = useState("");
  const [keyError, setKeyError] = useState("");
  const toast = useToast();

  async function save() {
    setBusy(true);
    try {
      const parsed: Record<string, string> = {};
      labels
        .split(/[\s,]+/)
        .filter(Boolean)
        .forEach((pair) => {
          const [k, v] = pair.split("=");
          if (k && v) parsed[k] = v;
        });
      const node = await api.post<Node>("/api/nodes", {
        name: name.trim(),
        hostname: hostname.trim() || name.trim(),
        ssh_port: port,
        ssh_user: user.trim(),
        kind,
        cluster_id: clusterId || null,
        labels: parsed,
      });
      if (node.status === "unreachable" && /publickey|permission denied/i.test(node.last_error)) {
        setNeedsKey(node);
        return;
      }
      onAdded(node);
      onClose();
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  }

  async function authorize() {
    if (!needsKey) return;
    setBusy(true);
    setKeyError("");
    try {
      const node = await api.post<Node>(`/api/nodes/${needsKey.id}/authorize`, { password });
      setPassword("");
      onAdded(node);
      onClose();
    } catch (e) {
      setKeyError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (needsKey) {
    return (
      <Modal
        title={`Authorize dgxctl on ${needsKey.name}`}
        width={560}
        onClose={() => {
          setPassword("");
          onAdded(needsKey);
          onClose();
        }}
        footer={
          <>
            <button
              className="btn"
              onClick={() => {
                setPassword("");
                onAdded(needsKey);
                onClose();
              }}
            >
              Skip for now
            </button>
            <button className="btn primary" disabled={busy || !password} onClick={authorize}>
              {busy ? "Installing…" : "Install key"}
            </button>
          </>
        }
      >
        <p className="hint mb">
          <strong>{needsKey.name}</strong> is reachable but refused our key — expected on a
          machine dgxctl has not managed before. Give the password for{" "}
          <code>{needsKey.ssh_port === 22 ? "" : `port ${needsKey.ssh_port}, `}</code>
          the SSH user, and the control server's public key is appended to its{" "}
          <code>authorized_keys</code> once. It is used for this and nothing else — not
          stored, not logged.
        </p>
        <label className="f">
          <span>SSH password</span>
          <input
            autoFocus
            type="password"
            value={password}
            autoComplete="off"
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && password && authorize()}
          />
        </label>
        {keyError && (
          <div className="finding error">
            <div className="t">Could not install the key</div>
            <div className="d">{keyError}</div>
          </div>
        )}
        <p className="hint">
          Prefer not to? Skip, run{" "}
          <code>ssh-copy-id -i secrets/fleet_key.pub {needsKey.hostname}</code> yourself,
          then use Probe now on the node.
        </p>
      </Modal>
    );
  }

  return (
    <Modal
      title="Register a GPU node"
      width={620}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={busy || !name.trim()} onClick={save}>
            {busy ? "Probing…" : "Add and probe"}
          </button>
        </>
      }
    >
      <p className="hint mb">
        The control server reaches nodes over SSH and drives docker there. Make sure its public key is in{" "}
        <code>~/.ssh/authorized_keys</code> for the user below, and that the user can run <code>docker</code>.
      </p>
      <div className="row">
        <label className="f">
          <span>Name</span>
          <input value={name} placeholder="dgx-05" onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="f">
          <span>Hostname or IP</span>
          <input value={hostname} placeholder="dgx-05.lan" onChange={(e) => setHostname(e.target.value)} />
        </label>
      </div>
      <div className="row">
        <label className="f">
          <span>SSH port</span>
          <input type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} />
        </label>
        <label className="f">
          <span>SSH user</span>
          <input value={user} placeholder="(server default)" onChange={(e) => setUser(e.target.value)} />
        </label>
        <label className="f">
          <span>Kind</span>
          <select value={kind} onChange={(e) => setKind(e.target.value as "dgx" | "workstation")}>
            <option value="dgx">DGX</option>
            <option value="workstation">Workstation</option>
          </select>
        </label>
      </div>
      <label className="f">
        <span>Cluster</span>
        <select value={clusterId} onChange={(e) => setClusterId(e.target.value)}>
          <option value="">Unassigned</option>
          {clusters.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      <label className="f">
        <span>Labels (key=value, space separated)</span>
        <input value={labels} placeholder="rack=r2 net=ib team=research" onChange={(e) => setLabels(e.target.value)} />
      </label>
    </Modal>
  );
}
