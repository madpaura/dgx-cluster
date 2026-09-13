import type { Deployment, Node } from "../types";
import { NODE_TONE, ago, fmtNum, parseTs } from "../lib/format";
import { Pill } from "./ui";

/** One node, reduced to what you need at a glance: is it alive, how full is it,
 *  and what is it serving. Per-GPU detail lives one click away in the drawer —
 *  a wall of GPU tiles does not scale past a handful of boxes. */
export function NodeTile({
  node,
  deployments,
  selected,
  draggable,
  onToggle,
  onOpen,
  onDragStart,
}: {
  node: Node;
  deployments: Deployment[];
  selected: boolean;
  draggable: boolean;
  onToggle: () => void;
  onOpen: () => void;
  onDragStart: (e: React.DragEvent) => void;
}) {
  const offline = node.status === "unreachable";
  const busy = node.gpus.filter((g) => g.tenants.length).length;
  const total = node.gpus.length;
  const sharing = node.gpus.some((g) => g.tenants.length > 1);
  const usedMb = node.gpus.reduce((a, g) => a + g.memory_used_mb, 0);
  const totalMb = node.gpus.reduce((a, g) => a + g.memory_total_mb, 0);
  const gpuName = node.gpus[0]?.name.replace("NVIDIA ", "").replace(" Generation", "") ?? "";
  const live = deployments.filter((d) => !["stopped", "failed"].includes(d.status));
  const models = [...new Set(live.map((d) => d.served_model_name))];
  const tps = live.reduce((a, d) => a + (d.last_metrics?.gen_tps ?? 0), 0);
  const degraded = live.filter((d) => d.status === "degraded").length;
  // Old failures are history, not news — only surface fresh ones on the map.
  const failed = deployments.filter(
    (d) => d.status === "failed" && Date.now() - parseTs(d.created_at).getTime() < 3_600_000
  ).length;
  const hot = node.gpus.some((g) => g.temperature_c >= 85 || g.ecc_errors > 0);

  return (
    <article
      className={`tile ${selected ? "sel" : ""} ${offline ? "offline" : ""}`}
      draggable={draggable}
      onDragStart={onDragStart}
      onClick={onToggle}
      title={
        offline
          ? `${node.name} — unreachable (${ago(node.last_seen)})`
          : `${node.name} · ${node.hostname}\n${total}× ${gpuName}\n` +
            `${busy} of ${total} GPUs in use` +
            (models.length ? `\nserving: ${models.join(", ")}` : "\nnothing deployed")
      }
    >
      <div className="tile-top">
        <span className="tile-name">{node.name}</span>
        <Pill tone={NODE_TONE[node.status]}>{node.status}</Pill>
        <button
          className="btn ghost sm tile-more"
          title="Node details, GPUs and diagnostics"
          onClick={(e) => {
            e.stopPropagation();
            onOpen();
          }}
        >
          ⋯
        </button>
      </div>

      <div className="tile-hw">{total ? `${total}× ${gpuName}` : node.hostname}</div>

      {/* One segment per GPU, filled in proportion to the VRAM claimed on it.
          A card holding two models reads as fuller than one holding a small
          model, which is the distinction that matters now that cards are
          shared. */}
      <div className="slots" aria-label={`${busy} of ${total} GPUs in use`}>
        {node.gpus.map((g) => {
          const claimed = g.memory_total_mb
            ? Math.min(100, (g.reserved_mb / g.memory_total_mb) * 100)
            : 0;
          return (
            <i key={g.index} className={g.tenants.length ? "on" : "off"}
               title={
                 g.tenants.length
                   ? `GPU ${g.index}: ${g.tenants.map((t) => t.model_name).join(", ")}`
                   : `GPU ${g.index}: free`
               }>
              <b style={{ width: `${claimed}%` }} />
            </i>
          );
        })}
        {total === 0 && <i className="off" style={{ flex: 1 }} />}
      </div>

      <div className="tile-foot">
        {offline ? (
          <span className="err-text">{node.last_error?.slice(0, 44) || "unreachable"}</span>
        ) : (
          <>
            <span title={sharing ? "some GPUs hold more than one model" : undefined}>
              {busy}/{total} GPUs{sharing ? " ·shared" : ""}
            </span>
            <span className="dot-sep">·</span>
            <span>
              {models.length} model{models.length === 1 ? "" : "s"}
            </span>
            {tps > 0 && (
              <>
                <span className="dot-sep">·</span>
                <span>{fmtNum(tps, 0)} tok/s</span>
              </>
            )}
            <span className="right mono">
              {totalMb ? `${Math.round(usedMb / 1024)}/${Math.round(totalMb / 1024)}G` : ""}
            </span>
          </>
        )}
      </div>

      {(failed > 0 || degraded > 0 || hot) && !offline && (
        <div className="tile-warn">
          {[
            failed > 0 && `${failed} failed to start`,
            degraded > 0 && `${degraded} degraded`,
            hot && "GPU needs attention",
          ]
            .filter(Boolean)
            .join(" · ")}
        </div>
      )}
    </article>
  );
}
