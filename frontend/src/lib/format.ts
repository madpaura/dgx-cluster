import type { DeployStatus, NodeStatus } from "../types";

export const gb = (mb: number) => mb / 1024;

export function fmtGb(mb: number, digits = 0) {
  return `${(mb / 1024).toFixed(digits)}G`;
}

export function fmtNum(n: number | undefined, digits = 0) {
  if (n === undefined || n === null || Number.isNaN(n)) return "–";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${(n / 1000).toFixed(1)}k`;
  return n.toFixed(digits);
}

/** Parse a server timestamp. A string with no timezone marker is UTC — some
 *  backends (SQLite in dev) drop the offset on the way out. */
export function parseTs(iso: string): Date {
  const hasZone = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(iso);
  return new Date(hasZone ? iso : `${iso}Z`);
}

export function ago(iso: string | null | undefined) {
  if (!iso) return "never";
  const s = Math.max(0, (Date.now() - parseTs(iso).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function clock(iso: string) {
  return parseTs(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false, // keeps the column one line wide and column-aligned
  });
}

export const DEPLOY_TONE: Record<DeployStatus, "ok" | "warn" | "err" | "info" | "muted" | "busy"> = {
  pending: "info",
  pulling: "busy",
  starting: "info",
  healthy: "ok",
  degraded: "warn",
  failed: "err",
  stopping: "muted",
  stopped: "muted",
};

export const NODE_TONE: Record<NodeStatus, "ok" | "warn" | "err" | "muted" | "busy"> = {
  online: "ok",
  unknown: "muted",
  unreachable: "err",
  draining: "warn",
  maintenance: "busy",
};
