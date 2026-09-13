import { useCallback, useEffect, useRef, useState } from "react";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    headers: init?.body ? { "content-type": "application/json" } : undefined,
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export const api = {
  get: <T,>(p: string) => request<T>(p),
  post: <T,>(p: string, body?: unknown) =>
    request<T>(p, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T,>(p: string, body: unknown) =>
    request<T>(p, { method: "PATCH", body: JSON.stringify(body) }),
  put: <T,>(p: string, body: unknown) => request<T>(p, { method: "PUT", body: JSON.stringify(body) }),
  del: (p: string) => request<void>(p, { method: "DELETE" }),
};

/** Poll an endpoint, and re-fetch immediately whenever the websocket says
 *  something changed. Keeps the UI live without a state-sync framework. */
/** Set whenever any poll fails, cleared when one succeeds. The dashboard shows
 *  live numbers; when they stop being live that has to be visible, or a frozen
 *  screen reads as a healthy fleet. */
const staleListeners = new Set<(since: number | null) => void>();
let firstFailureAt: number | null = null;

function reportPoll(ok: boolean) {
  const was = firstFailureAt;
  if (ok) firstFailureAt = null;
  else if (firstFailureAt === null) firstFailureAt = Date.now();
  if (was !== firstFailureAt) staleListeners.forEach((fn) => fn(firstFailureAt));
}

export function useStaleSince(): number | null {
  const [since, setSince] = useState<number | null>(firstFailureAt);
  useEffect(() => {
    staleListeners.add(setSince);
    return () => {
      staleListeners.delete(setSince);
    };
  }, []);
  return since;
}

export function usePolled<T>(path: string, intervalMs = 5000, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const value = await api.get<T>(path);
      reportPoll(true);
      if (alive.current) {
        setData(value);
        setError(null);
      }
    } catch (e) {
      // A 4xx is an answer; only a failure to reach the server is staleness.
      reportPoll(e instanceof ApiError && e.status > 0 ? true : false);
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);

  useEffect(() => {
    alive.current = true;
    refresh();
    const id = window.setInterval(refresh, intervalMs);
    return () => {
      alive.current = false;
      window.clearInterval(id);
    };
  }, [refresh, intervalMs]);

  return { data, error, loading, refresh };
}

type WsMessage = { topic: string; data: Record<string, unknown> };

/** Single shared websocket; components subscribe to topics they care about. */
const listeners = new Set<(m: WsMessage) => void>();
let socket: WebSocket | null = null;
let retry = 0;

function ensureSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/api/ws`);
  socket.onopen = () => {
    retry = 0;
  };
  socket.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data) as WsMessage;
      listeners.forEach((fn) => fn(msg));
    } catch {
      /* ignore malformed frame */
    }
  };
  socket.onclose = () => {
    socket = null;
    retry = Math.min(retry + 1, 6);
    setTimeout(ensureSocket, 500 * 2 ** retry);
  };
  socket.onerror = () => socket?.close();
}

export function useLiveEvents(topics: string[], onEvent: () => void, throttleMs = 1200) {
  const last = useRef(0);
  const cb = useRef(onEvent);
  cb.current = onEvent;

  useEffect(() => {
    ensureSocket();
    const fn = (m: WsMessage) => {
      if (!topics.includes(m.topic)) return;
      const now = Date.now();
      if (now - last.current < throttleMs) return;
      last.current = now;
      cb.current();
    };
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  }, [topics.join(","), throttleMs]);
}
