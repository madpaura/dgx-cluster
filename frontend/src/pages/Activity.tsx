import { useState } from "react";
import { DeploymentDrawer } from "../components/DeploymentDrawer";
import { Pill } from "../components/ui";
import { useLiveEvents, usePolled } from "../lib/api";
import { clock } from "../lib/format";
import type { AuditRow, EventRow } from "../types";

const TONE: Record<string, string> = { error: "err", warning: "warn", info: "info" };

/** What the fleet did, and what people did to it. First stop when someone asks
 *  "why did the model go down at 3am". */
export function Activity() {
  const [tab, setTab] = useState<"events" | "audit">("events");
  const [open, setOpen] = useState<string | null>(null);
  const { data: events, refresh } = usePolled<EventRow[]>("/api/events?limit=200", 8000);
  const { data: audit } = usePolled<AuditRow[]>("/api/audit?limit=200", 15000);
  useLiveEvents(["event"], refresh);

  return (
    <div className="page">
      <section className="card">
        <div className="tabs">
          <button className={tab === "events" ? "active" : ""} onClick={() => setTab("events")}>
            Fleet events
          </button>
          <button className={tab === "audit" ? "active" : ""} onClick={() => setTab("audit")}>
            Audit log
          </button>
        </div>

        {tab === "events" ? (
          <table className="t">
            <thead>
              <tr>
                <th style={{ width: 90 }}>Time</th>
                <th style={{ width: 90 }}>Level</th>
                <th style={{ width: 100 }}>Source</th>
                <th>What happened</th>
              </tr>
            </thead>
            <tbody>
              {(events ?? []).map((e) => (
                <tr
                  key={e.id}
                  className={e.source === "deployment" ? "clickable" : ""}
                  onClick={() => e.source === "deployment" && setOpen(e.source_id)}
                >
                  <td className="sub">{clock(e.ts)}</td>
                  <td>
                    <Pill tone={TONE[e.severity] ?? "muted"}>{e.severity}</Pill>
                  </td>
                  <td className="sub">{e.source}</td>
                  <td>{e.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="t">
            <thead>
              <tr>
                <th style={{ width: 90 }}>Time</th>
                <th style={{ width: 200 }}>Who</th>
                <th style={{ width: 160 }}>Action</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {(audit ?? []).map((a) => (
                <tr key={a.id}>
                  <td className="sub">{clock(a.ts)}</td>
                  <td className="sub">{a.actor}</td>
                  <td>
                    <code>{a.action}</code>
                  </td>
                  <td>
                    {a.summary} {!a.ok && <Pill tone="err">failed</Pill>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      {open && <DeploymentDrawer deploymentId={open} onClose={() => setOpen(null)} onChanged={refresh} />}
    </div>
  );
}
