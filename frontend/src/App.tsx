import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { ThemeSwitcher } from "./components/ThemeSwitcher";
import { useLiveEvents, usePolled } from "./lib/api";
import { fmtNum } from "./lib/format";
import type { Me, RuntimeConfig, Summary } from "./types";
import { Activity } from "./pages/Activity";
import { Catalog } from "./pages/Catalog";
import { Fleet } from "./pages/Fleet";
import { Models } from "./pages/Models";
import { Proxy } from "./pages/Proxy";

/** The always-visible strip. Six numbers that answer "is the cluster fine?"
 *  without clicking anything. */
function Strip({ s }: { s: Summary | null }) {
  if (!s) return null;
  const nodeTrouble = s.nodes_unreachable > 0;
  const depTrouble = s.deployments_failed > 0 || s.deployments_degraded > 0;
  return (
    <div className="strip">
      <div className={`stat ${nodeTrouble ? "alert" : ""}`}>
        <div className="k">Nodes</div>
        <div className="v">
          {s.nodes_online}
          <small>/{s.nodes_total}</small>
          {nodeTrouble && <small> · {s.nodes_unreachable} down</small>}
        </div>
      </div>
      <div className="stat">
        <div className="k">GPUs in use</div>
        <div className="v">
          {s.gpus_busy}
          <small>/{s.gpus_total}</small>
        </div>
      </div>
      <div className="stat">
        <div className="k">VRAM</div>
        <div className="v">
          {Math.round(s.vram_used_gb)}
          <small>/{Math.round(s.vram_total_gb)} GB</small>
        </div>
      </div>
      <div className={`stat ${depTrouble ? "warn" : ""}`}>
        <div className="k">Models serving</div>
        <div className="v">
          {s.models_served}
          {depTrouble && (
            <small>
              {" "}
              · {s.deployments_failed} failed{s.deployments_degraded ? `, ${s.deployments_degraded} degraded` : ""}
            </small>
          )}
        </div>
      </div>
      <div className="stat">
        <div className="k">Throughput</div>
        <div className="v">
          {fmtNum(s.tokens_per_second, 0)}
          <small> tok/s</small>
        </div>
      </div>
      <div className={`stat ${s.requests_waiting > 0 ? "warn" : ""}`}>
        <div className="k">Requests</div>
        <div className="v">
          {s.requests_running}
          <small> running · {s.requests_waiting} queued</small>
        </div>
      </div>
      <div className={`stat ${s.litellm_reachable ? "" : "alert"}`}>
        <div className="k">Proxy</div>
        <div className="v" style={{ fontSize: 22 }}>
          {s.litellm_reachable ? "healthy" : "unreachable"}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { data: me } = usePolled<Me>("/api/auth/me", 120000);
  const { data: cfg } = usePolled<RuntimeConfig>("/api/config", 120000);
  const { data: summary, refresh } = usePolled<Summary>("/api/summary", 6000);
  useLiveEvents(["deployment", "node"], refresh);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          dgxctl <small>GPU fleet control</small>
        </div>
        <nav className="nav">
          <NavLink to="/fleet" className={({ isActive }) => (isActive ? "active" : "")}>
            Fleet
          </NavLink>
          <NavLink to="/models" className={({ isActive }) => (isActive ? "active" : "")}>
            Models
          </NavLink>
          <NavLink to="/proxy" className={({ isActive }) => (isActive ? "active" : "")}>
            Proxy
          </NavLink>
          <NavLink to="/catalog" className={({ isActive }) => (isActive ? "active" : "")}>
            Catalog
          </NavLink>
          <NavLink to="/activity" className={({ isActive }) => (isActive ? "active" : "")}>
            Activity
          </NavLink>
        </nav>
        <div className="spacer" />
        <ThemeSwitcher />
        {cfg?.simulated && (
          <span className="simbadge" title="No real hardware is being touched. Set DGXCTL_DRIVER=ssh to go live.">
            SIMULATED FLEET
          </span>
        )}
        <span className="who">
          {me ? `${me.email} · ${me.role}` : "…"}
        </span>
      </header>

      <main className="main">
        <div className="page">
          <section className="card headline">
            <h1 className="hello">
              Welcome in{me?.name ? `, ${me.name.split(" ")[0]}` : ""}
            </h1>
            <Strip s={summary} />
          </section>
        </div>
        <Routes>
          <Route path="/fleet" element={<Fleet me={me} />} />
          <Route path="/models" element={<Models me={me} />} />
          <Route path="/proxy" element={<Proxy me={me} />} />
          <Route path="/catalog" element={<Catalog me={me} />} />
          <Route path="/activity" element={<Activity />} />
          <Route path="*" element={<Navigate to="/fleet" replace />} />
        </Routes>
      </main>
    </div>
  );
}
