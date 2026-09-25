import React, { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { Bell, Globe2, Layers, FolderOpen, Activity, BrainCircuit, Users, LogOut, ChevronDown } from "lucide-react";
import { api } from "./api.js";
import { AuthProvider, useAuth } from "./auth.jsx";
import { navigate, useRoute } from "./router.js";
import AdminPage from "./pages/AdminPage.jsx";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import MonitoringPage from "./pages/MonitoringPage.jsx";
import ProjectsPage from "./pages/ProjectsPage.jsx";
import StudioPage from "./pages/StudioPage.jsx";

// Cesium is large: the globe explorer is loaded on first visit.
const ExplorePage = lazy(() => import("./pages/ExplorePage.jsx"));

const DEFAULT_AOI = [77.5, 12.9, 77.56, 12.96]; // Bengaluru outskirts, ~6.5 km square

function Shell() {
  const { user, ready, logout, can } = useAuth();
  const route = useRoute();
  const [health, setHealth] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [aoi, setAoi] = useState(DEFAULT_AOI);
  const [place, setPlace] = useState("Bengaluru outskirts");
  const [unread, setUnread] = useState(0);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    api("/health").then(setHealth).catch(() => setHealth({ status: "down" }));
    api("/workflows").then(setCatalog).catch(() => {});
  }, []);

  const refreshAlerts = useCallback(() => {
    if (user) api("/alerts").then((a) => setUnread(a.unread)).catch(() => {});
    else setUnread(0);
  }, [user]);
  useEffect(() => {
    refreshAlerts();
    const t = setInterval(refreshAlerts, 30000);
    return () => clearInterval(t);
  }, [refreshAlerts]);

  const [page, arg] = route;
  const links = [
    ["", "Analyze", true, Layers],
    ["explore", "Explore", true, Globe2],
    ["projects", "Projects", !!user, FolderOpen],
    ["monitoring", "Monitoring", can("official"), Activity],
    ["studio", "Models", can("admin"), BrainCircuit],
    ["admin", "Users", can("admin"), Users],
  ];
  const active = (id) => (id === "" ? !page || page === "run" || page === "share" : page === id);
  const guard = (role, el) => (!ready ? null : can(role) ? el : <div className="page narrow"><div className="card">This page needs the <b>{role}</b> role. {!user && <a href="#/login">Sign in</a>}</div></div>);

  let body;
  if (page === "login") body = <LoginPage />;
  else if (page === "explore") body = (
    <Suspense fallback={<div className="map-loading">Loading the globe…</div>}>
      <ExplorePage setAoi={setAoi} setPlace={setPlace} />
    </Suspense>
  );
  else if (page === "projects") body = guard("public", <ProjectsPage />);
  else if (page === "monitoring") body = guard("official", <MonitoringPage catalog={catalog} aoi={aoi} place={place} onAlertsChanged={refreshAlerts} />);
  else if (page === "studio") body = guard("admin", <StudioPage />);
  else if (page === "admin") body = guard("admin", <AdminPage />);
  else body = (
    <AnalyzePage runId={page === "run" ? arg : null} shareToken={page === "share" ? arg : null} aoi={aoi} setAoi={setAoi}
      place={place} setPlace={setPlace} health={health} catalog={catalog} />
  );

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="logo" aria-hidden="true"><Globe2 size={14} strokeWidth={2.25} /></span>
          <span className="brand-name">GEO-VLA</span>
          <span className="brand-sub">Geospatial analysis</span>
        </a>
        <nav>
          {links.filter(([, , show]) => show).map(([id, label, , Icon]) => (
            <a key={id} href={`#/${id}`} className={active(id) ? "active" : ""}>
              <Icon size={15} strokeWidth={1.75} aria-hidden="true" />
              {label}
            </a>
          ))}
        </nav>
        <div className="topbar-right">
          {health?.status === "ok" && (
            <div className="sys-status" aria-label="System status">
              <span title={health.imagery_source === "synthetic" ? "No satellite credentials: synthetic imagery" : `Sentinel-2 via ${health.imagery_source}`}>
                <i className={`dot ${health.data_mode === "live" ? "ok" : "warn"}`} />
                {health.data_mode === "live" ? `S2 ${health.imagery_source}` : "S2 demo"}
              </span>
              <span title={health.model || "Rule-based planner"}>
                <i className={`dot ${health.planner === "llm" ? "ok" : "warn"}`} />
                {health.planner === "llm" ? `LLM ${health.provider}` : "LLM off"}
              </span>
              <span title="Active land-cover model">
                <i className={`dot ${health.checkpoints?.classifier ? "ok" : "warn"}`} />
                CNN {health.checkpoints?.classifier ?? "none"}
              </span>
            </div>
          )}
          {health?.status === "down" && <span className="chip bad">API offline</span>}
          {user && can("official") && (
            <button className="bell" onClick={() => navigate("/monitoring")} aria-label={`${unread} unread alerts`}>
              <Bell size={16} strokeWidth={1.75} />{unread > 0 && <span className="bell-count">{unread}</span>}
            </button>
          )}
          {user ? (
            <div className="user-menu">
              <button onClick={() => setMenu((m) => !m)} aria-expanded={menu}>
                {user.full_name || user.username} <span className="role">{user.role}</span> <ChevronDown size={14} />
              </button>
              {menu && (
                <div className="menu" onMouseLeave={() => setMenu(false)}>
                  <div className="muted small">{user.organization || user.username}</div>
                  <button onClick={() => { setMenu(false); logout(); navigate("/"); }}><LogOut size={14} /> Sign out</button>
                </div>
              )}
            </div>
          ) : (
            ready && <a className="button" href="#/login">Sign in</a>
          )}
        </div>
      </header>
      {body}
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
