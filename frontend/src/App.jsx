import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { AuthProvider, useAuth } from "./auth.jsx";
import { navigate, useRoute } from "./router.js";
import AdminPage from "./pages/AdminPage.jsx";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import MonitoringPage from "./pages/MonitoringPage.jsx";
import ProjectsPage from "./pages/ProjectsPage.jsx";
import StudioPage from "./pages/StudioPage.jsx";

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
    ["", "Analyze", true],
    ["projects", "Projects", !!user],
    ["monitoring", "Monitoring", can("official")],
    ["studio", "Training studio", can("admin")],
    ["admin", "Users", can("admin")],
  ];
  const active = (id) => (id === "" ? !page || page === "run" || page === "share" : page === id);
  const guard = (role, el) => (!ready ? null : can(role) ? el : <div className="page narrow"><div className="card">This page needs the <b>{role}</b> role. {!user && <a href="#/login">Sign in</a>}</div></div>);

  let body;
  if (page === "login") body = <LoginPage />;
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
          <span className="logo" aria-hidden="true">◆</span> Geo-VLA<span className="brand-sub">Geospatial AI</span>
        </a>
        <nav>
          {links.filter(([, , show]) => show).map(([id, label]) => (
            <a key={id} href={`#/${id}`} className={active(id) ? "active" : ""}>{label}</a>
          ))}
        </nav>
        <div className="topbar-right">
          {health?.status === "ok" && (
            <span className="chips">
              <span className={`chip ${health.planner === "llm" ? "ok" : "warn"}`} title={health.model || "Planner"}>{health.planner === "llm" ? `AI · ${health.provider}` : "offline planner"}</span>
              <span className={`chip ${health.data_mode === "live" ? "ok" : "warn"}`} title="Data source">{health.data_mode === "live" ? "live data" : "demo data"}</span>
            </span>
          )}
          {health?.status === "down" && <span className="chip bad">backend offline</span>}
          {user && can("official") && (
            <button className="bell" onClick={() => navigate("/monitoring")} aria-label={`${unread} unread alerts`}>
              🔔{unread > 0 && <span className="bell-count">{unread}</span>}
            </button>
          )}
          {user ? (
            <div className="user-menu">
              <button onClick={() => setMenu((m) => !m)} aria-expanded={menu}>
                {user.full_name || user.username} <span className="role">{user.role}</span>
              </button>
              {menu && (
                <div className="menu" onMouseLeave={() => setMenu(false)}>
                  <div className="muted small">{user.organization || user.username}</div>
                  <button onClick={() => { setMenu(false); logout(); navigate("/"); }}>Sign out</button>
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
