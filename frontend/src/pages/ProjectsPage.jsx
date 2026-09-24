import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { navigate } from "../router.js";

const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");

export default function ProjectsPage() {
  const [projects, setProjects] = useState([]);
  const [runs, setRuns] = useState([]);
  const [filter, setFilter] = useState("all");
  const [name, setName] = useState("");
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const q = filter === "all" ? "" : filter === "saved" ? "?saved=true" : `?project_id=${filter}`;
      const [p, r] = await Promise.all([api("/projects"), api(`/runs${q}`)]);
      setProjects(p);
      setRuns(r);
    } catch (e) {
      setError(e.message);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  const create = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    await api("/projects", { method: "POST", body: { name: name.trim() } });
    setName("");
    load();
  };
  const remove = async (run) => {
    if (!window.confirm(`Delete "${run.title}"? This cannot be undone.`)) return;
    await api(`/runs/${run.id}`, { method: "DELETE" });
    load();
  };
  const removeProject = async (p) => {
    if (!window.confirm(`Delete project "${p.name}"? Its analyses are kept, unassigned.`)) return;
    await api(`/projects/${p.id}`, { method: "DELETE" });
    setFilter("all");
    load();
  };

  return (
    <div className="page">
      <h1>Projects & history</h1>
      {error && <div className="error">{error}</div>}
      <div className="projects-layout">
        <aside className="card">
          <h2>Projects</h2>
          <ul className="nav-list">
            <li><button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>All analyses</button></li>
            <li><button className={filter === "saved" ? "active" : ""} onClick={() => setFilter("saved")}>Saved</button></li>
            {projects.map((p) => (
              <li key={p.id} className="project-item">
                <button className={filter === String(p.id) ? "active" : ""} onClick={() => setFilter(String(p.id))}>
                  {p.name} <em>{p.run_count}</em>
                </button>
                <button className="link danger" aria-label={`Delete project ${p.name}`} onClick={() => removeProject(p)}>✕</button>
              </li>
            ))}
          </ul>
          <form onSubmit={create} className="row">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New project" aria-label="New project name" />
            <button>Add</button>
          </form>
        </aside>
        <section className="card grow">
          {runs.length === 0 ? (
            <p className="muted">No analyses here yet. <a href="#/">Run one</a>.</p>
          ) : (
            <table className="data-table runs-table">
              <thead>
                <tr><th>Analysis</th><th>Area</th><th>Status</th><th>Date</th><th /></tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <button className="link strong" onClick={() => navigate(`/run/${r.id}`)}>{r.title}</button>
                      <div className="muted small">
                        {r.workflow ? "Workflow" : "Question"}
                        {r.saved ? " · saved" : ""}
                        {r.shared ? " · shared" : ""}
                        {r.monitor_id ? " · monitor" : ""}
                      </div>
                    </td>
                    <td className="small">{r.place || r.bbox.map((v) => v.toFixed(2)).join(", ")}</td>
                    <td><span className={`status status-${r.status}`}>{r.status}</span></td>
                    <td className="small">{when(r.created_at)}</td>
                    <td><button className="link danger" onClick={() => remove(r)}>Delete</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </div>
  );
}
