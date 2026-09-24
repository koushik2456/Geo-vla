import React, { useEffect, useState } from "react";
import { api, download } from "../api.js";
import { useAuth } from "../auth.jsx";
import { Chart, KeyFigures } from "./Charts.jsx";
import TracePanel from "./TracePanel.jsx";

const STATUS = { queued: "Queued", running: "Running", done: "Complete", failed: "Failed" };

function SaveShare({ run, onChanged }) {
  const { user } = useAuth();
  const [projects, setProjects] = useState([]);
  const [projectId, setProjectId] = useState(run.project_id ?? "");
  const [newName, setNewName] = useState("");
  const [msg, setMsg] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (user && run.can_edit) api("/projects").then(setProjects).catch(() => {});
  }, [user, run.can_edit]);

  if (!user) return <p className="muted small">Sign in to save this analysis to a project or share it.</p>;
  if (!run.can_edit) return null;
  const shareUrl = run.share_token ? `${window.location.origin}${window.location.pathname}#/share/${run.share_token}` : null;

  const save = async () => {
    let pid = projectId === "" ? null : Number(projectId);
    if (projectId === "new") {
      if (!newName.trim()) return setMsg("Name the new project");
      pid = (await api("/projects", { method: "POST", body: { name: newName.trim() } })).id;
      setProjects(await api("/projects"));
      setProjectId(String(pid));
      setNewName("");
    }
    await api(`/runs/${run.id}`, { method: "PATCH", body: pid ? { saved: true, project_id: pid } : { saved: true } });
    setMsg("Saved");
    onChanged();
  };
  const share = async (on) => {
    await api(`/runs/${run.id}/share`, { method: on ? "POST" : "DELETE" });
    onChanged();
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked — the link is selectable */
    }
  };

  return (
    <div className="save-share">
      <div className="row">
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)} aria-label="Project">
          <option value="">No project</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
          <option value="new">+ New project…</option>
        </select>
        {projectId === "new" && <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Project name" aria-label="New project name" />}
        <button onClick={save}>{run.saved ? "Update" : "Save"}</button>
      </div>
      {msg && <span className="muted small">{msg}</span>}
      <div className="row">
        {shareUrl ? (
          <>
            <input readOnly value={shareUrl} aria-label="Share link" onFocus={(e) => e.target.select()} />
            <button onClick={copy}>{copied ? "Copied" : "Copy"}</button>
            <button onClick={() => share(false)}>Revoke</button>
          </>
        ) : (
          <button onClick={() => share(true)}>Create public link</button>
        )}
      </div>
    </div>
  );
}

function Exports({ run, share }) {
  const [busy, setBusy] = useState(null);
  const get = async (key, path, name) => {
    setBusy(key);
    try {
      await download(path, name, share);
    } finally {
      setBusy(null);
    }
  };
  const short = run.id.slice(0, 8);
  return (
    <div className="exports">
      <div className="row">
        <button className="primary" disabled={busy} onClick={() => get("pdf", `/runs/${run.id}/report.pdf`, `geo-vla-report-${short}.pdf`)}>
          {busy === "pdf" ? "Preparing…" : "PDF report"}
        </button>
        <button disabled={busy} onClick={() => get("zip", `/runs/${run.id}/export.zip`, `geo-vla-${short}.zip`)}>
          {busy === "zip" ? "Preparing…" : "Everything (ZIP)"}
        </button>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th>Layer</th>
            <th>Download</th>
          </tr>
        </thead>
        <tbody>
          {run.layers.map((l) => (
            <tr key={l.id}>
              <td>{l.name}</td>
              <td className="dl">
                {l.downloads.geotiff && (
                  <button className="link" onClick={() => get(l.id + "tif", l.downloads.geotiff, `${l.id}.tif`)}>
                    GeoTIFF
                  </button>
                )}
                {l.downloads.geojson && (
                  <button className="link" onClick={() => get(l.id + "gj", l.downloads.geojson, `${l.id}.geojson`)}>
                    GeoJSON
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">GeoTIFF and GeoJSON open in QGIS, ArcGIS and Google Earth Pro (EPSG:4326).</p>
    </div>
  );
}

export default function ResultPanel({ run, share, onChanged }) {
  const [tab, setTab] = useState("summary");
  const running = run.status === "queued" || run.status === "running";
  const ins = run.insights || {};
  const steps = run.trace.filter((t) => t.type === "tool_call").length;
  const tabs = [
    ["summary", "Summary"],
    ["dashboard", `Dashboard${ins.charts ? ` (${ins.charts.length})` : ""}`],
    ["trace", `Trace (${steps})`],
    ["export", "Export"],
  ];

  return (
    <section className="result">
      <header>
        <h2>{run.title}</h2>
        <span className={`status status-${run.status}`}>{STATUS[run.status]}</span>
      </header>
      {run.place && <p className="muted small">{run.place}</p>}
      <div className="tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}
            disabled={running && (id === "dashboard" || id === "export")}>
            {label}
          </button>
        ))}
      </div>

      {tab === "summary" && (
        <div className="tab-body">
          {running && (
            <p className="progress-line">
              <span className="spinner" aria-hidden="true" /> {steps ? `Step ${steps}: ${run.trace.findLast((t) => t.type === "tool_call")?.tool}` : "Planning…"}
            </p>
          )}
          {run.status === "failed" && <div className="error">The analysis failed: {run.error}</div>}
          {run.status === "done" && (
            <>
              {run.data_mode === "synthetic" && <div className="warn">Demonstration data — synthetic imagery. Not for decisions.</div>}
              <KeyFigures figures={ins.key_figures} />
              <div className="answer-body">
                {run.answer.split("\n").filter((l) => l.trim() && !l.startsWith("_")).map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
              <p className="meta">
                {run.planner === "workflow" ? "Fixed workflow" : run.planner === "claude" ? run.model : "Offline planner"} · {steps} steps · data {run.data_mode}
              </p>
              <SaveShare run={run} onChanged={onChanged} />
            </>
          )}
        </div>
      )}
      {tab === "dashboard" && (
        <div className="tab-body">
          <KeyFigures figures={ins.key_figures} />
          {(ins.charts || []).map((c, i) => (
            <Chart key={i} chart={c} />
          ))}
          {ins.methods?.length > 0 && (
            <p className="muted small">Methods: {ins.methods.join("; ")}</p>
          )}
        </div>
      )}
      {tab === "trace" && (
        <div className="tab-body">
          <TracePanel trace={run.trace} running={running} />
        </div>
      )}
      {tab === "export" && run.status === "done" && (
        <div className="tab-body">
          <Exports run={run} share={share} />
        </div>
      )}
    </section>
  );
}
