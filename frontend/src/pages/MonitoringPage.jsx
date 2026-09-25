import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bell, Pause, Play, Plus, Trash2, X } from "lucide-react";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { fmt } from "../layers.js";
import PlaceSearch from "../components/PlaceSearch.jsx";
import { ParamFields, defaultParams } from "../components/WorkflowPanel.jsx";

const ICON = { size: 14, strokeWidth: 1.75 };
const OPS = { gt: ">", ge: "≥", lt: "<", le: "≤" };
const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "never");

function Sparkline({ history, threshold }) {
  const vals = history.map((h) => h.value).filter((v) => v != null);
  if (vals.length < 2) return <span className="muted small">{vals.length ? "1 check" : "no data"}</span>;
  const all = [...vals, threshold];
  const lo = Math.min(...all);
  const hi = Math.max(...all) || 1;
  const w = 120;
  const h = 32;
  const x = (i) => (i / (vals.length - 1)) * (w - 8) + 4;
  const y = (v) => h - 4 - ((v - lo) / (hi - lo || 1)) * (h - 8);
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label={`Trend of ${vals.length} checks`}>
      <line x1={0} x2={w} y1={y(threshold)} y2={y(threshold)} className="threshold" />
      <polyline points={vals.map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke="var(--series-1)" strokeWidth="1.5" />
      <circle cx={x(vals.length - 1)} cy={y(vals.at(-1))} r="2.5" fill="var(--series-1)" />
    </svg>
  );
}

function CreateMonitor({ catalog, aoi, place, onCreated }) {
  const workflows = useMemo(() => catalog?.sectors.flatMap((s) => s.workflows.map((w) => ({ ...w, sectorTitle: s.title }))) ?? [], [catalog]);
  const [wfId, setWfId] = useState("deforestation");
  const wf = workflows.find((w) => w.id === wfId);
  const [params, setParams] = useState({});
  const [area, setArea] = useState({ bbox: aoi, place });
  const [f, setF] = useState({ name: "", frequency: "weekly", metric: "", op: "gt", value: 0, notify_email: "" });
  const [error, setError] = useState(null);

  useEffect(() => {
    if (wf) {
      setParams(defaultParams(wf, { relative: true }));
      setF((x) => ({ ...x, metric: wf.metrics[0], name: x.name || wf.title }));
    }
  }, [wf]);
  useEffect(() => setArea({ bbox: aoi, place }), [aoi, place]);

  if (!wf) return null;
  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    const p = { ...params };
    for (const q of wf.params) if (q.type === "numbers") p[q.name] = String(p[q.name]).split(",").map(Number);
    try {
      await api("/monitors", {
        method: "POST",
        body: { name: f.name, workflow: wf.id, params: p, bbox: area.bbox, place: area.place, frequency: f.frequency,
          rule: { metric: f.metric, op: f.op, value: Number(f.value) }, notify_email: f.notify_email || null, run_now: true },
      });
      onCreated();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <form className="card monitor-form" onSubmit={submit}>
      <h2>New monitor</h2>
      <div className="field">
        <label htmlFor="m-name">Name</label>
        <input id="m-name" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} required />
      </div>
      <div className="field">
        <label>Area</label>
        <PlaceSearch onSelect={(r) => setArea({ bbox: r.analysis_bbox, place: r.label })} />
        <small>{area.place || `Map area [${area.bbox.map((v) => v.toFixed(2)).join(", ")}]`}</small>
      </div>
      <div className="field">
        <label htmlFor="m-wf">Workflow</label>
        <select id="m-wf" value={wfId} onChange={(e) => setWfId(e.target.value)}>
          {workflows.map((w) => (
            <option key={w.id} value={w.id}>
              {w.sectorTitle} — {w.title}
            </option>
          ))}
        </select>
      </div>
      <ParamFields wf={wf} values={params} onChange={setParams} relativeDates />
      <p className="hint">Relative dates (e.g. -1y) advance with each check.</p>
      <div className="field">
        <label>Alert when</label>
        <div className="row">
          <select value={f.metric} onChange={(e) => setF({ ...f, metric: e.target.value })} aria-label="Metric">
            {wf.metrics.map((m) => <option key={m}>{m}</option>)}
          </select>
          <select value={f.op} onChange={(e) => setF({ ...f, op: e.target.value })} aria-label="Comparison">
            {Object.entries(OPS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <input type="number" step="any" value={f.value} onChange={(e) => setF({ ...f, value: e.target.value })} aria-label="Threshold" />
        </div>
      </div>
      <div className="row">
        <div className="field">
          <label htmlFor="m-freq">Check</label>
          <select id="m-freq" value={f.frequency} onChange={(e) => setF({ ...f, frequency: e.target.value })}>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
        <div className="field grow">
          <label htmlFor="m-mail">Email (optional)</label>
          <input id="m-mail" type="email" value={f.notify_email} onChange={(e) => setF({ ...f, notify_email: e.target.value })} />
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      <button className="primary"><Play {...ICON} /> Start monitoring</button>
    </form>
  );
}

export default function MonitoringPage({ catalog, aoi, place, onAlertsChanged }) {
  const [monitors, setMonitors] = useState([]);
  const [alerts, setAlerts] = useState({ unread: 0, alerts: [] });
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    const [m, a] = await Promise.all([api("/monitors"), api("/alerts")]);
    setMonitors(m);
    setAlerts(a);
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);

  const act = async (fn) => {
    await fn();
    load();
  };
  const markRead = () => act(async () => {
    await api("/alerts/read", { method: "POST" });
    onAlertsChanged?.();
  });

  return (
    <div className="page">
      <div className="page-head">
        <h1>Monitoring</h1>
        <button className="primary" onClick={() => setCreating((c) => !c)}>
          {creating ? <><X {...ICON} /> Close</> : <><Plus {...ICON} /> New monitor</>}
        </button>
      </div>
      <div className="monitor-layout">
        <div className="grow">
          {creating && <CreateMonitor catalog={catalog} aoi={aoi} place={place} onCreated={() => { setCreating(false); load(); }} />}
          {monitors.length === 0 && !creating && <div className="card muted">No monitors.</div>}
          {monitors.map((m) => {
            const hit = m.last_value != null && { gt: m.last_value > m.rule.value, ge: m.last_value >= m.rule.value, lt: m.last_value < m.rule.value, le: m.last_value <= m.rule.value }[m.rule.op];
            return (
              <div key={m.id} className={`card monitor ${m.active ? "" : "paused"}`}>
                <div className="monitor-head">
                  <div>
                    <strong>{m.name}</strong>
                    <div className="muted small">{m.place || "Custom area"} · {m.workflow.replace(/_/g, " ")} · {m.frequency}</div>
                  </div>
                  <span className={`status ${hit ? "status-failed" : m.last_value != null ? "status-done" : "status-queued"}`}>
                    {m.last_value == null ? (m.last_status === "failed" ? "check failed" : "pending") : hit ? "alert" : "ok"}
                  </span>
                </div>
                <div className="monitor-body">
                  <div>
                    <div className="small muted">Rule</div>
                    <div className="mono">{m.rule.metric} {OPS[m.rule.op]} {m.rule.value}</div>
                  </div>
                  <div>
                    <div className="small muted">Latest</div>
                    <div className="strong mono">{m.last_value != null ? fmt(m.last_value, 3) : "—"}</div>
                  </div>
                  <Sparkline history={m.history} threshold={m.rule.value} />
                  <div className="small muted">Last: {when(m.last_checked_at)}<br />Next: {m.active ? when(m.next_run_at) : "paused"}</div>
                </div>
                <div className="row">
                  <button onClick={() => act(() => api(`/monitors/${m.id}/check`, { method: "POST" }))}>Check now</button>
                  <button onClick={() => act(() => api(`/monitors/${m.id}`, { method: "PATCH", body: { active: !m.active } }))}>
                    {m.active ? <><Pause {...ICON} /> Pause</> : <><Play {...ICON} /> Resume</>}
                  </button>
                  {m.last_run_id && <button onClick={() => navigate(`/run/${m.last_run_id}`)}>Open latest</button>}
                  <button
                    className="icon"
                    title="Delete"
                    aria-label={`Delete ${m.name}`}
                    onClick={() => window.confirm(`Stop watching "${m.name}"?`) && act(() => api(`/monitors/${m.id}`, { method: "DELETE" }))}
                  >
                    <Trash2 {...ICON} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <aside className="card alerts-card">
          <div className="page-head">
            <h2><Bell {...ICON} /> Alerts {alerts.unread > 0 && <span className="badge">{alerts.unread}</span>}</h2>
            {alerts.unread > 0 && <button className="link" onClick={markRead}>Mark all read</button>}
          </div>
          {alerts.alerts.length === 0 && <p className="muted small">No alerts.</p>}
          <ul className="alert-list">
            {alerts.alerts.map((a) => (
              <li key={a.id} className={`${a.level} ${a.read ? "" : "unread"}`}>
                <span className="alert-icon" aria-hidden="true"><AlertTriangle {...ICON} /></span>
                <div>
                  <div>{a.message}</div>
                  <div className="muted small">
                    {when(a.created_at)} {a.run_id && <button className="link" onClick={() => navigate(`/run/${a.run_id}`)}>view</button>}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
