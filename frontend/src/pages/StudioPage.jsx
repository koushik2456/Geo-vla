import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { fmt } from "../layers.js";
import { Chart } from "../components/Charts.jsx";

const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");
const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(1)}%`);

function Registry({ models, onChange }) {
  const [upload, setUpload] = useState({});
  const [msg, setMsg] = useState(null);
  const act = async (fn) => {
    setMsg(null);
    try {
      await fn();
      onChange();
    } catch (e) {
      setMsg(e.message);
    }
  };
  const send = (key) => act(async () => {
    const u = upload[key] || {};
    if (!u.file) throw new Error("Choose a .pth checkpoint");
    const form = new FormData();
    form.append("file", u.file);
    if (u.metrics) form.append("metrics", u.metrics);
    form.append("notes", u.notes || "");
    const r = await api(`/models/${key}/upload`, { method: "POST", form });
    setMsg(`Uploaded as ${r.version}`);
    setUpload({ ...upload, [key]: {} });
  });

  return Object.entries(models).map(([key, m]) => (
    <section className="card" key={key}>
      <div className="page-head">
        <h2>{m.title}</h2>
        <span className={`status ${m.active_checkpoint ? "status-done" : "status-queued"}`}>
          {m.active_checkpoint ? `in production: ${m.active_checkpoint}` : "classical fallback in use"}
        </span>
      </div>
      {m.versions.length === 0 ? (
        <p className="muted small">No versions yet — train one below or upload a checkpoint from the Colab notebook.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr><th>Version</th><th>Data</th><th>{m.score.label}</th><th>Source</th><th>Created</th><th /></tr>
          </thead>
          <tbody>
            {m.versions.map((v) => (
              <tr key={v.version} className={v.active ? "active-row" : ""}>
                <td className="strong">{v.version} {v.active && <span className="badge ok">active</span>}</td>
                <td className="small">{v.metrics.dataset_name || v.dataset || "—"}{v.metrics.pretrained === false ? " · no ImageNet init" : ""}</td>
                <td>{pct(v.metrics[m.score.key])}</td>
                <td className="small">{v.source}{v.notes ? ` · ${v.notes}` : ""}</td>
                <td className="small">{when(v.created_at)}</td>
                <td className="row">
                  {!v.active && <button onClick={() => act(() => api(`/models/${key}/versions/${v.version}/promote`, { method: "POST" }))}>Promote</button>}
                  {!v.active && <button className="link danger" onClick={() => window.confirm(`Delete ${v.version}?`) && act(() => api(`/models/${key}/versions/${v.version}`, { method: "DELETE" }))}>Delete</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="row wrap">
        {m.active_checkpoint && <button onClick={() => act(() => api(`/models/${key}/deactivate`, { method: "POST" }))}>Use classical fallback</button>}
        <details className="upload">
          <summary>Upload a checkpoint</summary>
          <div className="field"><label>Checkpoint (.pth)</label><input type="file" accept=".pth,.pt" onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], file: e.target.files[0] } })} /></div>
          <div className="field"><label>Metrics (.metrics.json, optional)</label><input type="file" accept=".json" onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], metrics: e.target.files[0] } })} /></div>
          <div className="field"><label>Notes</label><input value={upload[key]?.notes || ""} onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], notes: e.target.value } })} /></div>
          <button onClick={() => send(key)}>Upload & validate</button>
        </details>
      </div>
      {msg && <p className="small">{msg}</p>}
    </section>
  ));
}

function TrainForm({ datasets, onStarted }) {
  const [f, setF] = useState({ model: "classifier", dataset: "synthetic", epochs: 3, batch_size: 16, lr: 0.0003, samples: 100, pretrained: true });
  const [error, setError] = useState(null);
  const options = datasets.filter((d) => d.model === f.model || d.model === "both");
  const chosen = datasets.find((d) => d.id === f.dataset);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.type === "number" ? Number(e.target.value) : e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const { model, dataset, ...params } = f;
      const job = await api("/training/jobs", { method: "POST", body: { model, dataset, params } });
      onStarted(job.id);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <form className="card train-form" onSubmit={submit}>
      <h2>Train a model</h2>
      <div className="row wrap">
        <div className="field">
          <label htmlFor="t-model">Model</label>
          <select id="t-model" value={f.model} onChange={(e) => setF({ ...f, model: e.target.value, dataset: "synthetic", lr: e.target.value === "classifier" ? 0.0003 : 0.0001 })}>
            <option value="classifier">Land-cover classifier</option>
            <option value="change_detector">Change detector</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="t-data">Dataset</label>
          <select id="t-data" value={f.dataset} onChange={set("dataset")}>
            {options.map((d) => (
              <option key={d.id} value={d.id}>{d.title}{d.available || d.auto_download ? "" : " — not downloaded"}</option>
            ))}
          </select>
        </div>
        <div className="field"><label htmlFor="t-ep">Epochs</label><input id="t-ep" type="number" min={1} max={500} value={f.epochs} onChange={set("epochs")} /></div>
        <div className="field"><label htmlFor="t-bs">Batch size</label><input id="t-bs" type="number" min={1} max={512} value={f.batch_size} onChange={set("batch_size")} /></div>
        <div className="field"><label htmlFor="t-lr">Learning rate</label><input id="t-lr" type="number" step="any" value={f.lr} onChange={set("lr")} /></div>
        {f.dataset === "synthetic" && (
          <div className="field"><label htmlFor="t-n">{f.model === "classifier" ? "Samples per class" : "Training pairs"}</label><input id="t-n" type="number" min={4} max={20000} value={f.samples} onChange={set("samples")} /></div>
        )}
        <div className="field field-check"><label htmlFor="t-pre">ImageNet initialisation</label><input id="t-pre" type="checkbox" checked={f.pretrained} onChange={set("pretrained")} /></div>
      </div>
      {chosen && <p className="muted small">{chosen.how}</p>}
      {error && <div className="error">{error}</div>}
      <button className="primary">Start training</button>
    </form>
  );
}

function JobDetail({ id, onFinished }) {
  const [job, setJob] = useState(null);
  useEffect(() => {
    let stop = false;
    let timer;
    const tick = async () => {
      const j = await api(`/training/jobs/${id}`);
      if (stop) return;
      setJob(j);
      if (j.status === "running") timer = setTimeout(tick, 1500);
      else onFinished();
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);
  if (!job) return null;
  const isCls = job.model === "classifier";
  const hist = job.history;
  const valKey = isCls ? "val_acc" : "val_f1";
  return (
    <section className="card">
      <div className="page-head">
        <h2>Job #{job.id} · {job.model.replace("_", " ")} on {job.dataset}</h2>
        <span className={`status status-${job.status === "done" ? "done" : job.status === "running" ? "running" : "failed"}`}>{job.status}</span>
      </div>
      <div className="progress" aria-label="Training progress"><div style={{ width: `${100 * job.progress.fraction}%` }} /></div>
      <p className="small muted">
        Epoch {job.progress.epochs_done}/{job.progress.epochs_total}
        {job.progress.current && ` · step ${job.progress.current.step}/${job.progress.steps_per_epoch} · loss ${fmt(job.progress.current.loss, 4)}`}
        {job.progress.device && ` · ${job.progress.device}`}
        {job.version && ` · registered as ${job.version}`}
      </p>
      {job.warnings.map((w) => <div key={w} className="warn small">{w}</div>)}
      {job.error && <div className="error">{job.error}</div>}
      {job.status === "running" && <button onClick={() => api(`/training/jobs/${job.id}/cancel`, { method: "POST" })}>Cancel</button>}
      {hist.length > 0 && (
        <Chart chart={{ type: "line", title: isCls ? "Validation accuracy" : "Validation F1", unit: "%", x: hist.map((h) => `ep ${h.epoch}`),
          series: [{ name: isCls ? "Val accuracy" : "Val F1", values: hist.map((h) => +(100 * h[valKey]).toFixed(2)) }] }} />
      )}
      {job.batches.length > 1 && (
        <Chart chart={{ type: "line", title: "Training loss", unit: "loss", x: job.batches.map((b) => `${b.epoch}.${b.step}`),
          series: [{ name: "Loss", values: job.batches.map((b) => +b.loss.toFixed(4)) }] }} />
      )}
      {job.result && (
        <p className="strong">
          Test {isCls ? `accuracy ${pct(job.result.test_acc)}` : `F1 ${pct(job.result.test.f1)} · IoU ${pct(job.result.test.iou)}`}
        </p>
      )}
      <details>
        <summary>Log</summary>
        <pre className="json log">{job.log || "…"}</pre>
      </details>
    </section>
  );
}

export default function StudioPage() {
  const [models, setModels] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(null);

  const load = useCallback(async () => {
    const [m, d, j] = await Promise.all([api("/models"), api("/training/datasets"), api("/training/jobs")]);
    setModels(m);
    setDatasets(d);
    setJobs(j);
    setSelected((s) => s ?? j.find((x) => x.status === "running")?.id ?? j[0]?.id ?? null);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="page">
      <h1>Training studio</h1>
      <p className="muted">Train the vision models, compare versions, and promote the best to production. Promotion is instant — analyses pick up the new model without a restart.</p>
      {models && <Registry models={models} onChange={load} />}
      <TrainForm datasets={datasets} onStarted={(id) => { setSelected(id); load(); }} />
      {jobs.length > 0 && (
        <section className="card">
          <h2>Jobs</h2>
          <table className="data-table">
            <thead><tr><th>#</th><th>Model</th><th>Data</th><th>Status</th><th>Progress</th><th>Started</th></tr></thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className={j.id === selected ? "active-row" : ""} onClick={() => setSelected(j.id)} style={{ cursor: "pointer" }}>
                  <td>{j.id}</td><td>{j.model.replace("_", " ")}</td><td>{j.dataset}</td>
                  <td><span className={`status status-${j.status === "done" ? "done" : j.status === "running" ? "running" : "failed"}`}>{j.status}</span></td>
                  <td>{Math.round(100 * j.progress.fraction)}%{j.version ? ` → ${j.version}` : ""}</td>
                  <td className="small">{when(j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {selected && <JobDetail key={selected} id={selected} onFinished={load} />}
    </div>
  );
}
