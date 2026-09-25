import React, { useCallback, useEffect, useState } from "react";
import { ArrowUpCircle, Download, Play, RefreshCw, Square, Trash2, Upload } from "lucide-react";
import { api } from "../api.js";
import { fmt } from "../layers.js";
import { ArchitectureView, AuthImage, DatasetStats, EvaluationView, Stat, TrainingCurves } from "../components/NetworkAnalytics.jsx";

const ICON = { size: 14, strokeWidth: 1.75 };
const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");
const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(1)}%`);
const statusClass = (s) => (s === "done" ? "status-done" : s === "running" ? "status-running" : "status-failed");

function elapsed(start, end) {
  if (!start) return "—";
  const ms = Math.max(0, (end ? new Date(end) : new Date()) - new Date(start));
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

// -- model registry ------------------------------------------------------------------------------

function Registry({ models, onChange, onOpenVersion }) {
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
  });

  return Object.entries(models).map(([key, m]) => (
    <section className="card" key={key}>
      <div className="page-head">
        <h2>{m.title}</h2>
        <span className={`status ${m.active_checkpoint ? "status-done" : "status-queued"}`}>
          {m.active_checkpoint ? `active ${m.active_checkpoint}` : "classical fallback"}
        </span>
      </div>
      {m.versions.length === 0 ? (
        <p className="muted small">No versions. Train in the Train tab or upload a checkpoint.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Version</th>
              <th>Dataset</th>
              <th className="num">Epochs</th>
              <th className="num">{m.score.label}</th>
              <th>Status</th>
              <th>Source</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {m.versions.map((v) => (
              <tr key={v.version} className={v.active ? "active-row" : ""}>
                <td className="mono strong">{v.version}</td>
                <td className="small">{v.metrics.dataset_name || v.dataset || "—"}{v.metrics.pretrained === false ? " · random init" : ""}</td>
                <td className="num">{v.metrics.epochs ?? "—"}</td>
                <td className="num">{pct(v.metrics[m.score.key])}</td>
                <td>{v.active ? <span className="chip ok">active</span> : <span className="chip">inactive</span>}</td>
                <td className="small">{v.source}{v.notes ? ` · ${v.notes}` : ""}</td>
                <td className="small mono">{when(v.created_at)}</td>
                <td>
                  <div className="registry-actions">
                    {v.job_id && <button className="link" onClick={() => onOpenVersion(v.job_id)}>Analytics</button>}
                    {!v.active && (
                      <button
                        className="icon"
                        title="Promote"
                        aria-label={`Promote ${v.version}`}
                        onClick={() => act(() => api(`/models/${key}/versions/${v.version}/promote`, { method: "POST" }))}
                      >
                        <ArrowUpCircle {...ICON} />
                      </button>
                    )}
                    {!v.active && (
                      <button
                        className="icon"
                        title="Delete"
                        aria-label={`Delete ${v.version}`}
                        onClick={() => window.confirm(`Delete ${v.version}?`) && act(() => api(`/models/${key}/versions/${v.version}`, { method: "DELETE" }))}
                      >
                        <Trash2 {...ICON} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="row wrap">
        {m.active_checkpoint && (
          <button onClick={() => act(() => api(`/models/${key}/deactivate`, { method: "POST" }))}>
            Use classical fallback
          </button>
        )}
        <details className="upload">
          <summary><Upload {...ICON} /> Upload checkpoint</summary>
          <div className="field"><label>Checkpoint (.pth)</label><input type="file" accept=".pth,.pt" onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], file: e.target.files[0] } })} /></div>
          <div className="field"><label>Metrics (.metrics.json, optional)</label><input type="file" accept=".json" onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], metrics: e.target.files[0] } })} /></div>
          <div className="field"><label>Notes</label><input value={upload[key]?.notes || ""} onChange={(e) => setUpload({ ...upload, [key]: { ...upload[key], notes: e.target.value } })} /></div>
          <button onClick={() => send(key)}><Upload {...ICON} /> Upload & validate</button>
        </details>
      </div>
      {msg && <p className="small">{msg}</p>}
    </section>
  ));
}

// -- dataset explorer ------------------------------------------------------------------------------

const EXPLORE = [
  ["synthetic", "Synthetic land-cover"],
  ["eurosat", "EuroSAT RGB"],
  ["synthetic_change", "Synthetic change"],
  ["levir", "LEVIR-CD"],
];

function DatasetExplorer({ datasets, onRefresh }) {
  const [id, setId] = useState("synthetic");
  const [stats, setStats] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const eurosat = datasets.find((d) => d.id === "eurosat");
  const levir = datasets.find((d) => d.id === "levir");

  const explore = useCallback(async (refresh = false) => {
    setBusy(true);
    setError(null);
    setStats(null);
    try {
      setStats(await api(`/training/datasets/${id}/explore${refresh ? "?refresh=true" : ""}`));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [id]);

  useEffect(() => {
    explore();
  }, [explore]);

  useEffect(() => {
    if (eurosat?.download?.status !== "running") return;
    const t = setInterval(onRefresh, 2000);
    return () => clearInterval(t);
  }, [eurosat?.download?.status, onRefresh]);

  return (
    <div className="tab-body">
      <div className="card row wrap between">
        <div className="studio-tabs">
          {EXPLORE.map(([k, label]) => (
            <button key={k} className={id === k ? "active" : ""} onClick={() => setId(k)}>{label}</button>
          ))}
        </div>
        <button className="ghost" onClick={() => explore(true)} disabled={busy} title="Recompute" aria-label="Recompute">
          <RefreshCw {...ICON} /> Recompute
        </button>
      </div>
      <div className="card row wrap between">
        <dl className="kv" style={{ flex: 1 }}>
          <dt>EuroSAT</dt>
          <dd>
            {eurosat?.available ? "downloaded" : eurosat?.download?.status === "running" ? `downloading — ${eurosat.download.message}` : "not downloaded"}
            {eurosat?.download?.status === "failed" && <span className="danger"> ({eurosat.download.message.split("\n")[0]})</span>}
          </dd>
          <dt>LEVIR-CD</dt>
          <dd>{levir?.available ? "imported" : <>not imported — <code>python -m training.download_data levir --from &lt;zip&gt;</code></>}</dd>
        </dl>
        {!eurosat?.available && eurosat?.download?.status !== "running" && (
          <button onClick={async () => { await api("/training/datasets/eurosat/download", { method: "POST" }); onRefresh(); }}>
            <Download {...ICON} /> Download EuroSAT (~90 MB)
          </button>
        )}
      </div>
      {busy && <p className="progress-line"><span className="spinner" /> Computing dataset statistics…</p>}
      {error && <div className="warn">{error}</div>}
      {stats && <DatasetStats stats={stats} imagePath={`/training/datasets/${id}/samples.png`} />}
    </div>
  );
}

// -- training form -------------------------------------------------------------------------------------

const PRESETS = {
  demo: { label: "Classroom demo (CPU, ~1 min)", model: "classifier", dataset: "synthetic", epochs: 6, batch_size: 32, lr: 0.001, samples: 100, img_size: 64, max_samples: 0 },
  eurosat_quick: { label: "EuroSAT quick (CPU, ~10 min)", model: "classifier", dataset: "eurosat", epochs: 5, batch_size: 64, lr: 0.0005, samples: 100, img_size: 64, max_samples: 3000 },
  eurosat_full: { label: "EuroSAT full (GPU)", model: "classifier", dataset: "eurosat", epochs: 10, batch_size: 64, lr: 0.0003, samples: 100, img_size: 224, max_samples: 0 },
  change_demo: { label: "Change detector demo (CPU)", model: "change_detector", dataset: "synthetic", epochs: 4, batch_size: 4, lr: 0.0003, samples: 64, img_size: 64, max_samples: 0 },
  levir_full: { label: "LEVIR-CD full (GPU)", model: "change_detector", dataset: "levir", epochs: 50, batch_size: 8, lr: 0.0001, samples: 64, img_size: 64, max_samples: 0 },
};

function TrainForm({ datasets, onStarted }) {
  const [f, setF] = useState({ ...PRESETS.demo, pretrained: true });
  const [error, setError] = useState(null);
  const options = datasets.filter((d) => d.model === f.model || d.model === "both");
  const chosen = datasets.find((d) => d.id === f.dataset);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.type === "number" ? Number(e.target.value) : e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const { model, dataset, label, ...params } = f;
      const job = await api("/training/jobs", { method: "POST", body: { model, dataset, params } });
      onStarted(job.id);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <form className="card train-form" onSubmit={submit}>
      <h2>Train</h2>
      <div className="studio-tabs">
        {Object.entries(PRESETS).map(([k, p]) => (
          <button type="button" key={k} onClick={() => setF({ ...p, pretrained: true })} className={f.label === p.label ? "active" : ""}>{p.label}</button>
        ))}
      </div>
      <div className="row wrap">
        <div className="field">
          <label htmlFor="t-model">Model</label>
          <select id="t-model" value={f.model} onChange={(e) => setF({ ...f, model: e.target.value, dataset: "synthetic", label: "" })}>
            <option value="classifier">Land-cover classifier (ResNet-50)</option>
            <option value="change_detector">Change detector (Siamese U-Net)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="t-data">Dataset</label>
          <select id="t-data" value={f.dataset} onChange={set("dataset")}>
            {options.map((d) => <option key={d.id} value={d.id}>{d.title}{d.available || d.auto_download ? "" : " — not imported"}</option>)}
          </select>
        </div>
        <div className="field"><label htmlFor="t-ep">Epochs</label><input id="t-ep" type="number" min={1} max={500} value={f.epochs} onChange={set("epochs")} /></div>
        <div className="field"><label htmlFor="t-bs">Batch size</label><input id="t-bs" type="number" min={1} max={512} value={f.batch_size} onChange={set("batch_size")} /></div>
        <div className="field"><label htmlFor="t-lr">Learning rate</label><input id="t-lr" type="number" step="any" value={f.lr} onChange={set("lr")} /></div>
        {f.model === "classifier" && <div className="field"><label htmlFor="t-img">Input size (px)</label><input id="t-img" type="number" min={32} max={512} value={f.img_size} onChange={set("img_size")} /></div>}
        {f.dataset === "synthetic" ? (
          <div className="field"><label htmlFor="t-n">{f.model === "classifier" ? "Samples per class" : "Training pairs"}</label><input id="t-n" type="number" min={4} max={20000} value={f.samples} onChange={set("samples")} /></div>
        ) : (
          <div className="field"><label htmlFor="t-max">Max samples (0 = all)</label><input id="t-max" type="number" min={0} value={f.max_samples} onChange={set("max_samples")} /></div>
        )}
        <div className="field field-check"><label htmlFor="t-pre">ImageNet initialisation</label><input id="t-pre" type="checkbox" checked={f.pretrained} onChange={set("pretrained")} /></div>
      </div>
      {chosen && <p className="hint">{chosen.how}</p>}
      {error && <div className="error">{error}</div>}
      <button className="primary"><Play {...ICON} /> Start training</button>
    </form>
  );
}

// -- job detail ------------------------------------------------------------------------------------------

const JOB_TABS = [["live", "Training curves"], ["network", "Network"], ["evaluation", "Evaluation"], ["visuals", "What it learned"], ["dataset", "Dataset"], ["log", "Log"]];

function JobDetail({ id, onFinished }) {
  const [job, setJob] = useState(null);
  const [tab, setTab] = useState("live");
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
  if (!job) return <div className="spinner" />;
  const isCls = job.model === "classifier";
  const a = job.analytics || {};
  const art = (name) => `/training/jobs/${job.id}/artifacts/${name}`;
  const cur = job.progress.current;
  return (
    <section className="card">
      <div className="job-head">
        <h2>Job #{job.id}</h2>
        <span className={`status ${statusClass(job.status)}`}>
          {job.status}{job.version ? ` → ${job.version}` : ""}
        </span>
      </div>
      <div className="progress" aria-label="Training progress"><div style={{ width: `${100 * job.progress.fraction}%` }} /></div>
      <div className="readout">
        <Stat label="Model" value={isCls ? "ResNet-50" : "Siamese U-Net"} />
        <Stat label="Dataset" value={job.dataset} />
        <Stat label="Device" value={job.progress.device || "—"} note={job.progress.pretrained == null ? "" : job.progress.pretrained ? "ImageNet init" : "random init"} />
        <Stat label="Params" value={job.params_count ? `${(job.params_count / 1e6).toFixed(1)} M` : "—"} />
        <Stat label="Epoch" value={`${job.progress.epochs_done}/${job.progress.epochs_total}`} note={cur ? `step ${cur.step}/${job.progress.steps_per_epoch}` : ""} />
        <Stat label="Elapsed" value={elapsed(job.started_at || job.created_at, job.finished_at)} />
        {job.history.length > 0 && (
          <Stat label={isCls ? "Best val accuracy" : "Best val F1"} value={pct(Math.max(...job.history.map((h) => (isCls ? h.val_acc : h.val_f1))))} />
        )}
        {job.result && (
          <Stat label={isCls ? "Val accuracy (best ckpt)" : "Val F1 (best ckpt)"}
            value={pct(isCls ? job.result.val_acc ?? job.result.test_acc : (job.result.val ?? job.result.test)?.f1)}
            note="test split sealed" />
        )}
      </div>
      {job.warnings.map((w) => <div key={w} className="warn small">{w}</div>)}
      {job.error && <div className="error">{job.error}</div>}
      {job.status === "running" && (
        <button onClick={() => api(`/training/jobs/${job.id}/cancel`, { method: "POST" })}>
          <Square {...ICON} /> Cancel
        </button>
      )}
      <div className="tabs" role="tablist">
        {JOB_TABS.map(([k, label]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>)}
      </div>
      {tab === "live" && <TrainingCurves job={job} />}
      {tab === "network" && <ArchitectureView arch={a.architecture} />}
      {tab === "evaluation" && <EvaluationView ev={a.evaluation} model={job.model} />}
      {tab === "visuals" && (
        a.images?.length ? (
          <div className="tab-body">
            {a.images.includes("filters.png") && <div className="card"><h3 className="section-title">First-layer filters</h3><div className="sample-frame"><AuthImage path={art("filters.png")} alt="Learned filters" /></div></div>}
            {a.images.includes("feature_maps.png") && <div className="card"><h3 className="section-title">Feature maps</h3><div className="sample-frame"><AuthImage path={art("feature_maps.png")} alt="Feature maps" /></div></div>}
            {a.images.includes("predictions.png") && <div className="card"><h3 className="section-title">Predictions</h3><div className="sample-frame"><AuthImage path={art("predictions.png")} alt="Predictions" /></div></div>}
            {a.images.includes("confusion_matrix.png") && <div className="card"><h3 className="section-title">Confusion matrix</h3><div className="sample-frame"><AuthImage path={art("confusion_matrix.png")} alt="Confusion matrix" /></div></div>}
            {a.images.includes("training_curves.png") && <div className="card"><h3 className="section-title">Training curves</h3><div className="sample-frame"><AuthImage path={art("training_curves.png")} alt="Training curves" /></div></div>}
          </div>
        ) : <p className="muted">Figures are generated when training finishes.</p>
      )}
      {tab === "dataset" && (a.dataset ? <DatasetStats stats={a.dataset} imagePath={a.images?.includes("dataset_samples.png") ? art("dataset_samples.png") : null} /> : <p className="muted">Dataset statistics appear once data is loaded.</p>)}
      {tab === "log" && (
        <div className="tab-body">
          {job.info?.map((m, i) => <div key={i} className="small muted">{m}</div>)}
          <pre className="json log">{job.log || "…"}</pre>
        </div>
      )}
    </section>
  );
}

// -- page ---------------------------------------------------------------------------------------------------

const TABS = [["models", "Models"], ["data", "Dataset explorer"], ["train", "Train"], ["jobs", "Jobs"]];

export default function StudioPage() {
  const [tab, setTab] = useState("models");
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

  const open = (id) => {
    setSelected(id);
    setTab("jobs");
  };

  return (
    <div className="page">
      <div className="page-head">
        <h1>Training studio</h1>
      </div>
      <div className="tabs" role="tablist">
        {TABS.map(([k, label]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>)}
      </div>
      {tab === "models" && models && <Registry models={models} onChange={load} onOpenVersion={open} />}
      {tab === "data" && <DatasetExplorer datasets={datasets} onRefresh={load} />}
      {tab === "train" && <TrainForm datasets={datasets} onStarted={(id) => { load(); open(id); }} />}
      {tab === "jobs" && (
        <div className="tab-body">
          {jobs.length === 0 ? <p className="muted">No training jobs yet.</p> : (
            <section className="card">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>Model</th>
                    <th>Dataset</th>
                    <th>Status</th>
                    <th className="num">Progress</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.id} className={j.id === selected ? "active-row" : ""} onClick={() => setSelected(j.id)} style={{ cursor: "pointer" }}>
                      <td className="num">{j.id}</td>
                      <td>{j.model.replace("_", " ")}</td>
                      <td>{j.dataset}</td>
                      <td><span className={`status ${statusClass(j.status)}`}>{j.status}</span></td>
                      <td className="num">{Math.round(100 * j.progress.fraction)}%{j.version ? ` → ${j.version}` : ""}</td>
                      <td className="small mono">{when(j.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
          {selected && <JobDetail key={selected} id={selected} onFinished={load} />}
        </div>
      )}
    </div>
  );
}
