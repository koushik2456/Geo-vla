import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmt } from "../layers.js";
import { Chart, DataTable } from "./Charts.jsx";

// Land-cover identity colours — the same ones the maps and legends use.
const CLASS_COLORS = {
  AnnualCrop: "#e6c85a", Forest: "#2e9e4f", HerbaceousVegetation: "#96c85a", Highway: "#9a9aa6", Industrial: "#c83cc8",
  Pasture: "#bee678", PermanentCrop: "#dc963c", Residential: "#e63c3c", River: "#2878e6", SeaLake: "#3c5cc8",
};
const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(1)}%`);

/** <img> that sends the auth header (artifact endpoints are admin-only). */
export function AuthImage({ path, alt, className = "artifact" }) {
  const [src, setSrc] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let url;
    let stop = false;
    setFailed(false);
    api(path)
      .then((r) => r.blob())
      .then((b) => {
        if (stop) return;
        url = URL.createObjectURL(b);
        setSrc(url);
      })
      .catch(() => !stop && setFailed(true));
    return () => {
      stop = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [path]);
  if (failed) return <p className="muted small">Not available for this run.</p>;
  return src ? <img src={src} alt={alt} className={className} /> : <div className="spinner" />;
}

export function Stat({ label, value, note }) {
  return (
    <div className="figure">
      <div className="figure-value">{value}</div>
      <div className="figure-label">{label}</div>
      {note && <div className="figure-note">{note}</div>}
    </div>
  );
}

// -- dataset -----------------------------------------------------------------------------------

export function DatasetStats({ stats, imagePath }) {
  if (!stats) return null;
  if (stats.task === "binary change detection") {
    const h = stats.change_ratio_histogram;
    return (
      <div className="tab-body">
        <div className="stat-row">
          <Stat label="Image pairs" value={stats.total.toLocaleString()} />
          <Stat label="Patch size" value={stats.image_shape.slice(0, 2).join("×")} note="RGB, before + after" />
          <Stat label="Changed pixels" value={pct(stats.class_balance.changed)} note="class imbalance → weighted loss" />
          <Stat label="Pairs without change" value={pct(stats.change_pixel_ratio.pairs_without_change)} />
        </div>
        <div className="studio-grid">
          <Chart chart={{ type: "bar", title: "Split sizes", unit: "pairs", categories: Object.keys(stats.split_sizes),
            series: [{ name: "Pairs", values: Object.values(stats.split_sizes) }] }} />
          <Chart chart={{ type: "bar", title: "How much of each pair changed", unit: "pairs",
            categories: h.counts.map((_, i) => `${(100 * h.bins[i]).toFixed(1)}%`), series: [{ name: "Pairs", values: h.counts }] }} />
          <Chart chart={{ type: "bar", title: "Mean colour before vs after", unit: "0–1", categories: ["Red", "Green", "Blue"],
            series: [{ name: "Before (T1)", values: stats.channel_mean_t1 }, { name: "After (T2)", values: stats.channel_mean_t2 }] }} />
        </div>
        {imagePath && <><h3 className="section-title">Sample pairs</h3><AuthImage path={imagePath} alt="Sample change pairs" /></>}
      </div>
    );
  }
  const classes = stats.classes.filter((c) => stats.per_class[c]);
  const splits = Object.keys(stats.class_counts);
  const h = stats.histograms;
  return (
    <div className="tab-body">
      <div className="stat-row">
        <Stat label="Images" value={stats.total.toLocaleString()} note={stats.source} />
        <Stat label="Image size" value={stats.image_shape.join("×")} note="height × width × RGB" />
        <Stat label="Classes" value={classes.length} />
        <Stat label="Train / val / test" value={Object.values(stats.split_sizes).join(" / ")} note="stratified split" />
      </div>
      <div className="studio-grid">
        <Chart chart={{ type: "bar", title: "Class balance per split", unit: "images", categories: classes,
          series: splits.map((s) => ({ name: s, values: classes.map((c) => stats.class_counts[s][c]) })) }} />
        <Chart chart={{ type: "line", title: "Pixel intensity histograms", unit: "pixels", colors: ["#ef4444", "#22c55e", "#3b82f6"],  x: h.bins.slice(0, -1).map((b) => b.toFixed(2)),
          x_label: "intensity (0–1)", series: [{ name: "Red", values: h.R }, { name: "Green", values: h.G }, { name: "Blue", values: h.B }] }} />
        <Chart chart={{ type: "bar", title: "Spectral signature: mean RGB per class", unit: "0–1", colors: ["#ef4444", "#22c55e", "#3b82f6"],  categories: classes,
          series: ["Red", "Green", "Blue"].map((n, k) => ({ name: n, values: classes.map((c) => stats.per_class[c].mean_rgb[k]) })) }} />
      </div>
      <div className="studio-grid">
        <div className="card">
          <h3 className="section-title">Per-class features</h3>
          <DataTable columns={["Class", "Mean R", "Mean G", "Mean B", "Brightness", "Greenness"]}
            rows={classes.map((c) => [c, ...stats.per_class[c].mean_rgb, stats.per_class[c].brightness, stats.per_class[c].greenness_index])} />
          <p className="hint">Greenness = (G − R) / (G + R): the visible-band proxy for NDVI. Vegetation classes score high, water and built-up low.</p>
        </div>
        <div className="card">
          <h3 className="section-title">Normalisation</h3>
          <dl className="kv">
            <dt>Dataset mean (RGB)</dt><dd>{stats.channel_mean.join(", ")}</dd>
            <dt>Dataset std (RGB)</dt><dd>{stats.channel_std.join(", ")}</dd>
            <dt>Network input mean</dt><dd>{stats.normalisation.mean.join(", ")}</dd>
            <dt>Network input std</dt><dd>{stats.normalisation.std.join(", ")}</dd>
          </dl>
          <p className="hint">{stats.normalisation.note} — inputs are standardised the way the ImageNet-pretrained backbone expects. Statistics from {stats.stats_sample_size.toLocaleString()} images.</p>
        </div>
      </div>
      {imagePath && <><h3 className="section-title">Samples per class</h3><AuthImage path={imagePath} alt="Dataset samples" /></>}
    </div>
  );
}

// -- architecture --------------------------------------------------------------------------------

const shape = (s) => (Array.isArray(s?.[0]) ? s.map((x) => x.join("×")).join(" · ") : s ? s.join("×") : "—");

export function ArchitectureView({ arch }) {
  if (!arch) return <p className="muted">No architecture recorded.</p>;
  const maxLog = Math.max(...arch.layers.map((l) => Math.log10(l.params + 1)), 1);
  return (
    <div className="tab-body">
      <div className="stat-row">
        <Stat label="Parameters" value={(arch.total_params / 1e6).toFixed(2) + " M"} note={`${arch.trainable_params.toLocaleString()} trainable`} />
        <Stat label="Weights size" value={`${arch.size_mb} MB`} note="float32" />
        <Stat label="Input" value={shape(arch.input_shape)} note="batch × channels × H × W" />
        <Stat label="Blocks shown" value={arch.layers.length} />
      </div>
      <div className="card arch">
        <h3 className="section-title">{arch.title} — forward pass, layer by layer</h3>
        <div className="arch-row small muted"><span>Block</span><span>Parameters (log scale)</span><span>Output shape</span><span>Type</span></div>
        {arch.layers.map((l) => (
          <div className="arch-row" key={l.name}>
            <strong>{l.name}</strong>
            <div className="arch-bar" style={{ width: `${Math.max(2, (100 * Math.log10(l.params + 1)) / maxLog)}%` }}>{l.params.toLocaleString()}</div>
            <span className="mono">{shape(l.output_shape)}</span>
            <span className="mono">{l.type}</span>
          </div>
        ))}
      </div>
      {arch.notes && (
        <div className="card">
          <dl className="kv">{Object.entries(arch.notes).map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd style={{ fontFamily: "var(--font)" }}>{v}</dd></React.Fragment>)}</dl>
        </div>
      )}
    </div>
  );
}

// -- evaluation ----------------------------------------------------------------------------------

function ConfusionMatrix({ report }) {
  const cm = report.confusion_matrix;
  const rowsum = cm.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  return (
    <div className="table-wrap">
      <table className="cm">
        <thead>
          <tr><th />{report.classes.map((c) => <th key={c} className="rot"><span>{c}</span></th>)}</tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <th style={{ textAlign: "right", paddingRight: 6, width: "auto" }}>{report.classes[i]}</th>
              {row.map((v, j) => {
                const t = v / rowsum[i];
                return (
                  <td key={j} title={`true ${report.classes[i]} → predicted ${report.classes[j]}: ${v}`}
                    style={{ background: i === j ? `rgba(52,211,153,${0.15 + 0.85 * t})` : v ? `rgba(248,113,113,${0.15 + 0.85 * t})` : "rgba(255,255,255,0.03)",
                      color: t > 0.5 ? "#04100a" : "var(--text)" }}>
                    {v || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">Rows = true class, columns = prediction. Diagonal (green) = correct; off-diagonal (red) shows which classes the network confuses.</p>
    </div>
  );
}

function EmbeddingScatter({ emb, classes }) {
  const [hover, setHover] = useState(null);
  // Robust axes (2nd–98th percentile) so a few outliers don't squash the clusters.
  const q = (arr, p) => [...arr].sort((a, b) => a - b)[Math.min(arr.length - 1, Math.max(0, Math.round(p * (arr.length - 1))))];
  const xs = emb.points.map((p) => p[0]);
  const ys = emb.points.map((p) => p[1]);
  const [x0, x1, y0, y1] = [q(xs, 0.02), q(xs, 0.98), q(ys, 0.02), q(ys, 0.98)];
  const W = 420;
  const H = 300;
  const clamp = (v) => Math.min(1.03, Math.max(-0.03, v));
  const sx = (x) => 16 + clamp((x - x0) / (x1 - x0 || 1)) * (W - 32);
  const sy = (y) => H - 16 - clamp((y - y0) / (y1 - y0 || 1)) * (H - 32);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="scatter" role="img" aria-label="Embedding projection">
        {emb.points.map((p, i) => (
          <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={hover === emb.labels[i] ? 4 : 2.6} fill={CLASS_COLORS[classes[emb.labels[i]]]}
            opacity={hover === null || hover === emb.labels[i] ? 0.9 : 0.12} />
        ))}
      </svg>
      <div className="class-legend">
        {classes.map((c, k) => (
          <span key={c} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
            <i style={{ background: CLASS_COLORS[c] }} />{c}
          </span>
        ))}
      </div>
      <p className="hint">Each dot is a test image: its {emb.dim}-dimensional penultimate-layer feature vector projected to 2-D with PCA
        ({pct(emb.explained_variance[0])} + {pct(emb.explained_variance[1])} of variance). Tight, separated clusters = the network learned discriminative features. Hover a class to highlight it.</p>
    </div>
  );
}

export function EvaluationView({ ev, model }) {
  if (!ev) return <p className="muted">Evaluation appears when training finishes.</p>;
  if (model === "change_detector") {
    const c = ev.confusion;
    return (
      <div className="tab-body">
        <div className="stat-row">
          <Stat label="F1" value={pct(ev.f1)} note="harmonic mean of precision and recall" />
          <Stat label="IoU" value={pct(ev.iou)} note="overlap of predicted and true change" />
          <Stat label="Precision" value={pct(ev.precision)} />
          <Stat label="Recall" value={pct(ev.recall)} />
          <Stat label="Overall accuracy" value={pct(ev.overall_accuracy)} note="inflated by unchanged pixels" />
          <Stat label="Best threshold" value={ev.best_threshold.threshold} note={`F1 ${pct(ev.best_threshold.f1)}`} />
        </div>
        <div className="studio-grid">
          <Chart chart={{ type: "line", title: "Threshold sweep", unit: "%", x: ev.threshold_sweep.map((r) => r.threshold.toFixed(1)), x_label: "decision threshold",
            series: [{ name: "Precision", values: ev.threshold_sweep.map((r) => +(100 * r.precision).toFixed(2)) },
              { name: "Recall", values: ev.threshold_sweep.map((r) => +(100 * r.recall).toFixed(2)) },
              { name: "F1", values: ev.threshold_sweep.map((r) => +(100 * r.f1).toFixed(2)) }] }} />
          <div className="card">
            <h3 className="section-title">Pixel confusion (threshold 0.5)</h3>
            <DataTable columns={["", "Predicted change", "Predicted no change"]}
              rows={[["True change", c.tp.toLocaleString(), c.fn.toLocaleString()], ["True no change", c.fp.toLocaleString(), c.tn.toLocaleString()]]} />
            <p className="hint">Raising the threshold trades recall for precision; the sweep shows where F1 peaks.</p>
          </div>
        </div>
      </div>
    );
  }
  const classes = ev.classes;
  const per = ev.per_class;
  return (
    <div className="tab-body">
      <div className="stat-row">
        <Stat label="Test accuracy" value={pct(ev.accuracy)} />
        <Stat label="Macro F1" value={pct(ev.macro_f1)} note="mean F1 over classes" />
        <Stat label="Test loss" value={fmt(ev.test_loss, 3)} />
        <Stat label="Test images" value={Object.values(per).reduce((a, v) => a + v.support, 0)} />
      </div>
      <div className="studio-grid">
        <div className="card"><h3 className="section-title">Confusion matrix</h3><ConfusionMatrix report={ev} /></div>
        <div className="card"><h3 className="section-title">Learned feature space</h3><EmbeddingScatter emb={ev.embedding} classes={classes} /></div>
      </div>
      <Chart chart={{ type: "bar", title: "Per-class precision / recall / F1", unit: "%", categories: classes.filter((c) => per[c].support),
        series: ["precision", "recall", "f1"].map((k) => ({ name: k[0].toUpperCase() + k.slice(1), values: classes.filter((c) => per[c].support).map((c) => +(100 * per[c][k]).toFixed(1)) })) }} />
      <DataTable columns={["Class", "Precision", "Recall", "F1", "Support"]}
        rows={classes.map((c) => [c, pct(per[c].precision), pct(per[c].recall), pct(per[c].f1), per[c].support])} />
    </div>
  );
}

// -- live training curves ------------------------------------------------------------------------

export function TrainingCurves({ job }) {
  const isCls = job.model === "classifier";
  const hist = job.history || [];
  const b = job.batches || [];
  const x = b.map((e) => String(e.global_step ?? `${e.epoch}.${e.step}`));
  const charts = [];
  if (b.length > 1) {
    charts.push({ type: "line", title: "Loss per batch", unit: "loss", x, x_label: "optimisation step", series: [{ name: "Loss", values: b.map((e) => +e.loss.toFixed(4)) }] });
    if (b[0].lr != null) charts.push({ type: "line", title: "Learning rate schedule", unit: "lr", x, x_label: "step", series: [{ name: "Learning rate", values: b.map((e) => e.lr) }] });
    if (b[0].grad_norm != null) charts.push({ type: "line", title: "Gradient norm", unit: "‖∇‖", x, x_label: "step", series: [{ name: "Gradient L2 norm", values: b.map((e) => +e.grad_norm.toFixed(3)) }] });
    if (b[0].batch_acc != null) charts.push({ type: "line", title: "Batch accuracy", unit: "%", x, x_label: "step", series: [{ name: "Accuracy", values: b.map((e) => +(100 * e.batch_acc).toFixed(1)) }] });
  }
  if (hist.length) {
    const ep = hist.map((h) => `ep ${h.epoch}`);
    charts.push({ type: "line", title: "Train vs validation loss", unit: "loss", x: ep,
      series: [{ name: "Train", values: hist.map((h) => +h.train_loss.toFixed(4)) }, ...(hist[0].val_loss != null ? [{ name: "Validation", values: hist.map((h) => +h.val_loss.toFixed(4)) }] : [])] });
    charts.push(isCls
      ? { type: "line", title: "Train vs validation accuracy", unit: "%", x: ep,
        series: [{ name: "Train", values: hist.map((h) => +(100 * h.train_acc).toFixed(2)) }, { name: "Validation", values: hist.map((h) => +(100 * h.val_acc).toFixed(2)) }] }
      : { type: "line", title: "Validation F1 / IoU", unit: "%", x: ep,
        series: [{ name: "F1", values: hist.map((h) => +(100 * h.val_f1).toFixed(2)) }, { name: "IoU", values: hist.map((h) => +(100 * h.val_iou).toFixed(2)) }] });
  }
  return (
    <div className="tab-body">
      {job.hyperparameters && (
        <div className="card">
          <h3 className="section-title">Hyperparameters</h3>
          <dl className="kv">{Object.entries(job.hyperparameters).map(([k, v]) => <React.Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></React.Fragment>)}</dl>
        </div>
      )}
      {charts.length === 0 ? <p className="muted">Curves appear after the first batches.</p> : <div className="studio-grid">{charts.map((c) => <Chart key={c.title} chart={c} />)}</div>}
      {hist.length > 0 && (
        <DataTable columns={Object.keys(hist[0]).filter((k) => k !== "event" && k !== "time")}
          rows={hist.map((h) => Object.entries(h).filter(([k]) => k !== "event" && k !== "time").map(([, v]) => (typeof v === "number" ? +v.toFixed(5) : v)))} />
      )}
    </div>
  );
}
