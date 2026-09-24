import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { datedScenes, defaultLayerState, timelapseGroups } from "../layers.js";
import AskPanel from "../components/AskPanel.jsx";
import CompareView from "../components/CompareView.jsx";
import LayerPanel from "../components/LayerPanel.jsx";
import MapView from "../components/MapView.jsx";
import PlaceSearch from "../components/PlaceSearch.jsx";
import ResultPanel from "../components/ResultPanel.jsx";
import TimelapsePlayer from "../components/TimelapsePlayer.jsx";
import WorkflowPanel from "../components/WorkflowPanel.jsx";

// Cesium is large — only load it when the globe view is opened.
const GlobeView = lazy(() => import("../components/GlobeView.jsx"));

const VIEWS = [
  ["2d", "2D map"],
  ["terrain", "3D terrain"],
  ["globe", "Globe"],
  ["compare", "Compare"],
];

export default function AnalyzePage({ runId, shareToken, aoi, setAoi, place, setPlace, health, catalog }) {
  const [mode, setMode] = useState("workflows");
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);
  const [layerState, setLayerState] = useState({});
  const [view, setView] = useState("2d");
  const [compare, setCompare] = useState([null, null]);
  const [fitKey, setFitKey] = useState(0);
  const initialisedFor = useRef(null);
  const readOnly = !!shareToken;

  const load = useCallback(async () => {
    if (!runId && !shareToken) return null;
    const data = shareToken ? await api(`/share/${shareToken}`) : await api(`/runs/${runId}`);
    setRun(data);
    return data;
  }, [runId, shareToken]);

  // Load the run and poll while it is working (the trace grows live).
  useEffect(() => {
    setError(null);
    if (!runId && !shareToken) {
      setRun(null);
      return;
    }
    let stop = false;
    let timer;
    const tick = async () => {
      try {
        const data = await load();
        if (!stop && data && (data.status === "queued" || data.status === "running")) timer = setTimeout(tick, 900);
      } catch (e) {
        if (!stop) setError(e.status === 404 ? "This analysis does not exist, was deleted, or is private." : e.message);
      }
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [runId, shareToken, load]);

  // First time a run is seen: move the map to it; when it finishes: pick default layers.
  useEffect(() => {
    if (!run) return;
    if (initialisedFor.current !== run.id) {
      initialisedFor.current = run.id;
      setAoi(run.bbox);
      setPlace(run.place);
      setFitKey((k) => k + 1);
      setLayerState({});
    }
    if (run.status === "done" && run.layers.length && !Object.keys(layerState).length) {
      setLayerState(defaultLayerState(run.layers, run.insights?.focus_layer));
      const scenes = datedScenes(run.layers);
      const focus = run.insights?.focus_layer;
      const right = scenes.length > 1 ? scenes.at(-1).id : focus && run.layers.find((l) => l.id === focus)?.tiles ? focus : scenes.at(-1)?.id;
      setCompare([scenes[0]?.id ?? null, right ?? null]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  const start = async (request) => {
    setStarting(true);
    setError(null);
    try {
      const { id } = await api("/runs", { method: "POST", body: { bbox: aoi, place, ...request } });
      navigate(`/run/${id}`);
    } catch (e) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  };

  const layers = useMemo(() => (run?.status === "done" ? run.layers : []), [run]);
  const groups = useMemo(() => timelapseGroups(layers), [layers]);
  const showFrame = useCallback(
    (frames, id) => setLayerState((s) => ({ ...s, ...Object.fromEntries(frames.map((f) => [f.id, { ...(s[f.id] ?? { opacity: 0.85 }), visible: f.id === id }])) })),
    [],
  );
  const running = run && (run.status === "queued" || run.status === "running");
  const mapProps = { aoi, onAoiChange: (b) => { setAoi(b); setPlace(null); }, layers, layerState, share: shareToken };

  return (
    <div className="workspace">
      <aside className="sidebar">
        {readOnly && <div className="info">Shared analysis (read-only). <a href="#/">Start your own</a></div>}
        {!readOnly && run && (
          <button className="link back" onClick={() => navigate("/")}>
            ← New analysis
          </button>
        )}
        {!readOnly && !run && (
          <>
            <section className="aoi-card">
              <h2>1 · Area</h2>
              <p className="small">
                {place ? <strong>{place}</strong> : "Custom area"} — search above the map, draw a box, or use the current view.
              </p>
              <code className="small muted">[{aoi.map((v) => v.toFixed(3)).join(", ")}]</code>
            </section>
            <section>
              <h2>2 · Analysis</h2>
              <div className="tabs" role="tablist">
                <button role="tab" aria-selected={mode === "workflows"} className={mode === "workflows" ? "active" : ""} onClick={() => setMode("workflows")}>
                  Ready-made workflows
                </button>
                <button role="tab" aria-selected={mode === "ask"} className={mode === "ask" ? "active" : ""} onClick={() => setMode("ask")}>
                  Ask a question
                </button>
              </div>
              {mode === "workflows" ? <WorkflowPanel catalog={catalog} busy={starting} onRun={start} /> : <AskPanel busy={starting} onRun={start} planner={health?.planner} />}
            </section>
          </>
        )}
        {error && <div className="error">⚠ {error}</div>}
        {run && <ResultPanel run={run} share={shareToken} onChanged={load} />}
      </aside>

      <main className="map-area">
        <div className="view-toggle" role="tablist" aria-label="Map view">
          {VIEWS.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={view === id} className={view === id ? "active" : ""} onClick={() => setView(id)}
              disabled={id === "compare" && layers.filter((l) => l.tiles).length < 2}>
              {label}
            </button>
          ))}
        </div>
        {!readOnly && !run && (
          <div className="map-search">
            <PlaceSearch onSelect={(r) => { setAoi(r.analysis_bbox); setPlace(r.label); setFitKey((k) => k + 1); }} />
          </div>
        )}

        {view === "globe" ? (
          <Suspense fallback={<div className="map-loading">Loading 3D globe…</div>}>
            <GlobeView {...mapProps} />
          </Suspense>
        ) : view === "compare" && layers.length ? (
          <CompareView aoi={aoi} layers={layers} share={shareToken} left={compare[0]} right={compare[1]} onChange={(l, r) => setCompare([l, r])} />
        ) : (
          <MapView {...mapProps} terrain={view === "terrain"} fitKey={fitKey} readOnly={readOnly || !!run} />
        )}

        {layers.length > 0 && view !== "compare" && <LayerPanel layers={layers} layerState={layerState} onChange={setLayerState} />}
        {Object.keys(groups).length > 0 && view !== "compare" && <TimelapsePlayer groups={groups} onShow={showFrame} />}
        {running && <div className="map-busy"><span className="spinner" aria-hidden="true" /> Running analysis…</div>}
      </main>
    </div>
  );
}
