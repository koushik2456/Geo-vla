import React, { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { getHealth, runQuery } from "./api.js";
import ChatPanel from "./components/ChatPanel.jsx";
import TracePanel from "./components/TracePanel.jsx";
import LayerPanel from "./components/LayerPanel.jsx";
import MapView from "./components/MapView.jsx";
import { defaultLayerState } from "./layers.js";

// Cesium is large — only load it when the globe view is opened.
const GlobeView = lazy(() => import("./components/GlobeView.jsx"));

const DEFAULT_AOI = [77.5, 12.9, 77.56, 12.96]; // Bengaluru outskirts, ~6.5 km square

export default function App() {
  const [health, setHealth] = useState(null);
  const [aoi, setAoi] = useState(DEFAULT_AOI);
  const [view, setView] = useState("2d"); // 2d | terrain | globe
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [layerState, setLayerState] = useState({}); // id -> { visible, opacity }

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message));
  }, []);

  const submit = useCallback(
    async (instruction) => {
      setLoading(true);
      setError(null);
      try {
        const res = await runQuery(instruction, aoi);
        setLayerState(defaultLayerState(res.layers));
        setResult(res);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    },
    [aoi],
  );

  const layers = result?.layers ?? [];
  const mapProps = { aoi, onAoiChange: setAoi, layers, layerState };

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="brand">
          <h1>Geo-VLA</h1>
          <p>Tool-augmented vision-language-action agent for geospatial reasoning</p>
          {health && (
            <div className="chips">
              <span className={`chip ${health.planner === "claude" ? "ok" : "warn"}`}>
                planner: {health.planner === "claude" ? health.model : "offline rules"}
              </span>
              <span className={`chip ${health.data_mode === "live" ? "ok" : "warn"}`}>data: {health.data_mode}</span>
              <span className={`chip ${health.checkpoints.classifier ? "ok" : "warn"}`}>
                classifier: {health.checkpoints.classifier ? "trained" : "fallback"}
              </span>
              <span className={`chip ${health.checkpoints.change_detector ? "ok" : "warn"}`}>
                change model: {health.checkpoints.change_detector ? "trained" : "fallback"}
              </span>
            </div>
          )}
        </header>

        <ChatPanel aoi={aoi} loading={loading} onSubmit={submit} />

        {error && <div className="error">⚠ {error}</div>}

        {result && (
          <section className="answer">
            <h2>Answer</h2>
            <div className="answer-body">
              {result.answer.split("\n").map((line, i) => (
                <p key={i}>{line.replace(/^_(.*)_$/, "$1")}</p>
              ))}
            </div>
            <div className="meta">
              {result.planner === "claude" ? result.model : "offline planner"} · data {result.data_mode}
              {result.usage && ` · ${result.usage.input_tokens + result.usage.output_tokens} tokens`}
            </div>
          </section>
        )}

        {result && <TracePanel trace={result.trace} />}
      </aside>

      <main className="map-area">
        <div className="view-toggle" role="tablist" aria-label="Map view">
          {[
            ["2d", "2D map"],
            ["terrain", "3D terrain"],
            ["globe", "Globe"],
          ].map(([id, label]) => (
            <button key={id} role="tab" aria-selected={view === id} className={view === id ? "active" : ""} onClick={() => setView(id)}>
              {label}
            </button>
          ))}
        </div>

        {view === "globe" ? (
          <Suspense fallback={<div className="map-loading">Loading 3D globe…</div>}>
            <GlobeView {...mapProps} />
          </Suspense>
        ) : (
          <MapView {...mapProps} terrain={view === "terrain"} />
        )}

        {layers.length > 0 && <LayerPanel layers={layers} layerState={layerState} onChange={setLayerState} />}
        {loading && <div className="map-busy">Agent is reasoning and running tools…</div>}
      </main>
    </div>
  );
}
