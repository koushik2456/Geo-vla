import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Cartesian2,
  Cartesian3,
  ClassificationType,
  Color,
  HeightReference,
  LabelStyle,
  Rectangle,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  VerticalOrigin,
} from "cesium";
import { api } from "../api.js";
import { navigate } from "../router.js";
import PlaceSearch from "../components/PlaceSearch.jsx";
import { GlobeControls, GlobeHud, useGlobeHud } from "../globe/GlobeChrome.jsx";
import {
  clientConfig,
  createViewer,
  flyToBbox,
  flyToPoint,
  fmtDistance,
  fmtLatLon,
  google3dAvailable,
  googleEarthUrl,
  googleMapsUrl,
  pickCartographic,
  setBasemap,
} from "../globe/globeCore.js";

const monthAgo = () => new Date(Date.now() - 30 * 864e5).toISOString().slice(0, 10);
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1)}%`;

function labelFor(text) {
  return {
    text,
    font: "600 13px Inter, system-ui, sans-serif",
    fillColor: Color.WHITE,
    outlineColor: Color.BLACK,
    outlineWidth: 3,
    style: LabelStyle.FILL_AND_OUTLINE,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -16),
    showBackground: true,
    backgroundColor: Color.fromCssColorString("#0b1020").withAlpha(0.78),
    heightReference: HeightReference.CLAMP_TO_GROUND,
    disableDepthTestDistance: Number.POSITIVE_INFINITY,
  };
}

function addPin(viewer, lon, lat, color, text) {
  return viewer.entities.add({
    position: Cartesian3.fromDegrees(lon, lat),
    point: {
      pixelSize: 12,
      color: Color.fromCssColorString(color),
      outlineColor: Color.WHITE,
      outlineWidth: 2,
      heightReference: HeightReference.CLAMP_TO_GROUND,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: text ? labelFor(text) : undefined,
  });
}

function addGrid(viewer, grid) {
  const out = [];
  grid.forEach((cell) => {
    const color = Color.fromCssColorString(cell.color);
    const [w, s, e, n] = cell.bbox;
    const centre = cell.row === 1 && cell.col === 1;
    out.push(viewer.entities.add({
      rectangle: { coordinates: Rectangle.fromDegrees(w, s, e, n), material: color.withAlpha(centre ? 0.38 : 0.24), classificationType: ClassificationType.BOTH },
    }));
    out.push(viewer.entities.add({
      polyline: {
        positions: Cartesian3.fromDegreesArray([w, s, e, s, e, n, w, n, w, s]),
        width: centre ? 3.5 : 1.5,
        material: centre ? Color.WHITE : color,
        clampToGround: true,
        classificationType: ClassificationType.BOTH,
      },
    }));
  });
  return out;
}

// -- panels --------------------------------------------------------------------------------

function PlaceCard({ info, point, onAnalyze }) {
  if (!point) return null;
  const { lon, lat } = point;
  const place = info?.place;
  const comps = place?.components ?? {};
  const copy = () => navigator.clipboard?.writeText(`${lat.toFixed(6)}, ${lon.toFixed(6)}`);
  return (
    <section className="xp-card">
      <div className="xp-eyebrow">Location</div>
      <h3>{place ? place.name : "Looking up…"}</h3>
      {place && <p className="small">{place.formatted_address}</p>}
      {place && (
        <div className="xp-chips">
          {["suburb", "locality", "subdistrict", "district", "state", "country", "postcode"]
            .filter((k) => comps[k] && comps[k] !== place.name)
            .map((k) => <span key={k} className="chip" title={k}>{comps[k]}</span>)}
        </div>
      )}
      <dl className="xp-facts">
        <dt>Coordinates</dt>
        <dd className="mono">{lat.toFixed(6)}, {lon.toFixed(6)} <button className="link" onClick={copy}>copy</button></dd>
        <dt>DMS</dt>
        <dd className="mono">{fmtLatLon(lat, lon)}</dd>
        {info?.utm && (<><dt>UTM</dt><dd className="mono">zone {info.utm.zone} · EPSG:{info.utm.epsg}</dd></>)}
        {place?.plus_code && (<><dt>Plus code</dt><dd className="mono">{place.plus_code}</dd></>)}
        <dt>Elevation</dt>
        <dd>
          {info?.elevation?.m != null ? `${fmtDistance(info.elevation.m)} a.s.l.` : info ? "unavailable" : "…"}
          {info?.elevation?.local_min_m != null && <span className="muted small"> · 600 m window {Math.round(info.elevation.local_min_m)}–{Math.round(info.elevation.local_max_m)} m</span>}
        </dd>
        {place && (<><dt>Source</dt><dd className="small muted">{place.source}{info?.elevation?.source ? ` · ${info.elevation.source}` : ""}</dd></>)}
      </dl>
      <div className="xp-actions">
        <a className="button ghost" href={googleMapsUrl(lat, lon)} target="_blank" rel="noreferrer">Google Maps ↗</a>
        <a className="button ghost" href={googleEarthUrl(lat, lon)} target="_blank" rel="noreferrer">Google Earth ↗</a>
        <button className="button" onClick={onAnalyze}>Analyze this area →</button>
      </div>
    </section>
  );
}

function SpectralChart({ signature }) {
  const W = 260, H = 110, P = 26;
  const max = Math.max(0.3, ...signature.map((s) => s.reflectance)) * 1.1;
  const x = (nm) => P + ((nm - 470) / (860 - 470)) * (W - P - 8);
  const y = (r) => H - 20 - (r / max) * (H - 32);
  const pts = signature.map((s) => `${x(s.nm)},${y(s.reflectance)}`).join(" ");
  return (
    <svg className="xp-spectral" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Spectral signature">
      {[0, max / 2, max].map((v) => (
        <g key={v}>
          <line x1={P} x2={W - 6} y1={y(v)} y2={y(v)} className="grid" />
          <text x={P - 4} y={y(v) + 3} textAnchor="end">{v.toFixed(2)}</text>
        </g>
      ))}
      <polyline points={pts} className="line" />
      {signature.map((s) => (
        <g key={s.band}>
          <circle cx={x(s.nm)} cy={y(s.reflectance)} r="3.5" className="dot" />
          <text x={x(s.nm)} y={H - 6} textAnchor="middle">{s.band.split(" ")[1]}</text>
        </g>
      ))}
    </svg>
  );
}

function NeuralPanel({ result, busy, error, date, setDate, onRetry }) {
  return (
    <section className="xp-card xp-nn">
      <div className="xp-eyebrow">Neural network · what the model sees here</div>
      <label className="xp-date small">
        Imagery around
        <input type="date" value={date} max={new Date().toISOString().slice(0, 10)} onChange={(e) => setDate(e.target.value)} />
        {result && <button className="link" onClick={onRetry}>re-run</button>}
      </label>
      {busy && <div className="xp-busy"><span className="spinner" aria-hidden="true" /> Fetching Sentinel-2 and running the network…</div>}
      {error && <div className="error">⚠ {error}</div>}
      {result && !busy && (
        <>
          <div className="xp-pred" style={{ "--c": result.prediction.color }}>
            <span className="swatch" />
            <div>
              <strong>{result.prediction.class}</strong>
              <span className="small muted">
                confidence {pct(result.prediction.confidence)} · entropy {result.prediction.entropy_bits} bits
              </span>
            </div>
          </div>
          <p className="small muted">{result.method} · {result.imagery} · {result.date}</p>

          <h4>Class probabilities (softmax)</h4>
          <ul className="xp-bars">
            {result.prediction.top5.map((t) => (
              <li key={t.class}>
                <span>{t.class}</span>
                <span className="bar"><i style={{ width: `${Math.max(t.p * 100, 1)}%`, background: t.color }} /></span>
                <span className="mono">{pct(t.p)}</span>
              </li>
            ))}
          </ul>

          <div className="xp-images">
            <figure>
              <img src={result.context.image} alt="3×3 neighbourhood with predicted class outlines" />
              <figcaption>3×3 neighbourhood · {(result.patch.size_m * 3 / 1000).toFixed(2)} km</figcaption>
            </figure>
            <figure>
              <img src={result.patch.image} alt="Centre patch fed to the network" className="pixelated" />
              <figcaption>Input patch · {result.patch.size_px}×{result.patch.size_px} px = {result.patch.size_m} m</figcaption>
            </figure>
            {result.network && (
              <figure>
                <img src={result.network.gradcam} alt="Grad-CAM heatmap" />
                <figcaption>Grad-CAM · where it looked</figcaption>
              </figure>
            )}
          </div>

          <h4>Spectral signature (centre patch)</h4>
          <div className="xp-spec">
            <SpectralChart signature={result.spectral.signature} />
            <dl>
              <dt>NDVI</dt><dd className="mono">{result.spectral.ndvi}</dd>
              <dt>NDWI</dt><dd className="mono">{result.spectral.ndwi}</dd>
              <dt>Brightness</dt><dd className="mono">{result.spectral.brightness}</dd>
            </dl>
          </div>

          {result.network ? (
            <>
              <h4>Inside ResNet-50: most active feature maps per stage</h4>
              <p className="small muted">
                Input tensor {result.network.input.tensor_shape.join("×")}, {result.network.input.normalisation}. Deeper stages have more,
                smaller maps that respond to larger, more abstract patterns.
              </p>
              {result.network.stages.map((s) => (
                <div key={s.stage} className="xp-stage">
                  <div className="xp-stage-head">
                    <strong>{s.stage}</strong>
                    <span className="mono small">{s.shape.join("×")}</span>
                    <span className="small muted">mean {s.mean_activation} · {pct(s.sparsity)} zero</span>
                  </div>
                  <img src={s.image} alt={`${s.stage} feature maps`} className="pixelated" />
                </div>
              ))}
              <details>
                <summary>Logits and embedding</summary>
                <table className="xp-logits">
                  <tbody>
                    {Object.entries(result.network.logits).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td className="mono">{v.toFixed(3)}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="small">
                  Embedding: {result.network.embedding.dim}-d (global average pool), ‖z‖ = {result.network.embedding.l2_norm},{" "}
                  {pct(result.network.embedding.active_fraction)} units active. Strongest units: {result.network.embedding.top_units.join(", ")}.
                </p>
              </details>
            </>
          ) : (
            <div className="info small">
              No trained classifier is active, so these are rule-based spectral classes. Train one in the <a href="#/studio">Training studio</a>{" "}
              (the 1-minute classroom preset is enough) to see Grad-CAM and the network's feature maps here.
            </div>
          )}
        </>
      )}
      {!result && !busy && !error && <p className="small muted">Click anywhere on the globe to classify the 640 m patch under the cursor.</p>}
    </section>
  );
}

// -- page ----------------------------------------------------------------------------------

export default function ExplorePage({ setAoi, setPlace }) {
  const container = useRef(null);
  const [viewer, setViewer] = useState(null);
  const [cfg, setCfg] = useState(null);
  const [basemap, setBasemapState] = useState(null);
  const [mode, setMode] = useState("nn");
  const [point, setPoint] = useState(null);
  const [info, setInfo] = useState(null);
  const [nn, setNn] = useState({ result: null, busy: false, error: null });
  const [date, setDate] = useState(monthAgo);
  const [history, setHistory] = useState([]);
  const hud = useGlobeHud(viewer);
  const entities = useRef({ pin: null, grid: [] });
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const dateRef = useRef(date);
  dateRef.current = date;

  // Create the globe once the map keys are known.
  useEffect(() => {
    let v;
    let cancelled = false;
    clientConfig().then(async (c) => {
      if (cancelled) return;
      setCfg(c);
      v = createViewer(container.current, c);
      v.camera.setView({ destination: Cartesian3.fromDegrees(79, 18, 21_000_000) });
      setViewer(v);
      setBasemapState(await setBasemap(v, google3dAvailable(c) ? "google3d" : "hybrid"));
      // Slow rotation until the user takes over.
      const spin = () => v.camera.rotate(Cartesian3.UNIT_Z, -0.0012);
      const stop = () => {
        v.scene.postRender.removeEventListener(spin);
        v.canvas.removeEventListener("pointerdown", stop);
        v.canvas.removeEventListener("wheel", stop);
      };
      v.scene.postRender.addEventListener(spin);
      v.canvas.addEventListener("pointerdown", stop);
      v.canvas.addEventListener("wheel", stop);
      v._geo.stopSpin = stop;
    });
    return () => {
      cancelled = true;
      if (v && !v.isDestroyed()) v.destroy();
    };
  }, []);

  const changeBasemap = async (id) => setBasemapState(await setBasemap(viewer, id));

  const requestId = useRef(0);
  const classify = useCallback(async (lon, lat) => {
    const id = ++requestId.current;
    setNn({ result: null, busy: true, error: null });
    try {
      const r = await api("/explore/classify", { method: "POST", body: { lon, lat, date: dateRef.current } });
      if (id !== requestId.current) return; // a newer click superseded this one
      setNn({ result: r, busy: false, error: null });
      if (viewer && !viewer.isDestroyed()) {
        entities.current.grid.forEach((e) => viewer.entities.remove(e));
        entities.current.grid = addGrid(viewer, r.context.grid);
        setHistory((h) => [{ lon, lat, cls: r.prediction.class, p: r.prediction.confidence, color: r.prediction.color,
          pin: addPin(viewer, lon, lat, r.prediction.color) }, ...h].slice(0, 30));
      }
    } catch (e) {
      if (id === requestId.current) setNn({ result: null, busy: false, error: e.message });
    }
  }, [viewer]);

  const inspect = useCallback(async (lon, lat, { fly = false } = {}) => {
    viewer?._geo.stopSpin?.();
    setPoint({ lon, lat });
    setInfo(null);
    if (viewer) {
      if (entities.current.pin) viewer.entities.remove(entities.current.pin);
      entities.current.pin = addPin(viewer, lon, lat, "#22d3ee");
      if (fly || viewer.camera.positionCartographic.height > 60_000) flyToPoint(viewer, lon, lat, { range: 4200 });
    }
    api(`/explore/point?lon=${lon}&lat=${lat}`)
      .then((d) => {
        setInfo(d);
        if (viewer && entities.current.pin) entities.current.pin.label = labelFor(d.place?.name ?? "");
      })
      .catch(() => setInfo({ place: { name: "Unknown place", formatted_address: "", components: {}, source: "unavailable" } }));
    if (modeRef.current === "nn") classify(lon, lat);
  }, [viewer, classify]);

  useEffect(() => {
    if (!viewer) return undefined;
    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((click) => {
      const p = pickCartographic(viewer, click.position);
      if (p) inspect(p.lon, p.lat);
    }, ScreenSpaceEventType.LEFT_CLICK);
    return () => handler.destroy();
  }, [viewer, inspect]);

  const analyze = () => {
    const d = 0.03;
    setAoi([+(point.lon - d).toFixed(5), +(point.lat - d).toFixed(5), +(point.lon + d).toFixed(5), +(point.lat + d).toFixed(5)]);
    setPlace(info?.place?.name ?? "Explored area");
    navigate("/");
  };

  const clearHistory = () => {
    history.forEach((h) => viewer.entities.remove(h.pin));
    entities.current.grid.forEach((e) => viewer.entities.remove(e));
    entities.current.grid = [];
    setHistory([]);
  };

  return (
    <div className="explore">
      <div ref={container} className="explore-globe" />

      <aside className="xp-panel">
        <div className="xp-search">
          <PlaceSearch onSelect={(r) => { viewer?._geo.stopSpin?.(); flyToBbox(viewer, r.bbox); inspect(r.center[0], r.center[1]); }} />
        </div>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={mode === "nn"} className={mode === "nn" ? "active" : ""} onClick={() => setMode("nn")}>
            🧠 Classify with the network
          </button>
          <button role="tab" aria-selected={mode === "place"} className={mode === "place" ? "active" : ""} onClick={() => setMode("place")}>
            📍 Place details only
          </button>
        </div>
        {!point && (
          <div className="xp-card xp-welcome">
            <div className="xp-eyebrow">Globe Explorer</div>
            <h3>Fly anywhere, click to explore</h3>
            <p className="small">
              Search a place or click the globe. You get the exact address, coordinates, elevation and UTM zone. In <b>network</b> mode
              Geo-VLA also fetches Sentinel-2 imagery there, classifies the 640 m patch with the ResNet-50 and shows what it looked at.
            </p>
            <p className="small muted">
              Drag to pan · right-drag or scroll to zoom · Ctrl/middle-drag to tilt and rotate.
              {cfg && !google3dAvailable(cfg) && " Add GOOGLE_MAPS_API_KEY to .env for Google Photorealistic 3D."}
            </p>
          </div>
        )}
        <PlaceCard info={info} point={point} onAnalyze={analyze} />
        {mode === "nn" && point && (
          <NeuralPanel {...nn} date={date} setDate={setDate} onRetry={() => classify(point.lon, point.lat)} />
        )}
        {history.length > 0 && (
          <section className="xp-card">
            <div className="xp-eyebrow">Explored patches <button className="link" onClick={clearHistory}>clear</button></div>
            <ul className="xp-history">
              {history.map((h, i) => (
                <li key={`${h.lon}-${h.lat}-${i}`}>
                  <button className="link" onClick={() => inspect(h.lon, h.lat, { fly: true })}>
                    <span className="swatch" style={{ background: h.color }} /> {h.cls} <span className="muted">{pct(h.p)}</span>
                    <span className="mono small muted"> {h.lat.toFixed(3)}, {h.lon.toFixed(3)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </aside>

      <GlobeControls viewer={viewer} cfg={cfg} basemap={basemap} onBasemap={changeBasemap} camera={hud.camera} />
      <GlobeHud hud={hud} extra={cfg && <span className="muted">{cfg.geocoder === "google" ? "Google geocoding" : cfg.geocoder === "nominatim" ? "OSM geocoding" : "offline gazetteer"} · {cfg.imagery_source === "earthengine" ? "Earth Engine" : cfg.imagery_source === "copernicus" ? "Copernicus" : "demo imagery"}</span>} />
    </div>
  );
}
