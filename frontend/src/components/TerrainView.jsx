import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { api, withShare } from "../api.js";

/*
 * 3D terrain studio — renders the analysis's own DEM (no external tiles or tokens).
 *   surface      imagery / any texture draped on relief + result overlay + contours + flood water
 *   hypsometric  elevation-tinted relief, labelled contours, coordinate box
 *   contours     dense glowing contour lines floating at their elevation (survey look)
 *   stack        exploded layers: ALL · HEIGHT · NORMAL · AO (+ result)
 */

const MODES = [
  ["surface", "Surface"],
  ["hypsometric", "Hypsometric"],
  ["contours", "Contour lines"],
  ["stack", "Layer stack"],
];
const DERIVED = { hypsometric: "Hypsometric tint", hillshade: "Hillshade", height: "Height", normal: "Normal map", ao: "Ambient occlusion" };
const SIZE = 100; // scene units along the longer side

async function loadTexture(path, share) {
  const resp = await api(withShare(path, share));
  const url = URL.createObjectURL(await resp.blob());
  const tex = await new THREE.TextureLoader().loadAsync(url);
  URL.revokeObjectURL(url);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function decodeHeights(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Maps grid (row, col, metres) → scene coordinates for the current exaggeration. */
function makeFrame(model, exaggeration) {
  const longest = Math.max(model.width_m, model.height_m);
  const sx = (SIZE * model.width_m) / longest;
  const sz = (SIZE * model.height_m) / longest;
  const unit = SIZE / longest; // scene units per metre
  return {
    sx, sz,
    x: (c) => (c / (model.cols - 1) - 0.5) * sx,
    z: (r) => (r / (model.rows - 1) - 0.5) * sz,
    y: (h) => (h - model.min) * unit * exaggeration,
    top: (model.max - model.min) * unit * exaggeration,
  };
}

function terrainGeometry(model, heights, f) {
  const { rows, cols } = model;
  const pos = new Float32Array(rows * cols * 3);
  const uv = new Float32Array(rows * cols * 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      pos.set([f.x(c), f.y(heights[i]), f.z(r)], i * 3);
      uv.set([c / (cols - 1), 1 - r / (rows - 1)], i * 2);
    }
  }
  const idx = [];
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c;
      idx.push(a, a + cols, a + 1, a + 1, a + cols, a + cols + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Solid block sides from the surface edge down to a base, so the terrain reads as a slab. */
function skirtGeometry(model, heights, f, base) {
  const { rows, cols } = model;
  const edges = [
    [...Array(cols).keys()].map((c) => [0, c]),
    [...Array(rows).keys()].map((r) => [r, cols - 1]),
    [...Array(cols).keys()].reverse().map((c) => [rows - 1, c]),
    [...Array(rows).keys()].reverse().map((r) => [r, 0]),
  ];
  const pos = [];
  const col = [];
  for (const edge of edges) {
    for (let k = 0; k < edge.length - 1; k++) {
      const [r1, c1] = edge[k];
      const [r2, c2] = edge[k + 1];
      const h1 = f.y(heights[r1 * cols + c1]);
      const h2 = f.y(heights[r2 * cols + c2]);
      const p = [[f.x(c1), h1, f.z(r1)], [f.x(c1), base, f.z(r1)], [f.x(c2), h2, f.z(r2)], [f.x(c2), base, f.z(r2)]];
      for (const i of [0, 1, 2, 2, 1, 3]) {
        pos.push(...p[i]);
        const t = p[i][1] === base ? 0.05 : 0.22;
        col.push(t * 0.7, t * 0.8, t);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  return g;
}

const CONTOUR_RAMP = [[0, [0.35, 0.2, 0.85]], [0.35, [0.1, 0.55, 0.95]], [0.65, [0.1, 0.85, 0.8]], [1, [0.75, 0.95, 0.25]]];
function rampColor(t) {
  for (let i = 1; i < CONTOUR_RAMP.length; i++) {
    const [p1, c1] = CONTOUR_RAMP[i - 1];
    const [p2, c2] = CONTOUR_RAMP[i];
    if (t <= p2) {
      const k = (t - p1) / (p2 - p1 || 1);
      return c1.map((v, j) => v + (c2[j] - v) * k);
    }
  }
  return CONTOUR_RAMP.at(-1)[1];
}

function contourLines(model, sets, f, { color, lift = 0.05, opacity = 1, glow = false }) {
  const pos = [];
  const col = [];
  for (const set of sets) {
    const y = f.y(set.level) + lift;
    const t = (set.level - model.min) / (model.max - model.min || 1);
    const rgb = color ?? rampColor(t);
    for (const line of set.lines) {
      for (let k = 0; k < line.length - 1; k++) {
        pos.push(f.x(line[k][1]), y, f.z(line[k][0]), f.x(line[k + 1][1]), y, f.z(line[k + 1][0]));
        col.push(...rgb, ...rgb);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: opacity < 1 || glow, opacity,
    blending: glow ? THREE.AdditiveBlending : THREE.NormalBlending, depthWrite: !glow });
  return new THREE.LineSegments(g, m);
}

function label(text, className = "t3-label") {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  return new CSS2DObject(el);
}

function contourLabels(model, f, max = 14) {
  const group = new THREE.Group();
  const majors = model.contours.filter((c) => c.major);
  const pick = majors.length >= 3 ? majors : model.contours.filter((_, i) => i % 3 === 0);
  for (const set of pick.slice(0, max)) {
    const line = set.lines.reduce((a, b) => (b.length > a.length ? b : a), []);
    if (line.length < 6) continue;
    const [r, c] = line[Math.floor(line.length / 2)];
    const l = label(`${Math.round(set.level)}`, "t3-label contour");
    l.position.set(f.x(c), f.y(set.level) + 0.8, f.z(r));
    group.add(l);
  }
  return group;
}

function boundingBox(model, f, base) {
  const group = new THREE.Group();
  const box = new THREE.Box3(new THREE.Vector3(-f.sx / 2, base, -f.sz / 2), new THREE.Vector3(f.sx / 2, f.top + 2, f.sz / 2));
  group.add(new THREE.Box3Helper(box, new THREE.Color("#b04cff")));
  const [w, s, e, n] = model.bbox;
  const corners = [
    [-f.sx / 2, -f.sz / 2, `${n.toFixed(3)}°N ${w.toFixed(3)}°E`],
    [f.sx / 2, f.sz / 2, `${s.toFixed(3)}°N ${e.toFixed(3)}°E`],
  ];
  for (const [x, z, text] of corners) {
    const l = label(text, "t3-label coord");
    l.position.set(x, base - 1.5, z);
    group.add(l);
  }
  for (const [h, text] of [[model.min, `${Math.round(model.min)} m`], [model.max, `${Math.round(model.max)} m`]]) {
    const l = label(text, "t3-label elev");
    l.position.set(-f.sx / 2 - 2, f.y(h), f.sz / 2);
    group.add(l);
  }
  return group;
}

export default function TerrainView({ run, share }) {
  const mount = useRef(null);
  const three = useRef(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("surface");
  const [exaggeration, setExaggeration] = useState(3);
  const [explode, setExplode] = useState(1.4);
  const [showWater, setShowWater] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [textures, setTextures] = useState({});
  const [loading, setLoading] = useState(true);

  const rasterLayers = useMemo(() => run.layers.filter((l) => l.preview_url), [run.layers]);
  const scenes = rasterLayers.filter((l) => l.kind === "scene");
  const focus = rasterLayers.find((l) => l.id === run.insights?.focus_layer && l.kind !== "scene");
  const [base, setBase] = useState(scenes.at(-1) ? `layer:${scenes.at(-1).id}` : "derived:hypsometric");
  const [overlay, setOverlay] = useState(focus ? focus.id : "");

  // Terrain model
  useEffect(() => {
    let stop = false;
    setLoading(true);
    api(withShare(`/runs/${run.id}/terrain`, share))
      .then((m) => !stop && setModel({ ...m, heightsArr: decodeHeights(m.heights) }))
      .catch((e) => !stop && setError(e.message));
    return () => {
      stop = true;
    };
  }, [run.id, share]);

  // Textures needed by the current mode (cached)
  useEffect(() => {
    if (!model) return;
    const want = new Set();
    if (mode === "surface") {
      want.add(base);
      if (overlay) want.add(`layer:${overlay}`);
    }
    if (mode === "hypsometric") want.add("derived:hypsometric");
    if (mode === "stack") {
      want.add(base).add("derived:height").add("derived:normal").add("derived:ao");
      if (overlay) want.add(`layer:${overlay}`);
    }
    const missing = [...want].filter((k) => !textures[k]);
    if (!missing.length) return;
    let stop = false;
    setLoading(true);
    Promise.all(missing.map(async (key) => {
      const [kind, id] = key.split(":");
      const layer = rasterLayers.find((l) => l.id === id);
      const path = kind === "derived" ? `/runs/${run.id}/terrain/${id}.png` : `${layer.preview_url}?size=1024`;
      return [key, await loadTexture(path, share)];
    }))
      .then((pairs) => !stop && setTextures((t) => ({ ...t, ...Object.fromEntries(pairs) })))
      .catch((e) => !stop && setError(e.message));
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, mode, base, overlay]);

  // Renderer, camera, controls (once)
  useEffect(() => {
    const el = mount.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    el.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.domElement.className = "t3-labels";
    el.appendChild(labels.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#04060b");
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
    camera.position.set(70, 70, 95);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.set(0, 5, 0);
    scene.add(new THREE.HemisphereLight("#bcd7ff", "#1a1f2b", 1.6));
    const sun = new THREE.DirectionalLight("#fff4e0", 2.4);
    sun.position.set(-80, 120, -60);
    scene.add(sun);
    const world = new THREE.Group();
    scene.add(world);

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h);
      labels.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let frame;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    loop();
    three.current = { renderer, scene, camera, controls, world };
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.innerHTML = "";
      three.current = null;
    };
  }, []);

  useEffect(() => {
    if (three.current) three.current.controls.autoRotate = autoRotate;
  }, [autoRotate]);

  // Build the scene for the current mode
  useEffect(() => {
    const t = three.current;
    if (!t || !model) return;
    const { world } = t;
    world.traverse((o) => {
      o.geometry?.dispose?.();
      if (o.material) [].concat(o.material).forEach((m) => m.dispose());
      if (o.isCSS2DObject) o.element.remove();   // nested labels are not removed by clear()
    });
    world.clear();
    const f = makeFrame(model, exaggeration);
    const heights = model.heightsArr;
    const baseY = -Math.max(3, f.top * 0.12);
    const surface = terrainGeometry(model, heights, f);
    const tex = (k) => textures[k];
    const drape = (map, opts = {}) => new THREE.Mesh(surface, new THREE.MeshStandardMaterial({ map, roughness: 0.95, metalness: 0, ...opts }));
    const skirt = () => new THREE.Mesh(skirtGeometry(model, heights, f, baseY),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }));
    const overlayMesh = (texture) => new THREE.Mesh(surface, new THREE.MeshBasicMaterial({
      map: texture, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }));

    if (mode === "surface" || mode === "hypsometric") {
      const key = mode === "hypsometric" ? "derived:hypsometric" : base;
      if (!tex(key)) return;
      world.add(drape(tex(key)));
      world.add(skirt());
      if (mode === "surface" && overlay && tex(`layer:${overlay}`)) world.add(overlayMesh(tex(`layer:${overlay}`)));
      if (showContours || mode === "hypsometric") {
        world.add(contourLines(model, model.contours.filter((c) => !c.major), f,
          { color: mode === "hypsometric" ? [0.05, 0.05, 0.08] : [1, 1, 1], opacity: mode === "hypsometric" ? 0.55 : 0.25 }));
        world.add(contourLines(model, model.contours.filter((c) => c.major), f,
          { color: mode === "hypsometric" ? [0, 0, 0] : [1, 0.95, 0.7], opacity: 0.9 }));
        world.add(contourLabels(model, f));
      }
      if (mode === "hypsometric") world.add(boundingBox(model, f, baseY));
      if (showWater) {
        for (const w of model.water) {
          const plane = new THREE.Mesh(new THREE.PlaneGeometry(f.sx, f.sz), new THREE.MeshPhysicalMaterial({
            color: "#1e6bff", transparent: true, opacity: 0.62, roughness: 0.08, metalness: 0.1, clearcoat: 1 }));
          plane.rotation.x = -Math.PI / 2;
          plane.position.y = f.y(w.level);
          world.add(plane);
          const l = label(`Water ${Math.round(w.level)} m`, "t3-label water");
          l.position.set(f.sx / 2, f.y(w.level), -f.sz / 2);
          world.add(l);
        }
      }
    } else if (mode === "contours") {
      const dark = new THREE.Mesh(surface, new THREE.MeshBasicMaterial({ color: "#05070c", polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
      world.add(dark);
      world.add(contourLines(model, model.dense_contours, f, { lift: 0.02, opacity: 0.95 }));
      world.add(contourLines(model, model.contours.filter((c) => c.major), f, { glow: true, lift: 0.03 }));
      world.add(contourLabels(model, f));
    } else if (mode === "stack") {
      const slabs = [["ALL", base], ["HEIGHT", "derived:height"], ["NORMAL", "derived:normal"], ["AO", "derived:ao"]];
      if (slabs.some(([, k]) => !tex(k))) return;
      const gap = (f.top + 14) * explode;
      slabs.forEach(([name, key], i) => {
        const g = new THREE.Group();
        g.add(drape(tex(key), name === "NORMAL" ? { roughness: 0.6 } : {}));
        if (i === 0 && overlay && tex(`layer:${overlay}`)) g.add(overlayMesh(tex(`layer:${overlay}`)));
        const l = label(name, "t3-label slab");
        l.position.set(f.sx / 2 + 8, f.top * 0.4, 0);
        g.add(l);
        g.position.y = (slabs.length - 1 - i) * gap - gap * 1.2;
        world.add(g);
      });
    }
    setLoading(false);
  }, [model, textures, mode, exaggeration, explode, showWater, showContours, base, overlay]);

  // Frame the camera per mode
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    const presets = { surface: [85, 75, 120], hypsometric: [80, 75, 110], contours: [40, 110, 120], stack: [150, 70, 190] };
    t.camera.position.set(...presets[mode]);
    t.controls.target.set(0, mode === "stack" ? 0 : 5, 0);
  }, [mode]);

  const snapshot = () => {
    const a = document.createElement("a");
    a.href = three.current.renderer.domElement.toDataURL("image/png");
    a.download = `geo-vla-3d-${mode}.png`;
    a.click();
  };

  const water = model?.water ?? [];
  return (
    <div className="map-wrap terrain3d">
      <div ref={mount} className="map t3-canvas" />
      <div className="t3-panel glass">
        <div className="t3-modes" role="tablist" aria-label="3D render mode">
          {MODES.map(([id, labelText]) => (
            <button key={id} role="tab" aria-selected={mode === id} className={mode === id ? "active" : ""} onClick={() => setMode(id)}>
              {labelText}
            </button>
          ))}
        </div>
        {(mode === "surface" || mode === "stack") && (
          <label className="t3-field">
            <span>Surface</span>
            <select value={base} onChange={(e) => setBase(e.target.value)}>
              {scenes.map((l) => <option key={l.id} value={`layer:${l.id}`}>{l.name}</option>)}
              {Object.entries(DERIVED).map(([k, v]) => <option key={k} value={`derived:${k}`}>{v}</option>)}
            </select>
          </label>
        )}
        {(mode === "surface" || mode === "stack") && (
          <label className="t3-field">
            <span>Result overlay</span>
            <select value={overlay} onChange={(e) => setOverlay(e.target.value)}>
              <option value="">None</option>
              {rasterLayers.filter((l) => l.kind !== "scene").map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
            </select>
          </label>
        )}
        <label className="t3-field">
          <span>Vertical exaggeration ×{exaggeration}</span>
          <input type="range" min="1" max="12" step="0.5" value={exaggeration} onChange={(e) => setExaggeration(Number(e.target.value))} />
        </label>
        {mode === "stack" && (
          <label className="t3-field">
            <span>Explode</span>
            <input type="range" min="0.2" max="2" step="0.1" value={explode} onChange={(e) => setExplode(Number(e.target.value))} />
          </label>
        )}
        <div className="t3-toggles">
          {mode === "surface" && <label><input type="checkbox" checked={showContours} onChange={(e) => setShowContours(e.target.checked)} /> Contours</label>}
          {water.length > 0 && mode !== "contours" && mode !== "stack" && (
            <label><input type="checkbox" checked={showWater} onChange={(e) => setShowWater(e.target.checked)} /> Flood water</label>
          )}
          <label><input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} /> Rotate</label>
        </div>
        <button className="ghost" onClick={snapshot}>⤓ Snapshot PNG</button>
      </div>
      {model && (
        <div className="t3-legend glass">
          <div className="t3-ramp" />
          <div className="row between small">
            <span>{Math.round(model.min)} m</span>
            <span>{Math.round(model.max)} m</span>
          </div>
          <div className="small muted">
            Contours every {model.contour_interval} m · relief ×{exaggeration}
            {water.length > 0 && ` · water ${Math.round(water[0].level)} m`}
          </div>
        </div>
      )}
      {(loading || !model) && !error && <div className="map-loading"><span className="spinner" /> Building 3D terrain…</div>}
      {error && <div className="map-loading error">3D terrain unavailable: {error}</div>}
    </div>
  );
}
