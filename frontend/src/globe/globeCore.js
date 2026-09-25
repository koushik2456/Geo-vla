// Shared Cesium globe setup: basemaps (Google Photorealistic 3D Tiles, Esri satellite,
// labels, OSM, dark), result overlays that drape on both the globe and 3D tiles,
// picking and camera helpers for the HUD.
import {
  Cartesian3,
  Cartographic,
  createGooglePhotorealistic3DTileset,
  HeadingPitchRange,
  BoundingSphere,
  Color,
  ImageryLayer,
  Ion,
  Math as CesiumMath,
  OpenStreetMapImageryProvider,
  Rectangle,
  Terrain,
  UrlTemplateImageryProvider,
  Viewer,
  EllipsoidTerrainProvider,
  defined,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { api } from "../api.js";

let configPromise = null;

/** Map keys served by the backend (.env), with Vite env vars as a fallback for static hosting. */
export function clientConfig() {
  configPromise ??= api("/client-config").catch(() => ({}));
  return configPromise.then((c) => ({
    ...c,
    google_maps_key: c.google_maps_key || import.meta.env.VITE_GOOGLE_MAPS_API_KEY || null,
    cesium_ion_token: c.cesium_ion_token || import.meta.env.VITE_CESIUM_ION_TOKEN || null,
  }));
}

// Deep ocean blue shown before imagery loads (avoids the default bright blue flash).
const DEEP_SEA = Color.fromCssColorString("#0a1a2f");

export const google3dAvailable = (cfg) => !!(cfg?.google_maps_key || cfg?.cesium_ion_token);

export const BASEMAPS = [
  { id: "google3d", label: "Google 3D", hint: "Google Photorealistic 3D Tiles: 3D cities, terrain and landmarks" },
  { id: "hybrid", label: "Satellite + labels", hint: "Esri World Imagery with places, borders and roads" },
  { id: "satellite", label: "Satellite", hint: "Esri World Imagery (Maxar, Airbus)" },
  { id: "streets", label: "Streets", hint: "OpenStreetMap" },
  { id: "dark", label: "Dark", hint: "CARTO dark matter" },
];

const esri = (service, credit) =>
  new UrlTemplateImageryProvider({
    url: `https://server.arcgisonline.com/ArcGIS/rest/services/${service}/MapServer/tile/{z}/{y}/{x}`,
    credit,
    maximumLevel: 19,
  });

function basemapProviders(id) {
  switch (id) {
    case "streets":
      return [new OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" })];
    case "dark":
      return [new UrlTemplateImageryProvider({ url: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", credit: "© OpenStreetMap contributors © CARTO", maximumLevel: 19 })];
    case "hybrid":
      return [
        esri("World_Imagery", "Imagery © Esri, Maxar, Earthstar Geographics"),
        esri("Reference/World_Transportation", "Roads © Esri"),
        esri("Reference/World_Boundaries_and_Places", "Labels © Esri"),
      ];
    default:
      return [esri("World_Imagery", "Imagery © Esri, Maxar, Earthstar Geographics")];
  }
}

export function createViewer(container, cfg) {
  if (cfg.cesium_ion_token) Ion.defaultAccessToken = cfg.cesium_ion_token;
  const viewer = new Viewer(container, {
    baseLayer: false,
    terrain: cfg.cesium_ion_token ? Terrain.fromWorldTerrain({ requestWaterMask: true }) : undefined,
    baseLayerPicker: false,
    geocoder: false,
    animation: false,
    timeline: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    msaaSamples: 4,
  });
  const { scene } = viewer;
  scene.globe.showGroundAtmosphere = true;
  scene.globe.baseColor = DEEP_SEA;
  scene.fog.enabled = true;
  scene.highDynamicRange = false;
  scene.screenSpaceCameraController.enableCollisionDetection = true;
  scene.globe.depthTestAgainstTerrain = !!cfg.cesium_ion_token;
  viewer._geo = { cfg, base: [], tileset: null, basemap: null, overlays: new Set() };
  return viewer;
}

/** Switch basemap. Returns the id actually applied (falls back to hybrid if Google 3D fails). */
export async function setBasemap(viewer, id) {
  const st = viewer._geo;
  st.base.forEach((layer) => viewer.imageryLayers.remove(layer, true));
  st.base = [];
  if (id === "google3d") {
    try {
      if (!st.tileset) {
        const tileset = await createGooglePhotorealistic3DTileset(
          { key: st.cfg.google_maps_key || undefined, onlyUsingWithGoogleGeocoder: true },
          { showCreditsOnScreen: true, maximumScreenSpaceError: 8 },
        );
        if (viewer.isDestroyed()) return id;
        st.tileset = viewer.scene.primitives.add(tileset);
        st.overlays.forEach((o) => o.attachTileset(st.tileset));
      }
      st.tileset.show = true;
      viewer.scene.globe.show = false;
      st.basemap = id;
      return id;
    } catch (e) {
      console.warn("Google Photorealistic 3D Tiles unavailable", e);
      id = "hybrid";
    }
  }
  if (st.tileset) st.tileset.show = false;
  viewer.scene.globe.show = true;
  basemapProviders(id).forEach((provider, i) => st.base.push(viewer.imageryLayers.addImageryProvider(provider, i)));
  st.basemap = id;
  return id;
}

/** A raster overlay (analysis result) shown on the globe and draped on the 3D tiles when present. */
export class Overlay {
  constructor(viewer, provider) {
    this.viewer = viewer;
    this.provider = provider;
    this.layers = [viewer.imageryLayers.addImageryProvider(provider)];
    this.show = true;
    this.alpha = 1;
    if (viewer._geo.tileset) this.attachTileset(viewer._geo.tileset);
    viewer._geo.overlays.add(this);
  }

  attachTileset(tileset) {
    try {
      const layer = new ImageryLayer(this.provider);
      tileset.imageryLayers.add(layer);
      layer.show = this.show;
      layer.alpha = this.alpha;
      this.tilesetLayer = { tileset, layer };
      this.layers.push(layer);
    } catch (e) {
      console.warn("overlay cannot drape on 3D tiles", e);
    }
  }

  set(show, alpha) {
    this.show = show;
    this.alpha = alpha;
    this.layers.forEach((l) => {
      l.show = show;
      l.alpha = alpha;
    });
  }

  remove() {
    if (this.viewer.isDestroyed()) return;
    this.viewer.imageryLayers.remove(this.layers[0], true);
    if (this.tilesetLayer) this.tilesetLayer.tileset.imageryLayers.remove(this.tilesetLayer.layer, true);
    this.viewer._geo.overlays.delete(this);
  }
}

/** World position under a screen point: 3D tiles / terrain surface when available, else the ellipsoid. */
export function pickCartographic(viewer, windowPosition) {
  const { scene, camera } = viewer;
  let cartesian;
  if (viewer._geo.tileset?.show && scene.pickPositionSupported) {
    cartesian = scene.pickPosition(windowPosition);
  }
  if (!defined(cartesian)) {
    const ray = camera.getPickRay(windowPosition);
    cartesian = ray && scene.globe.pick(ray, scene);
  }
  if (!defined(cartesian)) cartesian = camera.pickEllipsoid(windowPosition, scene.globe.ellipsoid);
  if (!defined(cartesian)) return null;
  const c = Cartographic.fromCartesian(cartesian);
  let height = c.height;
  if (!viewer._geo.tileset?.show) {
    height = viewer.terrainProvider instanceof EllipsoidTerrainProvider ? null : scene.globe.getHeight(c) ?? null;
  }
  return { lon: CesiumMath.toDegrees(c.longitude), lat: CesiumMath.toDegrees(c.latitude), height };
}

export function cameraState(viewer) {
  const cam = viewer.camera;
  const c = cam.positionCartographic;
  return {
    altitude: c.height,
    heading: CesiumMath.toDegrees(cam.heading),
    pitch: CesiumMath.toDegrees(cam.pitch),
    lon: CesiumMath.toDegrees(c.longitude),
    lat: CesiumMath.toDegrees(c.latitude),
  };
}

/** Oblique fly-to a point, Google-Earth style. `range` = metres from the target. */
export function flyToPoint(viewer, lon, lat, { range = 1800, pitch = -35, heading = 0, duration = 2.4 } = {}) {
  const target = Cartesian3.fromDegrees(lon, lat, 0);
  viewer.camera.flyToBoundingSphere(new BoundingSphere(target, 1), {
    offset: new HeadingPitchRange(CesiumMath.toRadians(heading), CesiumMath.toRadians(pitch), range),
    duration,
  });
}

/** Fly to a bbox, tilted when it is small enough to see relief. */
export function flyToBbox(viewer, bbox, duration = 2.4) {
  const [w, s, e, n] = bbox;
  const spanKm = Math.max((e - w) * 111 * Math.cos(((s + n) / 2) * Math.PI / 180), (n - s) * 111);
  if (spanKm > 150) {
    viewer.camera.flyTo({ destination: Rectangle.fromDegrees(w, s, e, n), duration });
    return;
  }
  flyToPoint(viewer, (w + e) / 2, (s + n) / 2, { range: Math.max(spanKm * 1400, 900), duration });
}

export function flyHome(viewer, duration = 2.0) {
  viewer.camera.flyTo({ destination: Cartesian3.fromDegrees(79, 20, 16_000_000), duration });
}

export function zoom(viewer, factor) {
  const h = viewer.camera.positionCartographic.height;
  if (factor > 0) viewer.camera.zoomIn(h * factor);
  else viewer.camera.zoomOut(h * -factor);
}

export function resetNorth(viewer) {
  const cam = viewer.camera;
  const centre = pickCartographic(viewer, { x: viewer.canvas.clientWidth / 2, y: viewer.canvas.clientHeight / 2 });
  if (!centre || cam.positionCartographic.height > 3_000_000) {
    cam.flyTo({ destination: cam.positionWC.clone(), orientation: { heading: 0, pitch: cam.pitch, roll: 0 }, duration: 0.8 });
    return;
  }
  const target = Cartesian3.fromDegrees(centre.lon, centre.lat, centre.height ?? 0);
  const range = Cartesian3.distance(cam.positionWC, target);
  cam.flyToBoundingSphere(new BoundingSphere(target, 1), { offset: new HeadingPitchRange(0, cam.pitch, range), duration: 0.8 });
}

export function toggleTilt(viewer) {
  const cam = viewer.camera;
  const centre = pickCartographic(viewer, { x: viewer.canvas.clientWidth / 2, y: viewer.canvas.clientHeight / 2 });
  if (!centre) return;
  const target = Cartesian3.fromDegrees(centre.lon, centre.lat, centre.height ?? 0);
  const range = Cartesian3.distance(cam.positionWC, target);
  const pitch = CesiumMath.toDegrees(cam.pitch) < -80 ? -35 : -90;
  cam.flyToBoundingSphere(new BoundingSphere(target, 1), {
    offset: new HeadingPitchRange(cam.heading, CesiumMath.toRadians(pitch), range),
    duration: 0.9,
  });
}

// -- formatting --------------------------------------------------------------------

export function dms(value, pos, neg) {
  const hemi = value >= 0 ? pos : neg;
  const v = Math.abs(value);
  const d = Math.floor(v);
  const mFloat = (v - d) * 60;
  const m = Math.floor(mFloat);
  const s = ((mFloat - m) * 60).toFixed(1);
  return `${d}°${String(m).padStart(2, "0")}′${s.padStart(4, "0")}″${hemi}`;
}

export const fmtLatLon = (lat, lon) => `${dms(lat, "N", "S")} ${dms(lon, "E", "W")}`;

export function fmtDistance(m) {
  if (m == null || !Number.isFinite(m)) return "–";
  if (Math.abs(m) >= 10_000) return `${(m / 1000).toFixed(0)} km`;
  if (Math.abs(m) >= 1000) return `${(m / 1000).toFixed(1)} km`;
  return `${Math.round(m)} m`;
}

export const googleMapsUrl = (lat, lon) => `https://www.google.com/maps/@${lat.toFixed(6)},${lon.toFixed(6)},800m/data=!3m1!1e3`;
export const googleEarthUrl = (lat, lon, range = 1500) =>
  `https://earth.google.com/web/@${lat.toFixed(6)},${lon.toFixed(6)},0a,${range}d,35y,0h,45t,0r`;
