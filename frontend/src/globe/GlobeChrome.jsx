import React, { useEffect, useState } from "react";
import { ScreenSpaceEventHandler, ScreenSpaceEventType } from "cesium";
import {
  Box,
  Compass,
  Crosshair,
  Eye,
  Globe2,
  Layers,
  LocateFixed,
  Mountain,
  Minus,
  Plus,
} from "lucide-react";
import {
  BASEMAPS,
  cameraState,
  flyHome,
  flyToPoint,
  fmtDistance,
  fmtLatLon,
  google3dAvailable,
  pickCartographic,
  resetNorth,
  toggleTilt,
  zoom,
} from "./globeCore.js";

const ICO = { size: 14, strokeWidth: 1.75 };

/** Live cursor position + camera state for the HUD. */
export function useGlobeHud(viewer) {
  const [cursor, setCursor] = useState(null);
  const [camera, setCamera] = useState(null);
  useEffect(() => {
    if (!viewer) return undefined;
    let frame = 0;
    let last = null;
    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((move) => {
      last = move.endPosition;
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (!viewer.isDestroyed()) setCursor(pickCartographic(viewer, last));
      });
    }, ScreenSpaceEventType.MOUSE_MOVE);
    const onCamera = () => setCamera(cameraState(viewer));
    viewer.camera.percentageChanged = 0.002;
    const remove = viewer.camera.changed.addEventListener(onCamera);
    const removeEnd = viewer.camera.moveEnd.addEventListener(onCamera);
    onCamera();
    return () => {
      cancelAnimationFrame(frame);
      handler.destroy();
      remove();
      removeEnd();
    };
  }, [viewer]);
  return { cursor, camera };
}

export function GlobeHud({ hud, extra }) {
  const { cursor, camera } = hud;
  return (
    <div className="globe-hud" aria-live="off">
      <span title="Cursor position">
        <span className="hud-label"><Crosshair {...ICO} aria-hidden="true" /> Cursor</span>
        {cursor ? `${fmtLatLon(cursor.lat, cursor.lon)} · ${cursor.lat.toFixed(5)}, ${cursor.lon.toFixed(5)}` : "—"}
      </span>
      {cursor?.height != null && (
        <span title="Ground elevation under the cursor">
          <span className="hud-label"><Mountain {...ICO} aria-hidden="true" /> Elev</span>
          {fmtDistance(cursor.height)}
        </span>
      )}
      {camera && (
        <span title="Eye altitude">
          <span className="hud-label"><Eye {...ICO} aria-hidden="true" /> Alt</span>
          {fmtDistance(camera.altitude)}
        </span>
      )}
      {camera && (
        <span title="Heading / tilt">
          <span className="hud-label">Hdg</span>
          {Math.round((camera.heading + 360) % 360)}° · tilt {Math.round(90 + camera.pitch)}°
        </span>
      )}
      {extra}
    </div>
  );
}

export function GlobeControls({ viewer, cfg, basemap, onBasemap, camera }) {
  const [open, setOpen] = useState(false);
  const heading = camera?.heading ?? 0;
  const locate = () =>
    navigator.geolocation?.getCurrentPosition(
      (p) => flyToPoint(viewer, p.coords.longitude, p.coords.latitude, { range: 2500 }),
      () => alert("Location permission denied"),
    );
  if (!viewer) return null;
  return (
    <>
      <div className="globe-controls" role="toolbar" aria-label="Globe navigation">
        <button className="compass" title="Reset north" aria-label="Reset north" onClick={() => resetNorth(viewer)}>
          <Compass {...ICO} style={{ transform: `rotate(${-heading}deg)` }} aria-hidden="true" />
        </button>
        <button title="Zoom in" aria-label="Zoom in" onClick={() => zoom(viewer, 0.45)}>
          <Plus {...ICO} aria-hidden="true" />
        </button>
        <button title="Zoom out" aria-label="Zoom out" onClick={() => zoom(viewer, -0.8)}>
          <Minus {...ICO} aria-hidden="true" />
        </button>
        <button title="Tilt: 2D / 3D view" aria-label="Toggle tilt" onClick={() => toggleTilt(viewer)}>
          <Box {...ICO} aria-hidden="true" />
        </button>
        <button title="Whole Earth" aria-label="Whole Earth" onClick={() => flyHome(viewer)}>
          <Globe2 {...ICO} aria-hidden="true" />
        </button>
        <button title="My location" aria-label="My location" onClick={locate}>
          <LocateFixed {...ICO} aria-hidden="true" />
        </button>
      </div>
      <div className="basemap-switch">
        <button className="basemap-current" onClick={() => setOpen((o) => !o)} aria-expanded={open} title="Basemap">
          <Layers {...ICO} aria-hidden="true" />
          {BASEMAPS.find((b) => b.id === basemap)?.label ?? "Map"}
        </button>
        {open && (
          <div className="basemap-menu" onMouseLeave={() => setOpen(false)}>
            {BASEMAPS.map((b) => {
              const disabled = b.id === "google3d" && !google3dAvailable(cfg);
              return (
                <button key={b.id} className={b.id === basemap ? "active" : ""} disabled={disabled}
                  title={disabled ? "Add GOOGLE_MAPS_API_KEY (or CESIUM_ION_TOKEN) to .env" : b.hint}
                  onClick={() => { setOpen(false); onBasemap(b.id); }}>
                  <strong>{b.label}</strong>
                  <span>{disabled ? "needs a Google Maps key" : b.hint}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
