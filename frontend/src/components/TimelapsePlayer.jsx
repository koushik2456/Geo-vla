import React, { useEffect, useState } from "react";
import { Pause, Play } from "lucide-react";

const LABELS = { scene: "Imagery", scalar: "NDVI", classmap: "Land cover" };

/** Steps through a time series by toggling which dated layer of a group is visible. */
export default function TimelapsePlayer({ groups, onShow }) {
  const kinds = Object.keys(groups);
  const [kind, setKind] = useState(kinds.includes("scene") ? "scene" : kinds[0]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1200);
  const frames = groups[kind] ?? [];

  useEffect(() => {
    if (frames.length) onShow(frames, frames[Math.min(index, frames.length - 1)].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, kind]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setIndex((i) => (i + 1) % frames.length), speed);
    return () => clearInterval(t);
  }, [playing, speed, frames.length]);

  if (!frames.length) return null;
  const current = frames[Math.min(index, frames.length - 1)];
  return (
    <div className="timelapse glass" role="group" aria-label="Timelapse">
      <button className="icon" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"} title={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={14} strokeWidth={1.75} aria-hidden="true" /> : <Play size={14} strokeWidth={1.75} aria-hidden="true" />}
      </button>
      <input type="range" min={0} max={frames.length - 1} value={index} onChange={(e) => setIndex(Number(e.target.value))} aria-label="Frame" />
      <span className="timelapse-date mono">{current.meta.date}</span>
      {kinds.length > 1 && (
        <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Timelapse layer">
          {kinds.map((k) => (
            <option key={k} value={k}>
              {LABELS[k] ?? k}
            </option>
          ))}
        </select>
      )}
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Speed">
        <option value={2000}>Slow</option>
        <option value={1200}>Normal</option>
        <option value={600}>Fast</option>
      </select>
    </div>
  );
}
