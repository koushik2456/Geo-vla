import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

/** Search villages, cities, districts, lakes, forests; selecting sets the analysis area. */
export default function PlaceSearch({ onSelect }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);
  const chosen = useRef(null);

  useEffect(() => {
    clearTimeout(timer.current);
    if (q.trim().length < 2 || q === chosen.current) {
      setResults([]);
      return;
    }
    timer.current = setTimeout(async () => {
      setBusy(true);
      try {
        setResults(await api(`/geocode?q=${encodeURIComponent(q.trim())}`));
        setOpen(true);
      } catch {
        setResults([]);
      } finally {
        setBusy(false);
      }
    }, 350);
    return () => clearTimeout(timer.current);
  }, [q]);

  const choose = (r) => {
    onSelect(r);
    chosen.current = r.name;
    setQ(r.name);
    setOpen(false);
  };

  return (
    <div className="place-search">
      <input type="search" value={q} placeholder="Search a place — village, city, lake, forest…" aria-label="Search a place"
        onChange={(e) => setQ(e.target.value)} onFocus={() => results.length && setOpen(true)}
        onKeyDown={(e) => e.key === "Enter" && results[0] && choose(results[0])} />
      {busy && <span className="spinner" aria-hidden="true" />}
      {open && results.length > 0 && (
        <ul className="place-results" role="listbox">
          {results.map((r) => (
            <li key={`${r.label}-${r.center}`} role="option" aria-selected="false" onMouseDown={() => choose(r)}>
              <strong>{r.name}</strong>
              <span>{r.label}</span>
              <em>
                {r.type}
                {r.clipped ? " · centre area" : ""}
              </em>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
