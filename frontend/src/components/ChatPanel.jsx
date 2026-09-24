import React, { useState } from "react";

const EXAMPLES = [
  "Show me where deforestation increased near rivers in the last 3 years and estimate terrain slope in those zones.",
  "Compare the imagery between 2021-06-01 and 2025-06-01 and tell me what fraction of the area changed.",
  "If the river rises by 8 m, how much land floods and what land cover is affected?",
  "How healthy is the vegetation within 1 km of major roads?",
];

export default function ChatPanel({ aoi, loading, onSubmit }) {
  const [text, setText] = useState(EXAMPLES[0]);

  const submit = (e) => {
    e.preventDefault();
    if (text.trim() && !loading) onSubmit(text.trim());
  };

  return (
    <section className="chat">
      <form onSubmit={submit}>
        <label htmlFor="instruction">Instruction</label>
        <textarea
          id="instruction"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
          placeholder="Ask a geospatial question about the selected area…"
        />
        <div className="aoi-line">
          AOI: [{aoi.map((v) => v.toFixed(3)).join(", ")}] — draw a new one on the map
        </div>
        <button type="submit" disabled={loading || !text.trim()}>
          {loading ? "Running…" : "Run agent"}
        </button>
      </form>
      <details className="examples">
        <summary>Example instructions</summary>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="example" onClick={() => setText(ex)}>
            {ex}
          </button>
        ))}
      </details>
    </section>
  );
}
