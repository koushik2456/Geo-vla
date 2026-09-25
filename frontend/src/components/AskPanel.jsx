import React, { useState } from "react";

const EXAMPLES = [
  "Show me where deforestation increased near rivers in the last 3 years and estimate terrain slope in those zones.",
  "Which built-up areas would flood if the river rises 6 m, and how many km of road are affected?",
  "How has vegetation health changed each year since 2020?",
  "What share of cropland within 1 km of major roads looks stressed right now?",
];

export default function AskPanel({ busy, onRun, planner }) {
  const [text, setText] = useState(EXAMPLES[0]);
  const submit = (e) => {
    e.preventDefault();
    if (text.trim() && !busy) onRun({ instruction: text.trim() });
  };
  return (
    <form className="ask" onSubmit={submit}>
      <label htmlFor="instruction">Instruction</label>
      <textarea id="instruction" rows={4} value={text} onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && (e.metaKey || e.ctrlKey) && submit(e)}
        placeholder="Describe the analysis for the selected area…" />
      <p className="hint">
        {planner === "llm" ? "Agent plans and executes tools." : "Offline planner. Set GROQ_API_KEY for the LLM agent."}
      </p>
      <button type="submit" className="primary" disabled={busy || !text.trim()}>
        {busy ? "Running…" : "Run"}
      </button>
      <details className="examples">
        <summary>Examples</summary>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="example" onClick={() => setText(ex)}>
            {ex}
          </button>
        ))}
      </details>
    </form>
  );
}
