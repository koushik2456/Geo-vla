import React from "react";

function Json({ value }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

/** The transparent reasoning trace: every thought, tool call, input and output, in order. */
export default function TracePanel({ trace, running }) {
  let step = 0;
  return (
    <section className="trace">
      <ol>
        {trace.map((item, i) => {
          if (item.type === "tool_call") {
            step += 1;
            return (
              <li key={i} className={`tool ${item.is_error ? "failed" : ""}`}>
                <details>
                  <summary>
                    <span className="step mono">{String(step).padStart(2, "0")}</span>
                    <code className="tool-name">{item.tool}</code>
                    <span className="duration mono">{item.duration_ms} ms</span>
                    {item.is_error && <span className="chip bad">error</span>}
                  </summary>
                  <div className="io">
                    <h3>input</h3>
                    <Json value={item.input} />
                    <h3>output</h3>
                    <Json value={item.output} />
                  </div>
                </details>
              </li>
            );
          }
          return (
            <li key={i} className={item.type}>
              <span className="label">{item.type === "thinking" ? "thinking" : "plan"}</span>
              <p>{item.text}</p>
            </li>
          );
        })}
        {running && (
          <li className="pending">
            <span className="spinner" aria-hidden="true" /> working…
          </li>
        )}
      </ol>
    </section>
  );
}
