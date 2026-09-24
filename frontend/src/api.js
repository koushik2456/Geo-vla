const API = import.meta.env.VITE_API_URL || "";

export async function runQuery(instruction, bbox) {
  const resp = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction, bbox }),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

export async function getHealth() {
  const resp = await fetch(`${API}/health`);
  if (!resp.ok) throw new Error(`backend unavailable (${resp.status})`);
  return resp.json();
}
