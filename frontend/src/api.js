// API client: bearer-token auth, share tokens for public links, authenticated downloads.
const BASE = import.meta.env.VITE_API_URL || "";
const TOKEN_KEY = "geovla.token";

let token = null;
try {
  token = window.localStorage.getItem(TOKEN_KEY);
} catch {
  /* storage unavailable (private mode) — session lasts until reload */
}

export function setToken(value) {
  token = value;
  try {
    if (value) window.localStorage.setItem(TOKEN_KEY, value);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

export const getToken = () => token;

/** Absolute URL for an API path (MapLibre/Cesium need absolute tile URLs).
 *  Built by concatenation: URL() would percent-encode the {z}/{x}/{y} placeholders. */
export function absolute(path) {
  const root = new URL(BASE || "/", window.location.href).href.replace(/\/$/, "");
  return root + path;
}

export function withShare(path, share) {
  if (!share) return path;
  return `${path}${path.includes("?") ? "&" : "?"}share=${encodeURIComponent(share)}`;
}

/** True for requests to our own API (so auth headers are attached to tiles). */
export function isApiUrl(url) {
  return url.startsWith(absolute("/"));
}

export const authHeader = () => (token ? { Authorization: `Bearer ${token}` } : {});

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { method = "GET", body, form, share } = {}) {
  const headers = { ...authHeader() };
  let payload;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const resp = await fetch(absolute(withShare(path, share)), { method, headers, body: payload });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      detail = typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : JSON.stringify(data.detail);
    } catch {
      /* non-JSON body */
    }
    throw new ApiError(resp.status, detail || `HTTP ${resp.status}`);
  }
  const type = resp.headers.get("content-type") || "";
  return type.includes("application/json") ? resp.json() : resp;
}

/** Fetch a file with auth headers and hand it to the browser as a download. */
export async function download(path, filename, share) {
  const resp = await api(path, { share });
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
