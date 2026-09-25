import React, { useState } from "react";
import { useAuth } from "../auth.jsx";
import { navigate } from "../router.js";

export default function LoginPage() {
  const { login, signup, signupOpen } = useAuth();
  const [mode, setMode] = useState("login");
  const [f, setF] = useState({ username: "", password: "", full_name: "", organization: "", email: "" });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") await login(f.username, f.password);
      else await signup(f);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page narrow">
      <form className="card auth-card" onSubmit={submit}>
        <div className="brand">Geo-VLA</div>
        <h1>{mode === "login" ? "Sign in" : "Create account"}</h1>
        <div className="field">
          <label htmlFor="u">Username</label>
          <input id="u" autoComplete="username" value={f.username} onChange={set("username")} required />
        </div>
        <div className="field">
          <label htmlFor="p">Password</label>
          <input id="p" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={f.password} onChange={set("password")} required minLength={mode === "login" ? 1 : 8} />
        </div>
        {mode === "signup" && (
          <>
            <div className="field">
              <label htmlFor="n">Full name</label>
              <input id="n" value={f.full_name} onChange={set("full_name")} />
            </div>
            <div className="field">
              <label htmlFor="o">Organisation / department</label>
              <input id="o" value={f.organization} onChange={set("organization")} />
            </div>
            <div className="field">
              <label htmlFor="e">Email</label>
              <input id="e" type="email" value={f.email} onChange={set("email")} />
            </div>
          </>
        )}
        {error && <div className="error">{error}</div>}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
        {signupOpen && (
          <button type="button" className="link mode-switch" onClick={() => setMode(mode === "login" ? "signup" : "login")}>
            {mode === "login" ? "Create account" : "Sign in"}
          </button>
        )}
      </form>
    </div>
  );
}
