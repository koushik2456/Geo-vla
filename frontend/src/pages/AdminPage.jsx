import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";

const ROLES = ["public", "official", "admin"];

export default function AdminPage() {
  const { user: me } = useAuth();
  const [users, setUsers] = useState([]);
  const [f, setF] = useState({ username: "", password: "", role: "official", full_name: "", organization: "", email: "" });
  const [msg, setMsg] = useState(null);
  const load = useCallback(() => api("/admin/users").then(setUsers), []);

  useEffect(() => {
    load();
  }, [load]);

  const act = async (fn, ok) => {
    setMsg(null);
    try {
      await fn();
      if (ok) setMsg(ok);
      load();
    } catch (e) {
      setMsg(e.message);
    }
  };
  const create = (e) => {
    e.preventDefault();
    act(() => api("/admin/users", { method: "POST", body: f }), `Created ${f.username}`);
    setF({ ...f, username: "", password: "" });
  };
  const patch = (u, body, ok) => act(() => api(`/admin/users/${u.id}`, { method: "PATCH", body }), ok);
  const reset = (u) => {
    const pw = window.prompt(`New password for ${u.username} (min 8 characters)`);
    if (pw) patch(u, { password: pw }, `Password reset for ${u.username}`);
  };
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  return (
    <div className="page">
      <h1>Users</h1>
      {msg && <div className="info">{msg}</div>}
      <form className="card row wrap" onSubmit={create}>
        <div className="field"><label htmlFor="a-u">Username</label><input id="a-u" value={f.username} onChange={set("username")} required /></div>
        <div className="field"><label htmlFor="a-p">Temporary password</label><input id="a-p" type="password" value={f.password} onChange={set("password")} required minLength={8} /></div>
        <div className="field"><label htmlFor="a-r">Role</label><select id="a-r" value={f.role} onChange={set("role")}>{ROLES.map((r) => <option key={r}>{r}</option>)}</select></div>
        <div className="field"><label htmlFor="a-n">Full name</label><input id="a-n" value={f.full_name} onChange={set("full_name")} /></div>
        <div className="field"><label htmlFor="a-o">Department</label><input id="a-o" value={f.organization} onChange={set("organization")} /></div>
        <div className="field"><label htmlFor="a-e">Email</label><input id="a-e" type="email" value={f.email} onChange={set("email")} /></div>
        <button className="primary">Add user</button>
      </form>
      <section className="card">
        <table className="data-table">
          <thead><tr><th>User</th><th>Department</th><th>Role</th><th>Status</th><th /></tr></thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className={u.active ? "" : "muted"}>
                <td><strong>{u.username}</strong><div className="small muted">{u.full_name} {u.email && `· ${u.email}`}</div></td>
                <td className="small">{u.organization || "—"}</td>
                <td>
                  <select value={u.role} disabled={u.id === me.id} aria-label={`Role of ${u.username}`} onChange={(e) => patch(u, { role: e.target.value }, `${u.username} is now ${e.target.value}`)}>
                    {ROLES.map((r) => <option key={r}>{r}</option>)}
                  </select>
                </td>
                <td>{u.active ? "active" : "deactivated"}</td>
                <td className="row">
                  <button className="link" onClick={() => reset(u)}>Reset password</button>
                  {u.id !== me.id && <button className="link danger" onClick={() => patch(u, { active: !u.active })}>{u.active ? "Deactivate" : "Reactivate"}</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
