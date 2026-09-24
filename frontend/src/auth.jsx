import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, getToken, setToken } from "./api.js";

const AuthContext = createContext(null);
const RANK = { public: 0, official: 1, admin: 2 };

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [signupOpen, setSignupOpen] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const me = await api("/auth/me");
      setUser(me.user);
      setSignupOpen(me.signup_open);
      if (!me.user && getToken()) setToken(null);
    } catch {
      setUser(null);
    } finally {
      setReady(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const start = (session) => {
    setToken(session.token);
    setUser(session.user);
    return session.user;
  };
  const login = async (username, password) => start(await api("/auth/login", { method: "POST", body: { username, password } }));
  const signup = async (fields) => start(await api("/auth/signup", { method: "POST", body: fields }));
  const logout = async () => {
    try {
      await api("/auth/logout", { method: "POST" });
    } catch {
      /* already expired */
    }
    setToken(null);
    setUser(null);
  };
  const can = (role) => !!user && RANK[user.role] >= RANK[role];

  return <AuthContext.Provider value={{ user, ready, signupOpen, login, signup, logout, can, refresh }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
