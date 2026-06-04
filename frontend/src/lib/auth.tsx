import { ReactNode, createContext, useContext, useEffect, useMemo, useState } from "react";

import {
  fetchCurrentProfile,
  loginWithPassword,
  logoutSession,
  signUpWithPassword,
  type ProfilePayload,
  type UserSessionData,
} from "@/lib/api";

const STORAGE_KEY = "cropcare-auth-session";
const LEGACY_BROWSER_SESSION_PREFIX = ["local", "demo-token:"].join("-");

export type AuthUser = {
  id: string;
  email: string;
  fullName?: string | null;
  farmName?: string | null;
};

type StoredAuth = {
  token: string;
  refreshToken?: string | null;
  user: AuthUser;
};

type SignUpInput = {
  email: string;
  password: string;
  fullName: string;
  farmName?: string | null;
  phone?: string | null;
};

type AuthContextValue = {
  authReady: boolean;
  isAuthenticated: boolean;
  user: AuthUser | null;
  token: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (input: SignUpInput) => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readStoredAuth(): StoredAuth | null {
  if (typeof window === "undefined") {
    return null;
  }

  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (!stored) {
    return null;
  }

  try {
    const parsed = JSON.parse(stored) as StoredAuth;
    if (!parsed?.token || parsed.token.startsWith(LEGACY_BROWSER_SESSION_PREFIX)) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return parsed;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

function toUser(session: UserSessionData, profile?: ProfilePayload | null): AuthUser {
  return {
    id: session.user_id,
    email: profile?.email ?? session.email,
    fullName: profile?.full_name ?? session.email.split("@")[0],
    farmName: profile?.farm_name ?? null,
  };
}

function persistAuth(nextAuth: StoredAuth) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextAuth));
}

async function buildStoredAuth(sessionData: UserSessionData): Promise<StoredAuth> {
  const token = sessionData.session?.access_token;
  if (!token) {
    throw new Error("Backend did not return an access token.");
  }

  let profile: ProfilePayload | null = null;
  try {
    profile = (await fetchCurrentProfile(token)).data;
  } catch {
    profile = null;
  }

  return {
    token,
    refreshToken: sessionData.session?.refresh_token ?? null,
    user: toUser(sessionData, profile),
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<StoredAuth | null>(() => readStoredAuth());
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function validateStoredAuth() {
      if (!auth?.token) {
        if (!cancelled) {
          setAuthReady(true);
        }
        return;
      }

      try {
        const profile = (await fetchCurrentProfile(auth.token)).data;
        if (cancelled) {
          return;
        }

        const nextAuth: StoredAuth = {
          ...auth,
          user: {
            id: profile.id,
            email: profile.email,
            fullName: profile.full_name ?? profile.email.split("@")[0],
            farmName: profile.farm_name ?? null,
          },
        };
        persistAuth(nextAuth);
        setAuth(nextAuth);
      } catch {
        if (!cancelled) {
          window.localStorage.removeItem(STORAGE_KEY);
          setAuth(null);
        }
      } finally {
        if (!cancelled) {
          setAuthReady(true);
        }
      }
    }

    setAuthReady(false);
    void validateStoredAuth();
    return () => {
      cancelled = true;
    };
  }, [auth?.token]);

  const value = useMemo<AuthContextValue>(
    () => ({
      authReady,
      isAuthenticated: Boolean(auth?.token),
      user: auth?.user ?? null,
      token: auth?.token ?? null,
      signIn: async (email: string, password: string) => {
        const response = await loginWithPassword(email.trim().toLowerCase(), password);
        const nextAuth = await buildStoredAuth(response.data);
        persistAuth(nextAuth);
        setAuth(nextAuth);
      },
      signUp: async (input: SignUpInput) => {
        const normalizedEmail = input.email.trim().toLowerCase();
        if (!normalizedEmail) {
          throw new Error("Please enter an email address.");
        }
        const fullName = input.fullName.trim();
        if (fullName.length < 2) {
          throw new Error("Please enter your name.");
        }
        if (input.password.trim().length < 8) {
          throw new Error("Password must be at least 8 characters.");
        }

        const response = await signUpWithPassword({
          email: normalizedEmail,
          password: input.password,
          full_name: fullName,
          farm_name: input.farmName,
          phone: input.phone,
        });
        const nextAuth = await buildStoredAuth(response.data);
        persistAuth(nextAuth);
        setAuth(nextAuth);
      },
      signOut: async () => {
        const token = auth?.token;
        window.localStorage.removeItem(STORAGE_KEY);
        setAuth(null);
        if (token) {
          try {
            await logoutSession(token);
          } catch {
            // The client only needs to drop the local session if the server
            // already considers the token invalid.
          }
        }
      },
    }),
    [auth, authReady],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}
