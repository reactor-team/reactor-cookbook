"use client";

// One client surface.
//
// Every screen in this app talks to `useHappyOysterClient()`, never to the SDK
// hook directly. <LiveClientProvider> mounts the real <HappyOysterProvider> and
// adapts useHappyOyster() onto the surface below, rendering the live world into
// <HappyOysterVideo>.
//
// The surface is deliberately the shape of the SDK facade, so the adapter is a
// thin forwarding layer that keeps the SDK hook isolated to this file.

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  HappyOysterProvider,
  HappyOysterVideo,
  useHappyOyster,
  useHappyOysterTravelStatus,
} from "@reactor-models/happy-oyster/react";
import type {
  AdventureCommand,
  CreateWorldParams,
  HappyOysterMode,
  HappyOysterPhase,
  TravelStateMessage,
  WorldStateMessage,
} from "@reactor-models/happy-oyster";

export interface HappyOysterClient {
  phase: HappyOysterPhase;
  worldState: WorldStateMessage | null;
  travelState: TravelStateMessage | null;
  streaming: boolean;
  /** Seconds HappyOyster granted the live travel: null until one is open, and
   * for Directing travels, which run 3 min. */
  maxExperienceTimeSec: number | null;
  travelStatus: string;
  /** The last connect failure, humanized — cleared on the next attempt. */
  lastError: string | null;
  /** Open the Reactor session and sync the first world snapshot. */
  connect: () => Promise<void>;
  createWorld: (params: CreateWorldParams) => Promise<unknown>;
  attachWorld: (encryptedWorldId: string) => Promise<unknown>;
  startTravel: () => Promise<{ streaming: boolean }>;
  endTravelSession: () => Promise<void>;
  disconnect: () => Promise<void>;
  hold: (command: AdventureCommand) => Promise<void>;
  interact: (verb: string) => Promise<void>;
  release: (axes: {
    translation?: true;
    rotation?: true;
    interaction?: true;
  }) => Promise<void>;
  stop: () => Promise<void>;
  instruct: (content: string) => Promise<{ accepted: boolean }>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  rewind: (rewindToSec: number) => Promise<{ resumedAtSec: number }>;
}

interface ClientContextValue {
  client: HappyOysterClient;
  videoSlot: ReactNode;
}

const ClientContext = createContext<ClientContextValue | null>(null);

export function useHappyOysterClient(): HappyOysterClient {
  const ctx = useContext(ClientContext);
  if (!ctx) {
    throw new Error(
      "useHappyOysterClient must be used within a client provider",
    );
  }
  return ctx.client;
}

export function useVideoSlot(): ReactNode {
  const ctx = useContext(ClientContext);
  if (!ctx)
    throw new Error("useVideoSlot must be used within a client provider");
  return ctx.videoSlot;
}

// ── live ─────────────────────────────────────────────────────────────────────

// Local mode talks to a model served by the Reactor runtime on your own host
// (adventure on :8080, directing on :8081), skipping the Reactor Platform:
// connect() takes no JWT and there is no /tokens exchange. `local` lets the
// SDK pick the right per-mode port; an explicit NEXT_PUBLIC_REACTOR_API_URL
// always wins.
const LOCAL_RUNTIME = process.env.NEXT_PUBLIC_HO_LOCAL_RUNTIME === "1";
const REACTOR_API_URL = process.env.NEXT_PUBLIC_REACTOR_API_URL;

// The Reactor connection options, minus the mode the provider is mounted with.
const providerOptions = LOCAL_RUNTIME
  ? { local: true, ...(REACTOR_API_URL ? { apiUrl: REACTOR_API_URL } : {}) }
  : { apiUrl: REACTOR_API_URL ?? "https://api.reactor.inc" };

// JWT resolver: the SDK calls it on every Reactor Platform HTTP hop, so a
// short-lived token can't age out mid-session.
//
// The token is memoized here, in module scope, until shortly before it expires,
// and the fetch is no-store so the browser HTTP cache stays out of it. Holding
// the token in the app rather than in the browser cache is what makes its
// lifetime observable: a cache the app owns cannot be emptied out from under a
// live session by DevTools "Disable cache" or an eviction, and one mint then
// serves every hop of that session. (An app that downscopes its token with
// `authorization_details` needs this for correctness, since a session may only
// be operated by the token that created it. This route mints an account-scoped
// token, so here it is one round trip instead of many.)
const TOKEN_REFRESH_SKEW_MS = 60_000;
let cachedToken: { jwt: string; expiresAtMs: number } | null = null;
let inflightToken: Promise<string> | null = null;

async function fetchToken(): Promise<string> {
  if (
    cachedToken &&
    Date.now() < cachedToken.expiresAtMs - TOKEN_REFRESH_SKEW_MS
  ) {
    return cachedToken.jwt;
  }
  // Coalesce the parallel hops the SDK fires at connect time into one mint.
  if (inflightToken) return inflightToken;
  inflightToken = (async () => {
    try {
      const r = await fetch("/api/reactor/token", { cache: "no-store" });
      if (!r.ok) {
        const body = (await r.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `Token fetch failed: ${r.status}`);
      }
      const { jwt, expires_at } = (await r.json()) as {
        jwt: string;
        expires_at: number;
      };
      cachedToken = { jwt, expiresAtMs: expires_at * 1000 };
      return jwt;
    } finally {
      inflightToken = null;
    }
  })();
  return inflightToken;
}

// The mode is fixed for the life of the session — it picks which Reactor model
// (adventure or directing) the session connects to — so the provider is mounted
// (and keyed) on it and switching experiences remounts a fresh session.
export function LiveClientProvider({
  mode,
  children,
}: {
  mode: HappyOysterMode;
  children: ReactNode;
}) {
  return (
    <HappyOysterProvider mode={mode} {...providerOptions}>
      <LiveClientBridge>{children}</LiveClientBridge>
    </HappyOysterProvider>
  );
}

// Session-level failures (no GPU capacity, bad key, network) belong next to
// the Connect button, not in the console. 429 is the one users actually hit:
// Reactor had no free GPU for the session.
function describeError(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : String(cause);
  if (/\b429\b/.test(message))
    return "No GPU capacity available right now — try again in a moment.";
  return message;
}

function LiveClientBridge({ children }: { children: ReactNode }) {
  const ho = useHappyOyster();
  const [travelStatus, setTravelStatus] = useState("running");
  const [lastError, setLastError] = useState<string | null>(null);
  useHappyOysterTravelStatus(setTravelStatus);

  const connect = useCallback(() => {
    setLastError(null);
    return ho.connect(LOCAL_RUNTIME ? undefined : fetchToken).catch((cause) => {
      setLastError(describeError(cause));
      throw cause;
    });
  }, [ho]);

  const client = useMemo<HappyOysterClient>(
    () => ({
      phase: ho.phase,
      worldState: ho.worldState,
      travelState: ho.travelState,
      streaming: ho.streaming,
      maxExperienceTimeSec: ho.maxExperienceTimeSec,
      travelStatus,
      lastError,
      connect,
      createWorld: ho.createWorld,
      attachWorld: ho.attachWorld,
      startTravel: ho.startTravel,
      endTravelSession: ho.endTravelSession,
      disconnect: ho.disconnect,
      hold: ho.hold,
      interact: ho.interact,
      release: ho.release,
      stop: ho.stop,
      instruct: ho.instruct,
      pause: ho.pause,
      resume: ho.resume,
      rewind: ho.rewind,
    }),
    [ho, travelStatus, lastError, connect],
  );

  const value = useMemo<ClientContextValue>(
    () => ({
      client,
      videoSlot: (
        <HappyOysterVideo className="absolute inset-0 h-full w-full object-contain" />
      ),
    }),
    [client],
  );

  return (
    <ClientContext.Provider value={value}>{children}</ClientContext.Provider>
  );
}
