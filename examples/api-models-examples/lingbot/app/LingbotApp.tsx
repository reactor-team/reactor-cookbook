"use client";

import { LingbotProvider } from "@reactor-models/lingbot";
import { Header } from "./components/Header";
import { StatusBadge } from "./components/StatusBadge";
import { CommandError } from "./components/CommandError";
import { NowPlaying } from "./components/NowPlaying";
import { MovementControls } from "./components/MovementControls";
import { DynamicEvents } from "./components/DynamicEvents";
import { ScenePicker } from "./components/ScenePicker";
import { CustomStart } from "./components/CustomStart";
import { SnapClip } from "./components/SnapClip";
import { Video } from "./components/Video";

// JWT resolver passed to <LingbotProvider jwtToken>. js-sdk 3.x accepts a static
// string or a resolver; the SDK calls a resolver on every Reactor API hop
// — uploads, clip manifests, ICE refreshes, SDP renegotiation — so a static
// string would 401 those hops the moment the token ages out.
//
// The token is memoized here, in module scope, until shortly before it expires,
// and the fetch is no-store so the browser HTTP cache stays out of it. A token
// is session-scoped: it may only operate the sessions it created, so every hop
// of a session has to present the same JWT. If a cache miss let the resolver
// mint a fresh token mid-session, the next upload or clip call would 403.
//
// Known edge this does not cover: a session created just before the memoized
// token expires is orphaned at the re-mint. Covering it needs a re-mint naming
// the live session in `authorization_details.resources.sessions.bind`.
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

// The client tree. LingbotProvider owns the WebRTC connection lifecycle —
// it auto-disconnects on unmount and on `beforeunload`, so don't call
// connect()/disconnect() from a useEffect yourself.
//
// We deliberately do NOT pass `autoConnect: true` here. The user clicks
// "Connect" so they see the disconnected → connecting → waiting → ready
// state machine first-hand. Flip it on in your own product if you'd
// rather skip straight to "ready".
//
// `getJwt` is an inline arrow on purpose. The provider auto-stabilizes
// it via `useRef + useMemo`, so a parent re-render does NOT tear the
// session down. Wrapping in `useCallback` is unnecessary.
export function LingbotApp() {
  return (
    <LingbotProvider jwtToken={fetchToken}>
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 lg:flex-row lg:gap-6 lg:p-6">
          {/*
           * The sidebar has two phases driven by `snapshot.started`:
           *
           *   - Setup  (waiting):    <ScenePicker />     + <CustomStart />
           *   - Live   (generating): <NowPlaying />      + <MovementControls />
           *                                              + <DynamicEvents />
           *
           * Each component subscribes to the snapshot via
           * `useLingbotState` and returns null when it's not its phase.
           * On disconnect, each component also clears its snapshot via
           * a small useEffect — keeps the UI from showing stale data
           * from the previous session after a reconnect.
           *
           * <DynamicEvents /> is the live-phase prompt-swap surface —
           * one click appends a curated world-event sentence ("rain
           * begins", "fog rolls in") to the active prompt and re-sends
           * via `set_prompt`. The model picks it up on the next chunk.
           *
           * <SnapClip /> is model-agnostic — it only needs the base SDK
           * to capture the last N seconds of the live stream — so it
           * sits at the bottom of the sidebar and is visible whenever
           * the connection is `"ready"`.
           */}
          <aside className="flex w-full flex-col gap-4 lg:w-80 lg:shrink-0">
            <StatusBadge />
            <CommandError />
            <NowPlaying />
            <MovementControls />
            <DynamicEvents />
            <ScenePicker />
            <CustomStart />
            <SnapClip />
          </aside>
          <section className="flex-1">
            <Video />
          </section>
        </main>
      </div>
    </LingbotProvider>
  );
}
