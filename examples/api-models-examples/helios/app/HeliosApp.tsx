"use client";

import { HeliosProvider } from "@reactor-models/helios";
import { Header } from "./components/Header";
import { StatusBadge } from "./components/StatusBadge";
import { CommandError } from "./components/CommandError";
import { NowPlaying } from "./components/NowPlaying";
import { EvolveScene } from "./components/EvolveScene";
import { PromptComposer } from "./components/PromptComposer";
import { ImageStarter } from "./components/ImageStarter";
import { SnapClip } from "./components/SnapClip";
import { Video } from "./components/Video";

// JWT resolver passed to <HeliosProvider jwtToken>.
//
// `@reactor-team/js-sdk` 3.x takes a `JwtSource` — a static string or a
// resolver. Pass the resolver so the SDK can mint a fresh JWT on every
// Reactor API hop — uploads, clip manifests, ICE refreshes, SDP
// renegotiation. With a static string those hops 401 the moment the
// token ages out.
//
// The token is memoized HERE, in module scope, until shortly before it
// expires — and the fetch itself is `no-store`, so the browser HTTP
// cache is out of the picture. This matters because the token is
// session-scoped: a session can only be operated by the exact token
// that created it, so every hop of a session must present the same JWT.
// Relying on the browser cache for that breaks the moment it misses
// (DevTools "Disable cache", cache eviction): the resolver then mints a
// fresh token with no bound sessions and every upload/clip call 403s.
//
// Known edge this does not cover: a session created just before the
// memoized token expires is orphaned at the re-mint (the fresh token
// isn't bound to it). Fixing that requires re-minting with
// `authorization_details.resources.sessions.bind` naming the live
// session.
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

// The client tree. HeliosProvider owns the WebRTC connection lifecycle —
// it auto-disconnects on unmount and on `beforeunload`, so don't call
// connect()/disconnect() from a useEffect yourself.
//
// We deliberately do NOT pass `autoConnect: true` here. The user clicks
// "Connect" so they see the disconnected → connecting → waiting → ready
// state machine first-hand. It's the most important Reactor mental
// model to teach.
//
// `jwtToken` takes the module-scope resolver above. The provider
// auto-stabilizes it via `useRef + useMemo`, so a parent re-render does
// NOT tear the session down — an inline arrow would be safe here too.
export function HeliosApp() {
  return (
    <HeliosProvider jwtToken={fetchToken}>
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 lg:flex-row lg:gap-6 lg:p-6">
          {/*
           * The sidebar has two phases driven by `snapshot.started`:
           *
           *   - Setup  (waiting):   <PromptComposer /> + <ImageStarter />
           *   - Live   (generating): <NowPlaying />     + <EvolveScene />
           *
           * Each component subscribes to the snapshot via
           * `useHeliosState` and returns null when it's not its phase.
           * On disconnect, each component also clears its snapshot via
           * a small useEffect — keeps the UI from showing stale data
           * from the previous session after a reconnect.
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
            <EvolveScene />
            <PromptComposer />
            <ImageStarter />
            <SnapClip />
          </aside>
          <section className="flex-1">
            <Video />
          </section>
        </main>
      </div>
    </HeliosProvider>
  );
}
