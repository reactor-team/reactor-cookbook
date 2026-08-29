"use client";

import { LongliveV2Provider } from "@reactor-models/longlive-v2";
import { Header } from "./components/Header";
import { StatusBadge } from "./components/StatusBadge";
import { CommandError } from "./components/CommandError";
import { NowPlaying } from "./components/NowPlaying";
import { Storyboard } from "./components/Storyboard";
import { Director } from "./components/Director";
import { Timeline } from "./components/Timeline";
import { SnapClip } from "./components/SnapClip";
import { Video } from "./components/Video";

// JWT resolver passed to <LongliveV2Provider jwtToken>. js-sdk 3.x accepts a static
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

// The client tree. The sidebar is phase-driven by `snapshot.started`:
//   - Setup  (idle):       <Storyboard>  — compose shots & cuts, then start
//   - Live   (generating): <NowPlaying> + <Director> — drive it in real time
// Each component subscribes via `useLongliveV2State` and returns null when
// it's not its phase. The <Timeline> under the video visualizes the plan and
// the playhead in both phases. <SnapClip> is model-agnostic (base SDK).
//
// No `autoConnect` — the user clicks Connect so they see the
// disconnected → connecting → waiting → ready state machine first-hand.
export function LongLiveApp() {
  return (
    <LongliveV2Provider jwtToken={fetchToken}>
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 lg:flex-row lg:gap-6 lg:p-6">
          <aside className="flex w-full flex-col gap-4 lg:w-80 lg:shrink-0">
            <StatusBadge />
            <CommandError />
            <NowPlaying />
            <Storyboard />
            <Director />
            <SnapClip />
          </aside>
          <section className="flex flex-1 flex-col gap-4">
            <div className="flex-1">
              <Video />
            </div>
            <Timeline />
          </section>
        </main>
      </div>
    </LongliveV2Provider>
  );
}
