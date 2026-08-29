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

// JWT resolver passed to <HeliosProvider getJwt>.
//
// `@reactor-team/js-sdk` ≥ 2.10.1 takes a resolver instead of a static
// string so the SDK can mint a fresh JWT on every Coordinator HTTP hop
// — uploads, clip manifests, ICE refreshes, SDP renegotiation. With a
// static string those hops 401 the moment the token ages out.
//
// We don't write a cache layer here either. The route returns the JWT
// with `Cache-Control: private, max-age=<seconds-until-expiry>`, so
// the browser's HTTP cache serves repeat calls (after a reload, route
// change, HMR cycle, etc.) without ever hitting our server — until
// the JWT actually expires.
async function fetchToken(): Promise<string> {
  const r = await fetch("/api/reactor/token");
  if (!r.ok) {
    const body = (await r.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `Token fetch failed: ${r.status}`);
  }
  const { jwt } = (await r.json()) as { jwt: string };
  return jwt;
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
// `getJwt` is an inline arrow on purpose. The provider auto-stabilizes
// it via `useRef + useMemo`, so a parent re-render does NOT tear the
// session down. Wrapping in `useCallback` is unnecessary.
export function HeliosApp() {
  return (
    <HeliosProvider getJwt={fetchToken}>
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
