"use client";

import {
  SanaStreamingProvider,
  useSanaStreaming,
  useSanaStreamingCommandError,
  useSanaStreamingState,
} from "@reactor-models/sana-streaming";
import { useEffect, useRef, useState } from "react";
import { DEFAULT_STATE, type SanaMode } from "./lib/types";
import { reduce } from "./lib/state";
import { Header } from "./components/Header";
import { StatusBadge } from "./components/StatusBadge";
import { CommandError } from "./components/CommandError";
import { ModeInput } from "./components/ModeInput";
import { Prompt } from "./components/Prompt";
import { Stage } from "./components/Stage";
import { SnapClip } from "./components/SnapClip";
import { useCameraPublisher } from "./components/useCameraPublisher";

// JWT resolver passed to <SanaStreamingProvider jwtToken>. js-sdk 3.x accepts a static
// string or a resolver; the SDK calls a resolver on every Reactor API hop
// - uploads, clip manifests, ICE refreshes, SDP renegotiation - so a static
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

// No `autoConnect`: the user clicks Connect so they see the
// disconnected -> connecting -> waiting -> ready state machine first-hand.
// SanaStreamingProvider wraps ReactorProvider with the model name and tracks
// baked in, so commands and messages are typed all the way down.
export function SanaStreamingApp() {
  return (
    <SanaStreamingProvider jwtToken={fetchToken}>
      <Workspace />
    </SanaStreamingProvider>
  );
}

const BANNER_TTL_MS = 6000;

// The client tree. The model is the source of truth: only `state` messages
// mutate the reducer, and every control gates off the reduced SanaState
// rather than local guesses. Everything else (command_error banner,
// generation_reset bookkeeping) is handled imperatively here.
function Workspace() {
  const { status } = useSanaStreaming();

  const [state, setState] = useState(DEFAULT_STATE);
  // Webcam is the default source; switch to a clip to stream a pre-recorded
  // video into the model instead. Both feed the same `camera` track.
  const [mode, setMode] = useState<SanaMode>("webcam");

  // The active input source (webcam self-view or the video pane) produces a
  // track; one owner publishes it to `camera`. See useCameraPublisher.
  const [camTrack, setCamTrack] = useState<MediaStreamTrack | null>(null);
  const publishError = useCameraPublisher(camTrack);

  // URL of the clip selected in "video" mode (object URL for a local file, or
  // a preset's path). Owned here so the stage's input pane can stream it; the
  // setter revokes the previous object URL.
  const [videoUrl, setVideoUrlState] = useState<string | null>(null);
  const setVideoUrl = (url: string | null) =>
    setVideoUrlState((prev) => {
      if (prev && prev !== url && prev.startsWith("blob:")) {
        URL.revokeObjectURL(prev);
      }
      return url;
    });
  useEffect(() => {
    return () => {
      if (videoUrl && videoUrl.startsWith("blob:"))
        URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  // command_error banner: transient, not part of the reducer.
  const [commandError, setCommandError] = useState<string | null>(null);
  const bannerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showCommandError = (reason: string) => {
    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    setCommandError(reason);
    bannerTimerRef.current = setTimeout(
      () => setCommandError(null),
      BANNER_TTL_MS,
    );
  };

  // Bumped on generation_reset and used as a React key on the children that
  // hold local draft state (prompt draft), remounting them in step with the
  // model's reset.
  const [resetNonce, setResetNonce] = useState(0);

  // After reset, black out the stage (the WebRTC view would otherwise freeze
  // on the last transformed frame). Lifts when generation runs again.
  const [stageCleared, setStageCleared] = useState(false);
  useEffect(() => {
    if (state.running) setStageCleared(false);
  }, [state.running]);

  // The model is the source of truth: only the typed `state` snapshot feeds
  // the reducer. command_error is handled imperatively below, through its
  // own typed hook; the reset cleanup runs off the reply to `reset()`
  // instead, since that message is correlated to the caller rather than
  // broadcast (see handleGenerationReset).
  useSanaStreamingState((msg) => {
    setState((s) => reduce(s, msg));
  });

  useSanaStreamingCommandError((msg) => {
    showCommandError(msg.reason);
  });

  // Invoked by Playback when `reset()` resolves with `generation_reset`.
  const handleGenerationReset = () => {
    setResetNonce((n) => n + 1);
    setStageCleared(true);
  };

  // Reset local state on full disconnect so a reconnect starts clean.
  useEffect(() => {
    if (status === "disconnected") {
      setState(DEFAULT_STATE);
      setCommandError(null);
      setVideoUrl(null);
    }
  }, [status]);

  // Clean up the banner auto-dismiss timer on unmount.
  useEffect(() => {
    return () => {
      if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    };
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      {/* The stage comes first in the DOM: on mobile it is pinned (sticky)
          on top so the model output stays visible while the controls scroll
          beneath it, and the mobile padding lives on the children, not
          <main>, so it can sit flush against the viewport edge with a solid
          backdrop. lg:flex-row-reverse restores the desktop sidebar-left /
          stage-right split. */}
      <main className="flex flex-1 flex-col lg:flex-row-reverse lg:gap-6 lg:p-6">
        <section className="flex flex-col gap-4 max-lg:sticky max-lg:top-0 max-lg:z-10 max-lg:bg-zinc-950/95 max-lg:p-4 max-lg:pb-3 max-lg:backdrop-blur-sm lg:min-w-0 lg:flex-1">
          <Stage
            state={state}
            mode={mode}
            videoUrl={videoUrl}
            cleared={stageCleared}
            onTrack={setCamTrack}
          />
        </section>
        <aside className="flex w-full flex-col gap-4 p-4 pt-1 lg:w-80 lg:shrink-0 lg:p-0">
          <StatusBadge />
          {commandError && (
            <CommandError
              message={commandError}
              onDismiss={() => setCommandError(null)}
            />
          )}
          {publishError && (
            <p className="text-xs text-red-400">
              Publish error: {publishError}
            </p>
          )}
          <ModeInput
            started={state.started}
            paused={state.paused}
            mode={mode}
            modelSeed={state.seed}
            hasVideoUrl={!!videoUrl}
            onModeChange={setMode}
            onSelectVideo={(url) => setVideoUrl(url)}
            onTrack={setCamTrack}
            onReset={handleGenerationReset}
          />
          <Prompt key={resetNonce} currentPrompt={state.currentPrompt} />
          <SnapClip />
        </aside>
      </main>
    </div>
  );
}
