"use client";

// XMAX X2 — REFERENCE FRONTEND
//
// X2 is a real-time streaming video-to-video editing model: it edits
// whatever arrives on the `source` track, live, steered by a text
// prompt you can swap mid-stream, an optional reference image, and a
// drag-to-steer pointer on the output.
//
// The typed client comes from the published `@reactor-models/x2`
// package. It bakes the model name and tracks into <X2Provider> and
// exposes typed commands and per-message hooks, so the app reads the
// same as the sibling examples:
//
//   <X2Provider jwtToken={fetchToken} />     — session lifecycle
//   useX2()                                  — status + typed commands
//   setPrompt({ prompt })                    — model commands
//   useX2StateUpdate((msg) => …)             — model → client messages
//   <X2MainVideoView />                      — the live output
import { useEffect, useRef, useState } from "react";
import {
  X2Provider,
  useX2,
  useX2GenerationStopped,
  useX2StateUpdate,
} from "@reactor-models/x2";
import {
  DEFAULT_UI_STATE,
  type X2SourceMode,
  type X2UiState,
} from "@/app/lib/types";
import { reduce } from "@/app/lib/state";
import { Header } from "./components/Header";
import { StatusBadge } from "./components/StatusBadge";
import { CommandError } from "./components/CommandError";
import { SourcePanel } from "./components/SourcePanel";
import { ReferenceImage } from "./components/ReferenceImage";
import { PointerPanel } from "./components/PointerPanel";
import { Prompt } from "./components/Prompt";
import { Stage } from "./components/Stage";
import { SnapClip } from "./components/SnapClip";
import { useSourcePublisher } from "./components/useSourcePublisher";
import { REACTOR_API_URL } from "@/app/lib/config";

// JWT resolver passed to <X2Provider jwtToken>. js-sdk 3.x accepts a static
// string or a resolver; the SDK calls a resolver on every Reactor API request,
// which is what keeps uploads, clip manifests and ICE refreshes from 401ing
// once the token ages out.
//
// The token is memoized here, in module scope, until shortly before it expires,
// and the fetch is no-store so the browser HTTP cache stays out of it. A token
// is session-scoped: it may only operate the sessions it created, so every hop
// of a session has to present the same JWT. If a cache miss let the resolver
// mint a fresh token mid-session, the next upload or clip call would 403.
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
export function X2App() {
  return (
    <X2Provider
      apiUrl={REACTOR_API_URL}
      jwtToken={fetchToken}
      connectOptions={{ autoConnect: false }}
    >
      <Workspace />
    </X2Provider>
  );
}

const BANNER_TTL_MS = 6000;

// The client tree. The model is the source of truth: it broadcasts a full
// `state_update` snapshot on connect and after every observable change,
// reduced into X2UiState by lib/state.ts. generation_stopped (reset
// bookkeeping), reference_image_accepted (decoded dimensions), and
// command_error (transient banner) are handled as discrete events on top.
function Workspace() {
  const { status, lastError } = useX2();

  const [ui, setUi] = useState<X2UiState>(DEFAULT_UI_STATE);
  // Webcam is the default source; "video" streams a pre-recorded clip and
  // "image" repeats a still image as a constant feed (drag-to-animate). All
  // three feed the same `source` track.
  const [mode, setMode] = useState<X2SourceMode>("webcam");

  // Whichever source is active (webcam self-view in the panel, playing clip
  // or repeated canvas frame in the stage) produces a track and hands it
  // here; useSourcePublisher is the single owner of the `source` slot and
  // reconciles the wire to the latest track, so mode switches can't race
  // two publishers.
  const [sourceTrack, setSourceTrack] = useState<MediaStreamTrack | null>(null);
  const publishError = useSourcePublisher(sourceTrack);

  // URLs of the media selected in "video" / "image" mode (object URL for a
  // local file, or a preset's path). Owned here so the stage's input pane can
  // stream them; the setters revoke a replaced object URL.
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

  const [imageUrl, setImageUrlState] = useState<string | null>(null);
  const setImageUrl = (url: string | null) =>
    setImageUrlState((prev) => {
      if (prev && prev !== url && prev.startsWith("blob:")) {
        URL.revokeObjectURL(prev);
      }
      return url;
    });
  useEffect(() => {
    return () => {
      if (imageUrl && imageUrl.startsWith("blob:"))
        URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  // Command-failure banner: transient, not part of the reducer.
  //
  // The model reports a refusal as the command's own error reply now, not as a
  // broadcast `command_error` message — that message is gone from the schema, so
  // there is no `useX2CommandError` to subscribe to. A refused command resolves
  // `undefined` and the reason lands on the SDK's `lastError`, which the effect
  // below watches.
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

  // Bumped on a user reset and used as a React key on children that hold
  // local draft state (prompt draft, reference preview), remounting them in
  // step with the model's reset — which clears prompt, reference image, and
  // pointer server-side.
  const [resetNonce, setResetNonce] = useState(0);

  // While generation is stopped, black out the stage (the WebRTC view would
  // otherwise freeze on the last transformed frame). Lifts when generation
  // runs again.
  const [stageCleared, setStageCleared] = useState(false);

  // The model's snapshot is the source of truth: only the typed
  // `state_update` message feeds the reducer (see lib/state.ts).
  useX2StateUpdate((msg) => {
    if (msg.generating) setStageCleared(false);
    setUi((s) => reduce(s, msg));
  });

  useX2GenerationStopped((msg) => {
    setStageCleared(true);
    // A `reference_image_changed` stop is an automatic restart (a fresh
    // generation_started follows immediately) — keep drafts. Only a user
    // reset remounts the draft-holding children.
    if (msg.reason === "reset") setResetNonce((n) => n + 1);
  });

  // Surface a command failure. `lastError` is a persistent record — success
  // never clears it — so banner only a value that is new to this render, and
  // keep the last-seen one in a ref to tell "new failure" from "same failure
  // still recorded".
  const seenErrorRef = useRef(lastError);
  useEffect(() => {
    if (lastError && lastError !== seenErrorRef.current) {
      showCommandError(`[${lastError.code}] ${lastError.message}`);
    }
    seenErrorRef.current = lastError;
    // showCommandError is stable enough for this: it only touches refs and
    // setState.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastError]);

  // Reset local state on full disconnect so a reconnect starts clean.
  useEffect(() => {
    if (status === "disconnected") {
      setUi(DEFAULT_UI_STATE);
      setCommandError(null);
      setVideoUrl(null);
      setImageUrl(null);
      setStageCleared(false);
    }
  }, [status]);

  // Clean up the banner auto-dismiss timer on unmount.
  useEffect(() => {
    return () => {
      if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    };
  }, []);

  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <Header />
      {/* The stage comes first in the DOM: on mobile it is pinned (sticky)
          on top so the model output stays visible while the controls scroll
          beneath it, and the mobile padding lives on the children, not
          <main>, so it can sit flush against the viewport edge with a solid
          backdrop. lg:flex-row-reverse restores the desktop sidebar-left /
          stage-right split. */}
      <main className="flex flex-1 flex-col lg:min-h-0 lg:flex-row-reverse lg:gap-6 lg:p-6">
        <section className="flex flex-col gap-4 max-lg:sticky max-lg:top-0 max-lg:z-10 max-lg:bg-zinc-950/95 max-lg:p-4 max-lg:pb-3 max-lg:backdrop-blur-sm lg:min-h-0 lg:min-w-0 lg:flex-1 lg:overflow-hidden">
          <Stage
            ui={ui}
            mode={mode}
            videoUrl={videoUrl}
            imageUrl={imageUrl}
            cleared={stageCleared}
            onTrack={setSourceTrack}
          />
        </section>
        <aside className="flex w-full flex-col gap-4 p-4 pt-1 lg:w-80 lg:shrink-0 lg:min-h-0 lg:overflow-y-auto lg:p-0">
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
          <SourcePanel
            generating={ui.generating}
            keepBacklog={ui.keepBacklog}
            mode={mode}
            onModeChange={setMode}
            onSelectVideo={(url) => setVideoUrl(url)}
            onSelectImage={(url) => setImageUrl(url)}
            onTrack={setSourceTrack}
          />
          <Prompt key={`p${resetNonce}`} activePrompt={ui.activePrompt} />
          <PointerPanel />
          <ReferenceImage
            key={`r${resetNonce}`}
            generating={ui.generating}
            hasReference={ui.hasReference}
            accepted={ui.referenceAccepted}
            onAccepted={(size) =>
              setUi((s) => ({ ...s, referenceAccepted: size }))
            }
          />
          <SnapClip />
        </aside>
      </main>
    </div>
  );
}
