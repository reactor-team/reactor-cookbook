"use client";

// LTX — REFERENCE FRONTEND
//
// ltx2 is a talking-head model. You give it a still image and a script;
// it generates the voice and the lip-synced video together and streams both
// back over WebRTC, window by window. There is no inbound media track — the
// avatar image is a file upload and the speech is generated from text.
//
// The shape of this app follows from one property of the model: a run is
// atomic. A take is generated from the conditions as they stood at `start`,
// and nothing you change mid-run alters the take you are watching. But the
// conditions stay editable throughout — the model accepts each change, applies
// it to the NEXT take, and reports the field in `state_update.queued_changes`.
// So the UI never blocks an edit and never tracks pending edits itself; it
// sends every change straight to the wire and renders the queue off the
// snapshot. See app/lib/machine.ts for the two questions every component asks.
//
// The typed client comes from `@reactor-models/ltx2`, generated from
// the model's schema. Every command goes through a typed method — there are no
// hand-written command strings anywhere in this app:
//
//   <Ltx2Provider jwtToken={fetchToken} /> — session lifecycle
//   useLtx2()                              — status + typed commands
//   setScript({ script })                        — model commands
//   useLtx2StateUpdate((msg) => …)         — model → client messages
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Ltx2Provider,
  useLtx2,
  useLtx2CommandError,
  useLtx2GenerationFailed,
  useLtx2GenerationStarted,
  useLtx2StateUpdate,
} from "@reactor-models/ltx2";
import { REACTOR_API_URL } from "@/app/lib/config";
import { reduce } from "@/app/lib/state";
import type { TransportCommand } from "@/app/lib/machine";
import {
  DEFAULT_UI_STATE,
  type Ltx2UiState,
  type TakeEdit,
} from "@/app/lib/types";
import type { Preset } from "@/app/lib/presets";
import { Header } from "./components/Header";
import { Notice, type NoticeValue } from "./components/Notice";
import { PresetRail } from "./components/PresetRail";
import { SnapClip } from "./components/SnapClip";
import { Stage } from "./components/Stage";
import { StatusBadge } from "./components/StatusBadge";
import { Transport } from "./components/Transport";
import { TakePanel } from "./components/TakePanel";

// JWT resolver passed to <Ltx2Provider jwtToken>. js-sdk 3.x accepts a static
// string or a resolver; the SDK calls a resolver on every Reactor API request,
// which is what keeps uploads, clip manifests and ICE refreshes from 401ing
// once the token ages out.
//
// The token is memoized here, in module scope, until shortly before it expires,
// and the fetch is no-store so the browser HTTP cache stays out of it. Holding
// the token in the app rather than in the browser cache is what makes its
// lifetime observable: a cache the app owns cannot be emptied out from under a
// live session by DevTools "Disable cache" or an eviction, and one mint then
// serves every hop of that session. (Examples that downscope their token with
// `authorization_details` need this for correctness, since a session may only
// be operated by the token that created it. This route mints an account-scoped
// token, so here it is one round trip instead of many.)
//
// Owning the cache also settles what the API URL used to be a query
// parameter for. Tokens are signed per-environment, and the parameter existed
// only to keep the browser from replaying one environment's JWT at another.
// This memo lives for one page load, so it cannot outlive a rebuild that
// changes the API endpoint.
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

export function Ltx2App() {
  return (
    <Ltx2Provider
      apiUrl={REACTOR_API_URL}
      jwtToken={fetchToken}
      // Deliberately NOT autoConnect. Connecting on a click does two jobs at
      // once: a session holds a whole B200, so opening the page should not
      // claim one, and the click is the user gesture the browser's autoplay
      // policy needs before <video> may play with sound. Auto-connecting
      // forced an "enable audio" overlay onto the first frame instead.
      //
      // maxAttempts widens the SDK's SDP-answer polling window (default 6
      // attempts ≈ 13s) so connect() waits as long as the Reactor API will
      // hold a pending connection (~60s) before giving up. A session
      // scheduled onto a pod that takes longer than that to come up still
      // fails; pressing Connect again opens a fresh window.
      connectOptions={{ autoConnect: false, maxAttempts: 30 }}
    >
      <Workspace />
    </Ltx2Provider>
  );
}

function Workspace() {
  const {
    status,
    lastError,
    uploadFile,
    setAvatarImage: sendAvatarImage,
    setScript,
    setPrompt,
    setWpm,
    setDurationSeconds,
    setSeed,
    start,
    pause,
    resume,
    stop,
    reset,
  } = useLtx2();

  const [ui, setUi] = useState<Ltx2UiState>(DEFAULT_UI_STATE);
  const [notice, setNotice] = useState<NoticeValue | null>(null);

  // The two holds on Start. Either one means a condition is in flight that the
  // model has not taken on yet, so a take started now would be generated from
  // the conditions it is about to replace. See setAvatarImage below.
  const [presetPending, setPresetPending] = useState<string | null>(null);
  const [imagePending, setImagePending] = useState(false);

  // ── Messages ────────────────────────────────────────────────────────────
  // All message handling lives here, not scattered across leaf components:
  // ordering between handlers stays obvious, and there is one place to look
  // when a transition misbehaves.

  // The snapshot is the only thing that mutates UI state.
  useLtx2StateUpdate((msg) => setUi((prev) => reduce(prev, msg)));

  // A ref mirror of the snapshot, for the async action callbacks. They need
  // to read the *current* state mid-sequence without taking `ui` as a
  // dependency, which would rebuild them on every window's state_update.
  const uiRef = useRef(ui);
  uiRef.current = ui;

  // A ref mirror of `lastError`, readable mid-await. The store field captured
  // in an async closure is a render-time snapshot; the ref follows re-renders,
  // so comparing it against a value taken before a call sees errors that
  // arrived during that call. See setAvatarImage.
  const lastErrorRef = useRef(lastError);
  lastErrorRef.current = lastError;

  // Command rejections are always surfaced. The model is the authority on what
  // is valid — machine.ts passes `valid_commands` straight through rather than
  // reproducing the rules — so a refusal landing here means the snapshot the UI
  // acted on was already stale by the time the command arrived. Rare, and worth
  // seeing rather than swallowing: it is the difference between "the model said
  // no" and "the button did nothing".
  useLtx2CommandError((msg) =>
    setNotice({ kind: "error", text: `${msg.command} refused: ${msg.reason}` }),
  );
  useLtx2GenerationFailed((msg) =>
    setNotice({ kind: "error", text: `Generation failed: ${msg.reason}` }),
  );

  // ── Time to first frame ─────────────────────────────────────────────────
  // t0 is the moment `start` goes on the wire; the clock stops at the first
  // newly-composited frame (Stage measures that with
  // requestVideoFrameCallback).
  //
  // The subtlety: the WebRTC track stays live between takes, so frames keep
  // being composited while the model is idle. Stopping the clock on the next
  // frame after `start` therefore measures nothing — it reads a few
  // milliseconds. So the measurement is *armed* by `generation_started`, and
  // only frames after that count.
  const startSentAt = useRef<number | null>(null);
  const ttffArmed = useRef(false);
  const [ttffMs, setTtffMs] = useState<number | null>(null);

  // ── What the stage is showing ───────────────────────────────────────────
  // Same subtlety, second consequence: because the track stays live, the last
  // frame of a take stays composited on the <video> forever. Nothing about
  // the model going idle takes it down, so `reset` appears to do nothing to
  // the stage. This flag is the client-side truth of "those pixels belong to
  // a take that is still current", and Stage covers the frame when it is
  // false.
  //
  // It cannot be derived from the snapshot. `finished` is the closest field
  // and it is not the same question: per its own description, changing a
  // condition clears `finished` too, so keying off it would blank a completed
  // take the moment the user edits the script for the next one.
  const [stageHasTake, setStageHasTake] = useState(false);

  useLtx2GenerationStarted(() => {
    ttffArmed.current = true;
    setStageHasTake(true);
  });

  // `reset` clears every condition server-side. The take panel holds local
  // drafts for fields the user is mid-edit on (so an incoming snapshot can't
  // clobber a half-typed value), and those drafts would survive the reset and
  // re-apply themselves on the next commit. Bumping a nonce that keys the
  // panel remounts it, dropping the drafts along with the state they mirrored.
  //
  // The stage and its TTFF reading go with them: both describe the take the
  // reset just discarded. Leaving them would keep that take's final frame on
  // screen, captioned with its latency, for a session that has been cleared.
  //
  // `generation_reset` is the correlated reply to `reset()`, delivered to the
  // calling connection rather than broadcast, so runTransport invokes this on
  // the resolved await. A `useLtx2GenerationReset` listener would subscribe
  // and never fire.
  const [resetNonce, setResetNonce] = useState(0);
  const handleGenerationReset = useCallback(() => {
    setResetNonce((n) => n + 1);
    setStageHasTake(false);
    setTtffMs(null);
    startSentAt.current = null;
    ttffArmed.current = false;
  }, []);

  const markStartSent = useCallback(() => {
    startSentAt.current = performance.now();
    ttffArmed.current = false;
    setTtffMs(null);
  }, []);

  const markFirstFrame = useCallback(() => {
    if (!ttffArmed.current || startSentAt.current === null) return;
    ttffArmed.current = false;
    setTtffMs(performance.now() - startSentAt.current);
  }, []);

  // Clear everything session-scoped when the session goes away. The SDK does
  // not emit a final `state_update` on disconnect, so without this a reconnect
  // renders stale conditions from the previous session.
  useEffect(() => {
    if (status !== "disconnected") return;
    setUi(DEFAULT_UI_STATE);
    setPresetPending(null);
    // An upload in flight when the session dies resolves `undefined` (the
    // SDK times its correlated reply out), but the `finally` that lowers this
    // only runs once that timeout elapses — so drop the hold now rather than
    // leaving Start dead into the next session.
    setImagePending(false);
    setStageHasTake(false);
    setTtffMs(null);
    startSentAt.current = null;
    ttffArmed.current = false;
  }, [status]);

  // ── Actions ─────────────────────────────────────────────────────────────

  /**
   * Fire an argument-free command, anchoring the TTFF clock when it is
   * `start`.
   *
   * The record is the only place a command is reachable by name, and its
   * `Record<TransportCommand, …>` annotation makes it exhaustive — adding a
   * command to the union without wiring it here is a type error. Components
   * ask for one by name; nothing outside this file builds a command string.
   *
   * The methods resolve with whatever their command declares as its reply, so
   * the record's value type is what they have in common rather than `void`:
   * `pause`, `resume` and `reset` answer with a message, `start` and `stop`
   * with nothing. Only `reset`'s reply is read — it is the signal that the
   * model has actually cleared, and it arrives here rather than at a listener.
   */
  const runTransport = useCallback(
    async (command: TransportCommand) => {
      const transport: Record<TransportCommand, () => Promise<unknown>> = {
        start,
        pause,
        resume,
        stop,
        reset,
      };
      if (command === "start") markStartSent();
      const reply = await transport[command]();
      if (command === "reset" && reply) handleGenerationReset();
    },
    [start, pause, resume, stop, reset, markStartSent, handleGenerationReset],
  );

  /**
   * Upload a portrait, anchor the avatar to it, and resolve only once the
   * model has CONFIRMED the new image.
   *
   * The confirmation is the awaited call itself. `set_avatar_image` declares
   * `avatar_image_accepted` as its reply, so the model's ack is delivered to
   * this connection correlated to this command, and `sendAvatarImage` resolves
   * with it once the handler has fetched and decoded the upload. Nothing has
   * to be matched against a message listener, which is what an earlier version
   * of this file did — and what a `start` racing a still-decoding image would
   * otherwise defeat, generating the take with the previous face.
   *
   * `undefined` means no reply came back, which is either a refusal or a send
   * that failed (the SDK times the round trip out rather than hanging). Both
   * land on `lastError`, which is a persistent record — success never clears
   * it — so only an error that appeared since the pre-call snapshot belongs to
   * this call. A refusal has already surfaced through the command_error
   * handler; anything else gets its own notice.
   *
   * Awaiting is only half of it — something has to hold `start` for the
   * duration. `imagePending` does that, and it is raised HERE rather than at
   * the call sites so every path is covered by construction: the crop modal
   * fires this and forgets it (`void onAvatarImage(…)`), and a gate each
   * caller has to remember to raise is a gate that eventually gets forgotten.
   * The `finally` is load-bearing for the same reason — a throw out of the
   * upload must not leave Start dead for the rest of the session.
   */
  const setAvatarImage = useCallback(
    async (file: File | Blob, name: string): Promise<boolean> => {
      setImagePending(true);
      try {
        const ref = await uploadFile(file, { name });
        const errorBefore = lastErrorRef.current;
        const accepted = await sendAvatarImage({ avatar_image: ref });
        if (accepted) return true;
        if (lastErrorRef.current === errorBefore) {
          // No new error, no reply: the model refused, and the
          // command_error handler has already said why.
          return false;
        }
        setNotice({
          kind: "error",
          text: "The model did not confirm the avatar image — try again before starting.",
        });
        return false;
      } finally {
        setImagePending(false);
      }
    },
    [uploadFile, sendAvatarImage],
  );

  /**
   * Run a preset: the real command sequence, in order, exactly as you would
   * send it by hand. Presets are macros over the same form the take panel
   * exposes — that is the whole point of them being here.
   *
   * `presetPending` holds the Start button for the duration, and `start` only
   * goes out after the model confirms the new avatar image, so a fast
   * preset-then-Start can never render a take with the previous face.
   */
  const directPreset = useCallback(
    async (preset: Preset) => {
      setPresetPending(preset.id);
      try {
        // Portraits are not committed (see public/presets/README.md). When one
        // is missing, apply everything except the image so the command
        // sequence is still visible and the user can upload their own face.
        const res = await fetch(preset.portrait).catch(() => null);
        const blob =
          res?.ok && res.headers.get("content-type")?.startsWith("image/")
            ? await res.blob()
            : null;

        let hasImage = uiRef.current.hasAvatarImage;
        if (blob) {
          const confirmed = await setAvatarImage(blob, `${preset.id}.jpg`);
          if (!confirmed) return; // never start on an unconfirmed image
          hasImage = true;
        }

        await setScript({ script: preset.script });
        await setPrompt({ prompt: preset.prompt });
        await setWpm({ wpm: preset.wpm });
        await setSeed({ seed: preset.seed });
        // Presets derive their length from the script, so clear any duration
        // pinned by a previous take.
        await setDurationSeconds({ duration_seconds: 0 });

        if (!hasImage) {
          setNotice({
            kind: "info",
            text: `No portrait for ${preset.name} — drop ${preset.id}.jpg into public/presets/, or upload a face, then press Start.`,
          });
          return;
        }
        markStartSent();
        await start();
      } catch (error) {
        setNotice({
          kind: "error",
          text: `Preset failed: ${error instanceof Error ? error.message : String(error)}`,
        });
      } finally {
        setPresetPending(null);
      }
    },
    [
      setAvatarImage,
      setScript,
      setPrompt,
      setWpm,
      setSeed,
      setDurationSeconds,
      start,
      markStartSent,
    ],
  );

  /**
   * Commit one take field. Always straight to the wire — there is no
   * client-side staging, because there is nothing to stage: the model accepts
   * every `set_*` during a run and queues it for the next take itself. What
   * is queued comes back on the snapshot as `queued_changes`, which is what
   * the panel renders its chips from.
   *
   * The switch is exhaustive over the {@link TakeEdit} union, so each branch
   * hands an already-narrowed value to its typed method.
   */
  const commitField = useCallback(
    async (edit: TakeEdit) => {
      switch (edit.field) {
        case "script":
          return setScript({ script: edit.value });
        case "prompt":
          return setPrompt({ prompt: edit.value });
        case "wpm":
          return setWpm({ wpm: edit.value });
        case "duration_seconds":
          return setDurationSeconds({ duration_seconds: edit.value });
        case "seed":
          return setSeed({ seed: edit.value });
      }
    },
    [setScript, setPrompt, setWpm, setDurationSeconds, setSeed],
  );

  // Nothing on the preset rail is valid while a take runs — a preset ends in
  // `start`, which the model refuses mid-run — so the rail is dimmed and made
  // inert rather than left looking interactive. The take panel deliberately
  // stays live: every condition on it is still accepted during a run, and
  // setting up the next take is the useful thing to do while watching this one.
  const inert = ui.generating
    ? "pointer-events-none select-none opacity-40 transition-opacity duration-300"
    : "transition-opacity duration-300";

  // Layout mirrors every other example in this repo: the stage owns the
  // <section>, and the <aside> owns *every* control, scrolling on its own.
  return (
    <div className="flex h-dvh flex-col bg-zinc-950">
      <Header />
      <main className="flex min-h-0 flex-1 flex-col lg:flex-row-reverse lg:gap-6 lg:p-6">
        <section className="flex flex-col gap-3 max-lg:sticky max-lg:top-0 max-lg:z-10 max-lg:bg-zinc-950/95 max-lg:p-4 max-lg:pb-3 max-lg:backdrop-blur-sm lg:min-h-0 lg:min-w-0 lg:flex-1 lg:overflow-hidden">
          <Notice value={notice} onDismiss={() => setNotice(null)} />
          <Stage
            status={status}
            ui={ui}
            hasTake={stageHasTake}
            onFirstFrame={markFirstFrame}
          />
        </section>

        <aside className="flex w-full flex-col gap-4 p-4 pt-1 lg:w-80 lg:shrink-0 lg:min-h-0 lg:overflow-y-auto lg:p-0">
          <StatusBadge status={status} />
          <Transport
            status={status}
            ui={ui}
            hasTake={stageHasTake}
            ttffMs={ttffMs}
            presetPending={presetPending}
            imagePending={imagePending}
            onCommand={runTransport}
          />
          <div className={inert} aria-hidden={ui.generating}>
            <PresetRail
              status={status}
              ui={ui}
              presetPending={presetPending}
              onRun={directPreset}
            />
          </div>
          {/* Deliberately NOT inert while generating: every condition here is
              still accepted during a run, and setting up the next take while
              watching this one is the clearest demonstration of the model's
              queue-for-next-take behaviour. */}
          <TakePanel
            key={resetNonce}
            status={status}
            ui={ui}
            imagePending={imagePending}
            onCommit={commitField}
            onAvatarImage={setAvatarImage}
            onNotice={setNotice}
          />
          {/* Recording is base-SDK surface, identical in every example, so
              this file is copied in unchanged and needs no model-specific
              code. It renders nothing until the session is ready. */}
          <SnapClip durationSeconds={30} label="Download this take" />
        </aside>
      </main>
    </div>
  );
}
