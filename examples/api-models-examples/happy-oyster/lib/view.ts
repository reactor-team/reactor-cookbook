import type {
  HappyOysterPhase,
  WorldStateMessage,
} from "@reactor-models/happy-oyster";
import type { WorldIntent } from "./worlds";

// The app's one reducer: SDK snapshot in, view out.
//
// Everything on screen hinges on this function. The layout itself is fixed —
// header, control sidebar, content screen — and each region switches what it
// shows off the returned view. The inputs are the session's `phase` (the
// client lifecycle) and `worldState.phase` (the model's authoritative world
// lifecycle) plus the user's pending intent, so the UI can never disagree
// with the API: when the state machine moves, components turn on and off;
// nothing navigates away.

export type AppView =
  /** No pending intent: browse the gallery, compose, or attach. */
  | { kind: "browse" }
  /** Session opening (or reopening) for the pending intent. */
  | { kind: "connecting"; label: string }
  /** The model is creating or building the world. */
  | { kind: "building"; step: "creating" | "building"; restoring: boolean }
  /** World ready, no travel running (shown between travels). */
  | { kind: "ready" }
  /** A travel is starting or live; `live` means video is actually up. */
  | { kind: "traveling"; live: boolean }
  /** Something failed: an app-level error or a failed world build. */
  | { kind: "error"; message: string; buildFailed: boolean };

export function deriveView({
  intent,
  error,
  phase,
  worldState,
  streaming,
  starting,
  autoStartPending,
}: {
  intent: WorldIntent | null;
  error: string | null;
  phase: HappyOysterPhase;
  worldState: WorldStateMessage | null;
  streaming: boolean;
  starting: boolean;
  autoStartPending: boolean;
}): AppView {
  if (!intent) return { kind: "browse" };

  if (error) return { kind: "error", message: error, buildFailed: false };

  if (streaming || phase === "starting_stream" || starting || autoStartPending)
    return { kind: "traveling", live: streaming };

  if (phase !== "connected" || !worldState || worldState.phase === "no_world")
    return {
      kind: "connecting",
      label: phase === "connected" ? "Preparing world…" : "Connecting…",
    };

  if (worldState.phase === "failed")
    return {
      kind: "error",
      message:
        "The build didn't make it. Worlds are cheap — try the same prompt again or head back and pick another.",
      buildFailed: true,
    };

  if (worldState.phase === "creating" || worldState.phase === "building")
    return {
      kind: "building",
      step: worldState.phase,
      restoring: intent.kind === "attach",
    };

  return { kind: "ready" };
}

// ── the loading journey ──────────────────────────────────────────────────────
//
// The same reducer idea applied to loading: instead of one opaque spinner,
// derive the ordered steps of the API's own machine and mark where it is.
// Each step names the facade call that drives it and the state that
// completes it, so the loading pane doubles as a live trace of the protocol:
//
//   connect()      phase: connecting → connected
//   createWorld()  world_state.phase: creating → building   (attachWorld()
//                  jumps straight to ready when the world is already built)
//   (the build)    world_state.phase: building → ready, first_frame arrives
//   startTravel()  phase: starting_stream → streaming

export interface JourneyStep {
  key: "connect" | "request" | "generate" | "stream";
  /** Human label for the step. */
  label: string;
  status: "pending" | "active" | "done";
}

export function deriveJourney({
  intent,
  phase,
  worldState,
  streaming,
  starting,
  autoStartPending,
}: {
  intent: WorldIntent | null;
  phase: HappyOysterPhase;
  worldState: WorldStateMessage | null;
  streaming: boolean;
  starting: boolean;
  autoStartPending: boolean;
}): JourneyStep[] {
  const attach = intent?.kind === "attach";
  const worldPhase = worldState?.phase ?? "no_world";
  const connected =
    phase === "connected" ||
    phase === "starting_stream" ||
    phase === "streaming";
  const worldRequested =
    worldPhase === "creating" ||
    worldPhase === "building" ||
    worldPhase === "ready" ||
    worldPhase === "traveling";
  const worldBuilt = worldPhase === "ready" || worldPhase === "traveling";
  const streamOpening =
    phase === "starting_stream" || starting || autoStartPending;

  const status = (done: boolean, active: boolean): JourneyStep["status"] =>
    done ? "done" : active ? "active" : "pending";

  return [
    {
      key: "connect",
      label: "Connect the session",
      status: status(connected, !connected),
    },
    {
      key: "request",
      label: attach ? "Attach the world" : "Request the build",
      status: status(worldRequested, connected && !worldRequested),
    },
    {
      key: "generate",
      label: "Generate the world",
      status: status(worldBuilt, worldPhase === "building"),
    },
    {
      key: "stream",
      label: "Open the stream",
      status: status(streaming, streamOpening),
    },
  ];
}
