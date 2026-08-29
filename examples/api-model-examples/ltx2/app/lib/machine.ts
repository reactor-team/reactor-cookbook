import type { ReactorStatus } from "@reactor-team/js-sdk";
import type { Ltx2UiState } from "./types";

/** Every command the model exposes. */
export type Command =
  | "set_avatar_image"
  | "set_script"
  | "set_prompt"
  | "set_wpm"
  | "set_duration_seconds"
  | "set_seed"
  | "start"
  | "pause"
  | "resume"
  | "stop"
  | "reset";

/**
 * The commands that take no arguments — everything the transport fires.
 *
 * Kept separate from {@link Command} so the app shell can hold one exhaustive
 * record of them mapped to typed methods, and a component can ask for one by
 * name without any component ever naming a `set_*` command as a string.
 */
export type TransportCommand = "start" | "pause" | "resume" | "stop" | "reset";

/**
 * Which commands the session would accept right now.
 *
 * The model answers this itself, in `state_update.valid_commands`, and it is
 * authoritative: anything absent from the list comes back as `command_error`.
 * So this function does not reproduce the state machine, it just adds the one
 * thing the snapshot cannot know — whether there is a session at all.
 *
 * Keep asking through here rather than reading `ui.validCommands` directly.
 * It stays the single place a component's question about command validity is
 * answered, and the connection gate below is easy to forget at a call site.
 *
 * Worth knowing when reading the UI: every `set_*` is valid *during* a run,
 * not only when idle. The model accepts the change and applies it to the next
 * take, listing the field in `queued_changes`. Only `start` is gated on the
 * run ending.
 */
export function validCommands(
  status: ReactorStatus,
  ui: Ltx2UiState,
): Set<Command> {
  // Commands sent before the session is ready are rejected by the SDK, and a
  // stale snapshot from a previous session must not make buttons look live.
  if (status !== "ready") return new Set();
  return new Set(ui.validCommands as Command[]);
}

/**
 * Whether a take field is queued for the next take rather than already live.
 *
 * The model owns this: it lists the fields changed during the run in flight
 * in `state_update.queued_changes`. The UI never tracks pending edits itself.
 */
export function isQueued(ui: Ltx2UiState, wireName: string): boolean {
  return ui.queuedChanges.includes(wireName);
}

/**
 * The run has started but has not streamed anything yet.
 *
 * Deliberately affects ONLY the status text. An earlier version of this also
 * decided whether the stage covered the video, which turned any wrong answer
 * here into a black screen over a take that was streaming fine — the audio
 * kept playing underneath it. Visibility is now driven by `hasTake` alone;
 * the worst this can do is mislabel.
 *
 * Two independent signals, because `window_index` alone has not been
 * confirmed to advance on the current dev deployment: if seconds are on the
 * wire, the run is past warm-up whatever the window counter says.
 */
function isWarming(ui: Ltx2UiState): boolean {
  return ui.generating && ui.windowIndex < 0 && ui.secondsSent === 0;
}

/**
 * Why `start` is unavailable on an idle session, or null when it is available.
 *
 * `ready` is a single flag from the model, so a dead Start button otherwise
 * gives no clue which of the two conditions it is waiting on — and the two
 * text fields are easy to mistake for each other, since the scene prompt
 * ships pre-filled and so looks like the one that has been answered.
 */
export function startBlockedReason(ui: Ltx2UiState): string | null {
  if (ui.ready) return null;
  const hasScript = (ui.script ?? "").trim().length > 0;
  if (!ui.hasAvatarImage && !hasScript)
    return "Needs an avatar image and a script.";
  if (!ui.hasAvatarImage) return "Needs an avatar image.";
  return "Needs a script — the words spoken. The scene prompt describes the shot.";
}

/** Human status line, derived from connection + snapshot. */
export function statusLine(status: ReactorStatus, ui: Ltx2UiState): string {
  const warming = isWarming(ui);
  if (status === "disconnected") return "not connected — press Connect";
  if (status === "connecting") return "connecting to Reactor";
  if (status === "waiting") return "waiting for a GPU";
  // The leading window has to denoise and decode before anything can stream,
  // so there is a real gap between `generation_started` and the first frame.
  if (ui.generating && warming) return "warming up — first window denoising";
  if (ui.generating && ui.paused) return "paused";
  if (ui.generating) {
    const w = ui.windowIndex + 1;
    return `generating — window ${w}/${ui.totalWindows} · ${ui.secondsSent.toFixed(1)}s streamed`;
  }
  if (ui.finished) return "take complete";
  if (!ui.ready) return "set an avatar image and a script, or pick a preset";
  return "ready — press start";
}
