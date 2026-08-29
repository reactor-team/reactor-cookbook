import type { Ltx2StateUpdateMessage } from "@reactor-models/ltx2";
import type { Ltx2UiState } from "./types";

// Projects model `state_update` snapshots into Ltx2UiState. Returns the
// previous object when nothing changed so React can bail out of re-rendering
// the whole tree on the model's frequent identical echoes — this model emits a
// snapshot after every window, so the bail-out matters.
//
// Only `state_update` feeds this reducer. The discrete acks
// (`script_accepted`, `avatar_image_accepted`, …) are the correlated replies
// to the commands that earned them — they resolve the awaited call rather than
// reaching any listener — and each is followed by a snapshot carrying the same
// information anyway, so reconstructing state from them would just be a
// second, racier path to the same place.
export function reduce(
  state: Ltx2UiState,
  msg: Ltx2StateUpdateMessage,
): Ltx2UiState {
  const next: Ltx2UiState = {
    script: msg.script ?? null,
    prompt: msg.prompt,
    hasAvatarImage: msg.has_avatar_image,
    wpm: msg.wpm,
    durationSeconds: msg.duration_seconds,
    effectiveSeconds: msg.effective_seconds,
    seed: msg.seed,
    ready: msg.ready,
    generating: msg.generating,
    validCommands: msg.valid_commands,
    queuedChanges: msg.queued_changes,
    wpmMin: msg.wpm_min,
    wpmMax: msg.wpm_max,
    paused: msg.paused,
    finished: msg.finished,
    windowIndex: msg.window_index,
    totalWindows: msg.total_windows,
    secondsSent: msg.seconds_sent,
  };
  // `valid_commands` and `queued_changes` arrive as fresh arrays in every
  // snapshot, so identity comparison would report a change on each one and
  // defeat the bail-out entirely. Compare those two by content.
  const changed = (Object.keys(next) as (keyof Ltx2UiState)[]).some((k) => {
    const a = next[k];
    const b = state[k];
    if (Array.isArray(a) && Array.isArray(b)) {
      return a.length !== b.length || a.some((v, i) => v !== b[i]);
    }
    return a !== b;
  });
  return changed ? next : state;
}
