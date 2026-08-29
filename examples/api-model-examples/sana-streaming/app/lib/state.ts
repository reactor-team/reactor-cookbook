import type { SanaStreamingStateMessage } from "@reactor-models/sana-streaming";
import type { SanaState } from "./types";

// Projects model `state` snapshots into SanaState. Returns the previous
// object when nothing changed so React can bail out of re-rendering the
// whole tree on the model's frequent identical echoes.
export function reduce(
  state: SanaState,
  msg: SanaStreamingStateMessage,
): SanaState {
  const next: SanaState = {
    running: msg.running,
    started: msg.started,
    paused: msg.paused,
    currentChunk: msg.current_chunk,
    // current_prompt is typed `unknown` on the wire (free-form); the model
    // only ever sends a string or null.
    currentPrompt: (msg.current_prompt as string | null) ?? null,
    seed: msg.seed,
  };
  const changed = (Object.keys(next) as (keyof SanaState)[]).some(
    (k) => next[k] !== state[k],
  );
  return changed ? next : state;
}

// Typed slice of useSanaStreaming() the start flow needs.
interface StartControls {
  start: () => Promise<undefined>;
}

// Both input sources stream into the `camera` track, and the model is live-only
// from v2.0.0 on — the file path and its `set_mode` / `set_video` commands are
// gone from the schema, so starting is just `start`.
//
// `start` declares no reply. Awaiting it is still a completion barrier: the
// runtime acknowledges a correlated command once its handler has run, so the
// resolved await means the model has started, not merely that the bytes left
// the browser.
export async function startGeneration(model: StartControls): Promise<void> {
  await model.start();
}
