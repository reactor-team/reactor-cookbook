"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { IconButton } from "./ui";

// Live-phase playback controls. Once generation has started, these replace the
// setup controls in the Input panel: pause/resume the running edit, or reset
// back to the setup state. `paused` decides between the pause and resume
// affordance; it comes from the model `state` snapshot, not local guesses.
//
// `generation_reset` is the correlated reply to `reset()` — it reaches the
// connection that asked, not a broadcast listener — so the shell's cleanup
// runs from `onReset` here, on the resolved await. A falsy reply means the
// send failed, and `command_error` carries the reason.
export function Playback({
  paused,
  onReset,
}: {
  paused: boolean;
  onReset: () => void;
}) {
  const { pause, resume, reset, status } = useSanaStreaming();
  const notReady = status !== "ready";

  return (
    <div className="flex items-center gap-2">
      {paused ? (
        <IconButton
          icon="play"
          label="Resume"
          disabled={notReady}
          onClick={() => resume()}
        />
      ) : (
        <IconButton
          icon="pause"
          label="Pause"
          disabled={notReady}
          onClick={() => pause()}
        />
      )}
      <IconButton
        icon="reset"
        label="Reset"
        tone="danger"
        disabled={notReady}
        onClick={() =>
          void reset().then((reply) => {
            if (reply) onReset();
          })
        }
      />
      <span className="ml-1 text-xs text-zinc-500">
        {paused ? "Paused" : "Editing — reset to change the input"}
      </span>
    </div>
  );
}
