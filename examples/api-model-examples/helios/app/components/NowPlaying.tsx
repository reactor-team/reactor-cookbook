"use client";

import { useEffect, useState } from "react";
import {
  useHelios,
  useHeliosState,
  type HeliosStateMessage,
} from "@reactor-models/helios";

// Live-phase panel. Renders only while the model is generating.
//
// Helios emits a `state` snapshot after every command and every
// completed chunk. We hold the latest snapshot in useState and read
// fields off it — single source of truth for the running scene,
// no event aggregation needed.
//
// When `snapshot.started === false` (never started, or just reset),
// this component renders null and the setup panel (PromptComposer +
// ImageStarter) takes over. That's the phase switch.
export function NowPlaying() {
  const { status, pause, resume, reset } = useHelios();
  const [snapshot, setSnapshot] = useState<HeliosStateMessage | null>(null);

  useHeliosState((msg) => setSnapshot(msg));

  // Clear on disconnect. The SDK doesn't emit a final `state` message
  // when the session ends, so without this we'd keep showing the
  // previous session's state across a reconnect.
  useEffect(() => {
    if (status !== "ready") setSnapshot(null);
  }, [status]);

  if (status !== "ready" || !snapshot?.started) return null;

  const currentPrompt =
    typeof snapshot.current_prompt === "string" ? snapshot.current_prompt : "";
  const runState = snapshot.running
    ? "running"
    : snapshot.paused
      ? "paused"
      : "idle";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <span className="text-[10px] uppercase tracking-wider text-zinc-500">
        Now playing · {runState}
      </span>

      <p className="mt-2 line-clamp-3 text-sm leading-snug text-zinc-200">
        {currentPrompt || "(no prompt yet)"}
      </p>

      <div className="mt-3 flex gap-3 text-[11px] text-zinc-500">
        <span>chunk {snapshot.current_chunk}</span>
        <span>·</span>
        <span>{snapshot.current_frame} frames</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        {snapshot.running ? (
          <button
            onClick={() => pause()}
            className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm font-medium text-zinc-200 hover:bg-zinc-800"
          >
            Pause
          </button>
        ) : (
          <button
            onClick={() => resume()}
            className="rounded-md bg-brand px-3 py-2 text-sm font-medium text-brand-fg hover:opacity-90"
          >
            Resume
          </button>
        )}
        <button
          onClick={() => reset()}
          className="rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm font-medium text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
