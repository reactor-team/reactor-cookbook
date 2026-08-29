"use client";

import { useEffect, useRef, useState } from "react";
import {
  useLingbot,
  useLingbotState,
  type LingbotStateMessage,
} from "@reactor-models/lingbot";
import { DYNAMIC_EVENTS } from "../lib/dynamic-events";

// Live-phase panel — lets the user hot-swap the world by appending a
// preset environmental sentence to the active prompt and re-sending
// via `set_prompt`. Lingbot picks up the new prompt on the next chunk
// and the scene visibly shifts (rain begins, fog rolls in, etc.) —
// no restart, no flash, the reference image stays untouched. This
// is Lingbot's signature mid-stream prompt-swap capability put on a
// surface a non-author can press.
//
// State model (kept deliberately small):
//
//   - basePromptRef holds the "scene base" — the prompt the user
//     started the session with (or selected via the live custom-
//     prompt path, if you add one). We capture it the first time we
//     see a `started === true` snapshot and never overwrite it while
//     the session runs, because the snapshot's `current_prompt` will
//     reflect OUR composed prompts after the first event lands.
//
//   - activeId tracks which event is currently appended. Re-clicking
//     the same event toggles it off (back to the base scene). Picking
//     a different event swaps which sentence is appended. There is no
//     stacking — each press fully determines the next prompt the
//     model sees, which keeps the wire output unambiguous.
//
// On `reset()` (snapshot.started flips back to false), or on
// disconnect, we drop the captured base so the next session starts
// fresh.
export function DynamicEvents() {
  const { status, setPrompt } = useLingbot();
  const [snapshot, setSnapshot] = useState<LingbotStateMessage | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const basePromptRef = useRef<string | null>(null);

  useLingbotState((msg) => setSnapshot(msg));

  // Standard snapshot-clear on disconnect. Also drops the captured
  // base prompt so a reconnect doesn't reuse stale state from the
  // previous session.
  useEffect(() => {
    if (status !== "ready") {
      setSnapshot(null);
      basePromptRef.current = null;
      setActiveId(null);
    }
  }, [status]);

  // Capture the base prompt on first "started" snapshot. We
  // deliberately do NOT update `basePromptRef.current` again while
  // the session is running — once the user clicks an event, the
  // snapshot's `current_prompt` will be OUR composed prompt, and
  // re-capturing would lock in the augmented version as the new
  // "base", making toggle-off impossible.
  useEffect(() => {
    if (!snapshot) return;
    if (!snapshot.started) {
      // Reset / not-yet-started — drop captured base so the next
      // `start` re-captures from the new scene.
      basePromptRef.current = null;
      setActiveId(null);
      return;
    }
    if (
      basePromptRef.current === null &&
      typeof snapshot.current_prompt === "string"
    ) {
      basePromptRef.current = snapshot.current_prompt;
    }
  }, [snapshot]);

  if (status !== "ready" || !snapshot?.started) return null;

  async function apply(id: string) {
    const base = basePromptRef.current;
    if (!base) return;

    if (activeId === id) {
      // Toggle off — back to the pristine scene.
      setActiveId(null);
      await setPrompt({ prompt: base });
      return;
    }

    const event = DYNAMIC_EVENTS.find((e) => e.id === id);
    if (!event) return;
    setActiveId(id);
    await setPrompt({ prompt: `${base} ${event.text}` });
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        World events
      </label>

      <p className="mt-1 text-[11px] leading-snug text-zinc-500">
        Hot-swap the world. Each click sends a fresh prompt — the model picks it
        up on the next chunk. Re-click to revert.
      </p>

      <div className="mt-2 grid grid-cols-2 gap-1.5">
        {DYNAMIC_EVENTS.map((event) => {
          const active = activeId === event.id;
          return (
            <button
              key={event.id}
              onClick={() => apply(event.id)}
              className={`group flex items-center gap-2 rounded-md border p-2 text-left transition-colors ${
                active
                  ? "border-brand bg-zinc-900"
                  : "border-zinc-800 bg-zinc-950 hover:border-brand"
              }`}
              title={event.text}
            >
              <span aria-hidden className="text-base leading-none">
                {event.icon}
              </span>
              <span
                className={`text-[11px] font-medium ${
                  active ? "text-brand" : "text-zinc-200 group-hover:text-brand"
                }`}
              >
                {event.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
