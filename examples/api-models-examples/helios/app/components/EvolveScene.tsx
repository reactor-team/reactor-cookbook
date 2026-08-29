"use client";

import { useEffect, useState } from "react";
import {
  useHelios,
  useHeliosState,
  type HeliosStateMessage,
} from "@reactor-models/helios";
import { findSceneForPrompt } from "../lib/prompts";

// Live-phase panel that lets the user "evolve the scene" by clicking
// a curated continuation prompt.
//
// We look up which scene the session belongs to by matching the
// active `current_prompt` against our curated library (see
// `lib/prompts.ts`). If the prompt matches a known scene's `initial`
// or any of its `evolutions`, we render that scene's evolution list
// as one-click hot-swaps. If the user typed a custom prompt we don't
// recognise, this component renders nothing — they can still drive
// the model via Pause / Reset in NowPlaying.
//
// Clicking an evolution calls `setPrompt({ prompt })`. There's no
// `start()` here: we're already generating, so this is a mid-stream
// prompt swap. Helios picks up the new prompt on the next 33-frame
// chunk, no restart, the scene continues from where it left off.
export function EvolveScene() {
  const { status, setPrompt } = useHelios();
  const [snapshot, setSnapshot] = useState<HeliosStateMessage | null>(null);

  useHeliosState((msg) => setSnapshot(msg));

  useEffect(() => {
    if (status !== "ready") setSnapshot(null);
  }, [status]);

  if (status !== "ready" || !snapshot?.started) return null;

  const currentPrompt =
    typeof snapshot.current_prompt === "string" ? snapshot.current_prompt : "";
  const scene = findSceneForPrompt(currentPrompt);
  if (!scene) return null;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        Evolve the scene
      </label>

      <div className="mt-2 grid grid-cols-2 gap-1.5">
        {scene.evolutions.map((evolution) => {
          const active = currentPrompt === evolution.text;
          return (
            <button
              key={evolution.title}
              onClick={() => setPrompt({ prompt: evolution.text })}
              disabled={active}
              className={`group rounded-md border p-2 text-left transition-colors ${
                active
                  ? "border-brand bg-zinc-900"
                  : "border-zinc-800 bg-zinc-950 hover:border-brand"
              }`}
              title={evolution.text}
            >
              <div
                className={`text-[11px] font-medium ${
                  active ? "text-brand" : "text-zinc-200 group-hover:text-brand"
                }`}
              >
                {evolution.title}
              </div>
              <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-zinc-500">
                {evolution.text}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
