"use client";

import { useEffect, useState } from "react";
import {
  useHelios,
  useHeliosState,
  type HeliosStateMessage,
} from "@reactor-models/helios";
import { TEXT_SCENES } from "../lib/prompts";

// Setup-phase panel. Lets the user pick a preset or write their own
// prompt and kicks off generation with `set_prompt` → `start`.
//
// Renders null once generation has started — the user-facing surface
// switches to <NowPlaying> from there. Mid-stream prompt switching
// (a Helios capability) is intentionally NOT exposed in this tutorial
// to keep the lesson tight.
export function PromptComposer() {
  const { status, setPrompt, start } = useHelios();
  const [text, setText] = useState("");
  const [snapshot, setSnapshot] = useState<HeliosStateMessage | null>(null);

  useHeliosState((msg) => setSnapshot(msg));

  useEffect(() => {
    if (status !== "ready") setSnapshot(null);
  }, [status]);

  // Hide once we're generating — but keep rendering (in disabled form)
  // when the user is just not connected, so the page doesn't go blank
  // after disconnect.
  if (status === "ready" && snapshot?.started) return null;

  const ready = status === "ready";

  async function send(prompt: string) {
    if (!ready || !prompt.trim()) return;
    await setPrompt({ prompt: prompt.trim() });
    await start();
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        Try a prompt
      </label>

      <div className="mt-2 grid grid-cols-2 gap-1.5">
        {TEXT_SCENES.map((scene) => (
          <button
            key={scene.id}
            disabled={!ready}
            onClick={() => send(scene.initial.text)}
            className="group rounded-md border border-zinc-800 bg-zinc-950 p-2 text-left transition-colors hover:border-brand disabled:opacity-40 disabled:hover:border-zinc-800"
            title={scene.initial.text}
          >
            <div className="text-[11px] font-medium text-zinc-200 group-hover:text-brand">
              {scene.label}
            </div>
            <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-zinc-500">
              {scene.initial.text}
            </p>
          </button>
        ))}
      </div>

      <label className="mt-4 block text-[10px] uppercase tracking-wider text-zinc-500">
        Or write your own
      </label>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Describe the scene you want to generate…"
        disabled={!ready}
        rows={3}
        className="mt-2 w-full resize-none rounded-md border border-zinc-800 bg-zinc-950 p-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-brand focus:outline-none disabled:opacity-40"
      />

      <button
        disabled={!ready || !text.trim()}
        onClick={async () => {
          await send(text);
          setText("");
        }}
        className="mt-2 w-full rounded-md bg-brand px-3 py-1.5 text-sm font-medium text-brand-fg hover:opacity-90 disabled:opacity-40"
      >
        Start generating
      </button>
    </div>
  );
}
