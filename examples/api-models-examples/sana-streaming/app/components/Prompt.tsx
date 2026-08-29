"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { useState } from "react";
import { Button, Panel, cn, EYEBROW, FOCUS_RING } from "./ui";
import { PROMPT_EXAMPLES } from "../lib/examples";

// Prompt draft + preset chips + the active-prompt readout. Prompts can be
// changed mid-stream at any time; the model applies them at the next chunk
// boundary (about one chunk later). The parent keys this component on the
// reset nonce, so a model generation_reset remounts it and clears the draft.
export function Prompt({ currentPrompt }: { currentPrompt: string | null }) {
  const { setPrompt, status } = useSanaStreaming();
  const [text, setText] = useState("");

  const ready = status === "ready";
  const apply = (prompt: string) => {
    if (!ready) return;
    setPrompt({ prompt }).catch(console.error);
  };
  const applyPreset = (prompt: string) => {
    setText(prompt);
    apply(prompt);
  };

  return (
    <Panel label="Prompt">
      <div className="mb-2 flex flex-wrap gap-1.5">
        {PROMPT_EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            disabled={!ready}
            onClick={() => applyPreset(ex.prompt)}
            className={cn(
              "rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40 disabled:pointer-events-none",
              FOCUS_RING,
            )}
          >
            {ex.label}
          </button>
        ))}
      </div>
      <textarea
        aria-label="Prompt"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Describe the edit. Changes apply live, about one chunk later."
        rows={3}
        className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-900/40 px-3 py-2 font-mono text-sm leading-relaxed text-zinc-200 outline-none transition placeholder:text-zinc-600 focus:border-brand/60"
      />
      <Button
        variant="primary"
        size="md"
        onClick={() => apply(text)}
        disabled={!text.trim() || !ready}
        className="mt-2 w-full"
      >
        Apply prompt
      </Button>
      {currentPrompt && (
        <p
          title={currentPrompt}
          className={cn(
            EYEBROW,
            "mt-2 line-clamp-2 break-words normal-case tracking-normal text-zinc-500",
          )}
        >
          active: “{currentPrompt}”
        </p>
      )}
    </Panel>
  );
}
