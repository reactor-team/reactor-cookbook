"use client";

import { useState } from "react";
import { useX2 } from "@reactor-models/x2";
import { PROMPT_EXAMPLES } from "@/app/lib/examples";
import { Button, Panel, cn, EYEBROW, FOCUS_RING } from "./ui";

// Prompt draft + preset chips + the active-prompt readout. Setting a prompt
// is what arms generation: X2 starts on its own once a non-empty prompt is
// set and source frames are arriving. Mid-stream re-prompts apply from the
// next generated block. The parent keys this component on the reset nonce,
// so a model reset (generation_stopped) remounts it and clears the draft.
export function Prompt({ activePrompt }: { activePrompt: string | null }) {
  const { setPrompt, status } = useX2();
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
            title={ex.prompt}
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
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            if (text.trim()) apply(text);
          }
        }}
        placeholder="Describe the edit. Applying a prompt starts (or re-steers) generation."
        rows={3}
        maxLength={1000}
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
      {activePrompt && (
        <p
          title={activePrompt}
          className={cn(
            EYEBROW,
            "mt-2 line-clamp-2 break-words normal-case tracking-normal text-zinc-500",
          )}
        >
          active: “{activePrompt}”
        </p>
      )}
    </Panel>
  );
}
