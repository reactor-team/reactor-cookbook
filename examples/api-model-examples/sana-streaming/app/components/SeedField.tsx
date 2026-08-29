"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { cn, EYEBROW } from "./ui";

// Noise seed — a pre-start setting. The model reads it when generation begins
// (the same source, prompt, and seed reproduce the same result), so it only
// appears in the Input panel's setup view; change it then reset to re-seed.
export function SeedField({ modelSeed }: { modelSeed: number }) {
  const { setSeed, status } = useSanaStreaming();
  const notReady = status !== "ready";

  return (
    <label className="flex items-center gap-1.5">
      <span className={EYEBROW}>Seed</span>
      {/* Uncontrolled, keyed to the model-reported seed: a reset or an
          external set_seed remounts the input with the fresh value. */}
      <input
        key={modelSeed}
        type="number"
        min={0}
        defaultValue={modelSeed}
        disabled={notReady}
        onBlur={(e) => setSeed({ seed: +e.target.value }).catch(console.error)}
        className={cn(
          "w-20 rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1 font-mono text-xs text-zinc-200 outline-none transition focus:border-brand/60 disabled:opacity-40",
        )}
      />
    </label>
  );
}
