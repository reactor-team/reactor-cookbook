"use client";

import { useState } from "react";
import { GLASS_PANEL } from "../../lib/ui/glass";
import type { CompositionResult } from "../../lib/world/types";

/**
 * Debug/tuning panel: "here's the score we composed for this song".
 * Shows the Nemotron event series, the Stage 1 MusicalSummary it was
 * composed from (collapsible raw JSON — inspectable independently of
 * the LLM), and the per-song token counts. During playback the active
 * event is highlighted; the panel never edits anything — the score is
 * immutable once composed.
 */
export function ScorePanel({
  composition,
  activeIndex,
}: {
  composition: CompositionResult;
  activeIndex: number;
}) {
  const [showSummary, setShowSummary] = useState(false);
  const { summary, tokenUsage, seeds } = composition;
  const seedById = new Map(seeds.map((s) => [s.id, s]));

  return (
    <div className={`w-full p-4 text-left ${GLASS_PANEL}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase tracking-wider text-zinc-400">
          Composed world score
        </span>
        <span className="rounded-full border border-active/20 bg-active/10 px-2 py-0.5 text-[10px] font-medium text-active">
          {composition.model}
        </span>
      </div>

      {composition.interpretation && (
        <p className="mt-2 text-sm italic text-zinc-300">
          “{composition.interpretation}”
        </p>
      )}

      <p className="mt-2 font-mono text-[11px] text-zinc-500">
        {summary.bpm !== null ? `${Math.round(summary.bpm)} BPM` : "no beat grid"}
        {summary.lowConfidenceRhythm && " (low-confidence rhythm)"}
        {summary.downbeatPhaseUncertain && " (downbeat phase uncertain)"} ·{" "}
        {summary.keyEstimate}
        {summary.keyMode !== "unknown" ? ` (${summary.keyMode})` : ""} ·{" "}
        {summary.brightnessLabel} · {summary.harmonicPercussiveBalance} ·{" "}
        {summary.spectralMotion} · {summary.spectralWidth}
        {summary.harmonicCharacter ? ` · ${summary.harmonicCharacter}` : ""} ·{" "}
        {summary.sections.length} sections (
        {summary.labelRegime}
        {summary.sectionsDerivedFromEnergy && ", energy-derived"}) ·{" "}
        {summary.integratedLufs !== null
          ? `${summary.integratedLufs} LUFS`
          : "no loudness reference"}{" "}
        · {summary.durationSec.toFixed(1)}s
      </p>

      {tokenUsage && (
        <p className="mt-1 font-mono text-[11px] text-zinc-600">
          tokens: {tokenUsage.promptTokens} prompt +{" "}
          {tokenUsage.completionTokens} completion ={" "}
          {tokenUsage.totalTokens} total
        </p>
      )}

      <button
        onClick={() => setShowSummary((s) => !s)}
        className="mt-2 text-[11px] text-zinc-500 underline-offset-2 hover:text-zinc-300 hover:underline"
      >
        {showSummary ? "hide" : "show"} Stage 1 MusicalSummary (what the LLM saw)
      </button>
      {showSummary && (
        <pre className="mt-2 max-h-64 overflow-auto rounded-xl border border-white/10 bg-black/30 p-3 text-[10px] leading-relaxed text-zinc-400 backdrop-blur-xl">
          {JSON.stringify(summary, null, 2)}
        </pre>
      )}

      <ol className="mt-3 max-h-80 space-y-2 overflow-auto">
        {composition.events.map((event, i) => (
          <li
            key={`${event.timestamp}-${i}`}
            className={`rounded-xl border p-2.5 backdrop-blur-xl transition-colors ${
              i === activeIndex
                ? "border-brand/50 bg-brand/10"
                : "border-white/10 bg-black/25"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
              <span className={i === activeIndex ? "text-brand" : "text-zinc-400"}>
                {event.timestamp.toFixed(1)}s
              </span>
              <span
                className={`rounded px-1.5 py-px text-[10px] uppercase ${
                  event.transition === "cut"
                    ? "bg-red-500/15 text-red-400"
                    : "bg-sky-500/15 text-sky-400"
                }`}
              >
                {event.transition}
              </span>
              {/* Seed image chip — thumbnail + id, titled with its one-liner. */}
              <span
                title={seedById.get(event.seedId)?.oneLiner}
                className="flex items-center gap-1 rounded bg-violet-500/15 px-1.5 py-px text-[10px] text-violet-300"
              >
                {seedById.get(event.seedId) && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={seedById.get(event.seedId)!.imageUrl}
                    alt=""
                    className="h-3.5 w-3.5 rounded-sm object-cover"
                  />
                )}
                {event.seedId}
              </span>
            </div>
            <p className="mt-1 line-clamp-2 text-xs leading-snug text-zinc-400">
              {event.prompt}
            </p>
          </li>
        ))}
      </ol>
    </div>
  );
}
