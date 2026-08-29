"use client";

import { useState } from "react";
import { useLongliveV2State } from "@reactor-models/longlive-v2";
import type { LongliveV2StateMessage } from "@reactor-models/longlive-v2";
import {
  useStoryboard,
  sceneStarts,
  SCENE_BUDGET,
  CHUNK_SECONDS,
} from "../lib/storyboard-store";
import { EYEBROW, Panel, Icon } from "./ui";

// The director's timeline. A read-only visual of the storyboard on a chunk
// axis: scene dividers at each cut, beats as ticks, and — once running — a
// playhead at the model's cumulative `session_chunk`. The webapp playground
// makes this draggable; here it stays simple to keep the example readable.
export function Timeline() {
  const { beats } = useStoryboard();
  const [snapshot, setSnapshot] = useState<LongliveV2StateMessage | null>(null);
  useLongliveV2State((msg) => setSnapshot(msg));

  if (beats.length === 0) return null;

  const sessionChunk = snapshot?.session_chunk ?? 0;
  const running = !!snapshot?.running;
  const lastChunk = beats[beats.length - 1]?.atChunk ?? 0;
  const totalChunks = Math.max(SCENE_BUDGET, lastChunk + 8, sessionChunk + 4);
  const pct = (c: number) => `${Math.min(100, (c / totalChunks) * 100)}%`;
  const starts = sceneStarts(beats);

  return (
    <Panel>
      <div className="mb-2 flex items-center justify-between">
        <span className={EYEBROW}>Timeline</span>
        <span className="font-mono text-[11px] tabular-nums text-zinc-500">
          {totalChunks} chunks · ~{Math.round(totalChunks * CHUNK_SECONDS)}s
          {running && ` · chunk ${sessionChunk}`}
        </span>
      </div>

      <div className="relative h-12 overflow-hidden rounded-md border border-zinc-800 bg-zinc-950">
        {/* faint chunk grid: a hairline at every 12th chunk */}
        {Array.from(
          { length: Math.floor(totalChunks / 12) },
          (_, i) => (i + 1) * 12,
        ).map((c) => (
          <div
            key={`grid-${c}`}
            className="pointer-events-none absolute inset-y-0 w-px bg-zinc-800"
            style={{ left: pct(c) }}
          />
        ))}

        {/* subtle scene shading: alternating bands between cuts */}
        {starts.map((start, i) => {
          const nextStart = starts[i + 1] ?? totalChunks;
          return (
            <div
              key={`band-${start}`}
              className={`pointer-events-none absolute inset-y-0 ${
                i % 2 === 0 ? "bg-white/[0.02]" : ""
              }`}
              style={{
                left: pct(start),
                width: `calc(${pct(nextStart)} - ${pct(start)})`,
              }}
            />
          );
        })}

        {/* scene dividers at each cut */}
        {starts.map((c, i) =>
          i === 0 ? null : (
            <div
              key={`s-${c}`}
              className="pointer-events-none absolute inset-y-0 w-px bg-brand/50"
              style={{ left: pct(c) }}
            />
          ),
        )}

        {/* beats */}
        {beats.map((b) => (
          <div
            key={b.id}
            title={`${b.kind === "cut" ? "Cut" : "Shot"} @ chunk ${b.atChunk}: ${b.prompt}`}
            className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{ left: pct(b.atChunk) }}
          >
            <div className="flex size-4 items-center justify-center rounded">
              {b.atChunk === 0 ? (
                <Icon name="dot" className="size-3 text-zinc-300" />
              ) : b.kind === "cut" ? (
                <Icon name="scissors" className="size-3 text-brand" />
              ) : (
                <Icon name="dot" className="size-3 text-sky-300/80" />
              )}
            </div>
          </div>
        ))}

        {/* playhead */}
        {running && (
          <div
            className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-active"
            style={{ left: pct(sessionChunk), transition: "left 1.2s linear" }}
          >
            <div className="absolute -left-1 -top-1 h-2.5 w-2.5 rounded-full bg-active" />
          </div>
        )}
      </div>

      <p className="mt-1.5 text-[10px] text-zinc-500">
        <Icon
          name="dot"
          className="inline align-middle size-3 text-sky-300/80"
        />{" "}
        shot keeps the scene ·{" "}
        <Icon
          name="scissors"
          className="inline align-middle size-3 text-brand"
        />{" "}
        cut starts a new one (fresh {SCENE_BUDGET}-chunk budget)
      </p>
    </Panel>
  );
}
