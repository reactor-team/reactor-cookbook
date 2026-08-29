"use client";

import { useState } from "react";
import { useLongliveV2, useLongliveV2State } from "@reactor-models/longlive-v2";
import type { LongliveV2StateMessage } from "@reactor-models/longlive-v2";
import { useStoryboard, type BeatKind } from "../lib/storyboard-store";
import { STORYBOARDS } from "../lib/prompts";
import {
  cn,
  secs,
  FOCUS_RING,
  Panel,
  Button,
  SegmentedToggle,
  Icon,
} from "./ui";

// SETUP-PHASE PANEL. Compose a storyboard before pressing start: an opening
// shot, then shots (soft) and cuts (hard) scheduled at chunk positions.
// "Start storyboard" compiles the plan into the wire sequence:
//   set_shot(opener) → schedule_shot / schedule_scene_cut(...) → start
//
// Hidden once generation is running — directing live moves to <Director>.
export function Storyboard() {
  const { status, setShot, scheduleShot, scheduleSceneCut, start } =
    useLongliveV2();
  const { beats, setOpening, addBeat, remove, load, clear } = useStoryboard();
  const [snapshot, setSnapshot] = useState<LongliveV2StateMessage | null>(null);
  const [text, setText] = useState("");
  const [kind, setKind] = useState<BeatKind>("shot");
  const [atChunk, setAtChunk] = useState(20);
  const [starting, setStarting] = useState(false);

  useLongliveV2State((msg) => setSnapshot(msg));

  // Live phase belongs to <Director> / <NowPlaying>.
  if (status === "ready" && snapshot?.started) return null;

  const opener = beats.find((b) => b.atChunk === 0);
  const ready = status === "ready";

  function add() {
    const prompt = text.trim();
    if (!prompt) return;
    if (!opener) setOpening(prompt);
    else addBeat(kind, prompt, atChunk);
    setText("");
  }

  async function startStoryboard() {
    if (!ready || !opener || starting) return;
    setStarting(true);
    try {
      await setShot({ prompt: opener.prompt });
      for (const b of beats.filter((b) => b.atChunk !== 0)) {
        if (b.kind === "cut") {
          await scheduleSceneCut({
            prompt: b.prompt,
            at_session_chunk: b.atChunk,
          });
        } else {
          await scheduleShot({ prompt: b.prompt, at_session_chunk: b.atChunk });
        }
      }
      await start();
    } finally {
      setStarting(false);
    }
  }

  return (
    <Panel
      label="Storyboard"
      action={
        beats.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={clear}>
            clear
          </Button>
        ) : undefined
      }
    >
      {/* Presets */}
      <div className="mb-3 flex flex-col gap-1.5">
        {STORYBOARDS.map((s) => (
          <button
            key={s.id}
            onClick={() => load(s.beats.map((b) => ({ ...b })))}
            className={cn(
              "w-full rounded-md border border-zinc-700 px-2.5 py-2 text-left transition-colors hover:bg-zinc-800",
              FOCUS_RING,
            )}
          >
            <span className="block text-sm font-semibold text-zinc-200">
              {s.label}
            </span>
            <span className="block text-[11px] text-zinc-500">
              {s.description}
            </span>
          </button>
        ))}
      </div>

      {/* Composer */}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          opener
            ? "Describe a shot or the next scene…"
            : "Describe the opening shot…"
        }
        className="min-h-[64px] w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 p-2 text-sm text-zinc-100 outline-none focus:border-zinc-500"
      />

      {opener && (
        <div className="mt-2 flex items-center gap-1.5 text-xs">
          <SegmentedToggle
            aria-label="Beat kind"
            value={kind}
            onChange={(v) => setKind(v as BeatKind)}
            options={[
              { value: "shot", label: "Shot", hint: " · soft" },
              { value: "cut", label: "Cut", hint: " · hard" },
            ]}
          />
          <span className="text-zinc-500">at chunk</span>
          <input
            type="number"
            min={1}
            value={atChunk}
            onChange={(e) => setAtChunk(Math.max(1, Number(e.target.value)))}
            className="w-14 rounded border border-zinc-700 bg-zinc-950 px-1 py-1 text-center font-mono tabular-nums text-zinc-100"
          />
          <span className="text-zinc-500">~{secs(atChunk)}</span>
        </div>
      )}

      <Button
        variant="secondary"
        size="sm"
        onClick={add}
        disabled={!text.trim()}
        className="mt-2 w-full"
      >
        {opener ? `+ Add ${kind}` : "+ Set opening shot"}
      </Button>

      {/* Beat list */}
      {beats.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1">
          {beats.map((b) => (
            <li
              key={b.id}
              className="flex items-center gap-2 rounded bg-zinc-950/60 px-2 py-1 text-xs"
            >
              {b.atChunk === 0 ? (
                <span className="font-mono text-[10px] uppercase text-zinc-400">
                  open
                </span>
              ) : b.kind === "cut" ? (
                <span className="flex items-center gap-1">
                  <Icon name="scissors" className="size-3 text-brand" />
                  <span className="font-mono tabular-nums text-zinc-500">
                    @{b.atChunk}
                  </span>
                </span>
              ) : (
                <span className="flex items-center gap-1">
                  <Icon name="dot" className="size-3 text-sky-300/80" />
                  <span className="font-mono tabular-nums text-zinc-500">
                    @{b.atChunk}
                  </span>
                </span>
              )}
              <span className="flex-1 truncate text-zinc-300">{b.prompt}</span>
              <button
                onClick={() => remove(b.id)}
                className={cn("text-zinc-500 hover:text-red-400", FOCUS_RING)}
                aria-label="Remove beat"
              >
                <Icon name="x" className="size-3" />
              </button>
            </li>
          ))}
        </ul>
      )}

      <Button
        variant="primary"
        onClick={startStoryboard}
        disabled={!ready || !opener || starting}
        className="mt-3 w-full"
        leadingIcon={<Icon name="play" />}
      >
        {starting ? "Starting…" : "Start storyboard"}
      </Button>
      {!ready && (
        <p className="mt-1 text-[11px] text-zinc-500">
          Connect first to start.
        </p>
      )}
    </Panel>
  );
}
