"use client";

import { useState } from "react";
import { useLongliveV2, useLongliveV2State } from "@reactor-models/longlive-v2";
import type { LongliveV2StateMessage } from "@reactor-models/longlive-v2";
import { type BeatKind } from "../lib/storyboard-store";
import { Panel, Button, SegmentedToggle, Icon, secs } from "./ui";

// LIVE-PHASE PANEL. Direct the running session: fire a soft shot or hard cut
// at the next chunk boundary ("now"), or schedule one ahead at a chunk index.
// Hidden until generation has started — composing the plan is <Storyboard>.
export function Director() {
  const { status, setShot, sceneCut, scheduleShot, scheduleSceneCut } =
    useLongliveV2();
  const [snapshot, setSnapshot] = useState<LongliveV2StateMessage | null>(null);
  const [text, setText] = useState("");
  const [kind, setKind] = useState<BeatKind>("shot");
  const [when, setWhen] = useState<"now" | "at">("now");
  const [atChunk, setAtChunk] = useState(0);

  useLongliveV2State((msg) => setSnapshot(msg));

  if (status !== "ready" || !snapshot?.started) return null;

  const sessionChunk = snapshot.session_chunk ?? 0;

  async function fire() {
    const prompt = text.trim();
    if (!prompt) return;
    if (when === "now") {
      if (kind === "cut") await sceneCut({ prompt });
      else await setShot({ prompt });
    } else {
      const target = Math.max(sessionChunk + 1, atChunk);
      if (kind === "cut")
        await scheduleSceneCut({ prompt, at_session_chunk: target });
      else await scheduleShot({ prompt, at_session_chunk: target });
    }
    setText("");
  }

  return (
    <Panel label="Direct live">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          kind === "cut"
            ? "Describe the scene to cut to…"
            : "Describe the next shot…"
        }
        className="min-h-[56px] w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 p-2 text-sm text-zinc-100 outline-none focus:border-zinc-500"
      />

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
        <SegmentedToggle
          aria-label="Timing"
          value={when}
          onChange={(v) => setWhen(v as "now" | "at")}
          options={[
            { value: "now", label: "Now" },
            { value: "at", label: "At" },
          ]}
        />
        {when === "at" && (
          <>
            <input
              type="number"
              min={sessionChunk + 1}
              value={atChunk}
              onChange={(e) => setAtChunk(Number(e.target.value))}
              className="w-14 rounded border border-zinc-700 bg-zinc-950 px-1 py-1 text-center font-mono tabular-nums text-zinc-100"
            />
            <span className="text-zinc-500">~{secs(atChunk)}</span>
          </>
        )}
      </div>

      <Button
        variant="primary"
        onClick={fire}
        disabled={!text.trim()}
        className="mt-2 w-full"
        leadingIcon={<Icon name={kind === "cut" ? "scissors" : "play"} />}
      >
        {when === "now"
          ? kind === "cut"
            ? "Cut now"
            : "Shot now"
          : `Schedule ${kind} @ ${atChunk}`}
      </Button>
    </Panel>
  );
}
