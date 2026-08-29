"use client";

import { useEffect, useState } from "react";
import { useLongliveV2, useLongliveV2State } from "@reactor-models/longlive-v2";
import type { LongliveV2StateMessage } from "@reactor-models/longlive-v2";
import { useStoryboard, SCENE_BUDGET } from "../lib/storyboard-store";
import { cn, Panel, Button, IconButton, Icon } from "./ui";

// LIVE-PHASE PANEL. Shows the active prompt, the per-scene chunk budget
// (current_chunk / 48) and the cumulative session_chunk, plus pause / resume /
// reset transport. Reset clears the model AND the local storyboard so the
// composer starts fresh.
export function NowPlaying() {
  const { status, pause, resume, reset } = useLongliveV2();
  const clearStoryboard = useStoryboard((s) => s.clear);
  const [snapshot, setSnapshot] = useState<LongliveV2StateMessage | null>(null);

  useLongliveV2State((msg) => setSnapshot(msg));

  // Clear stale snapshot on disconnect so a reconnect doesn't show old data.
  useEffect(() => {
    if (status === "disconnected") setSnapshot(null);
  }, [status]);

  if (status !== "ready" || !snapshot?.started) return null;

  const sceneChunk = snapshot.current_chunk ?? 0;
  const remaining = Math.max(0, SCENE_BUDGET - sceneChunk);

  async function onReset() {
    await reset();
    clearStoryboard();
  }

  return (
    <Panel label="Now playing">
      <p className="line-clamp-3 text-sm text-zinc-200">
        {typeof snapshot.current_prompt === "string"
          ? snapshot.current_prompt
          : "—"}
      </p>

      <div className="mt-2 flex items-center gap-3 font-mono tabular-nums text-[11px] text-zinc-400">
        <span>
          scene chunk {sceneChunk}/{SCENE_BUDGET}
        </span>
        <span>session {snapshot.session_chunk ?? 0}</span>
        {remaining <= 8 && (
          <span className="font-semibold text-amber-400">cut to continue</span>
        )}
      </div>

      <div className="mt-2 h-1 w-full overflow-hidden rounded bg-zinc-800">
        <div
          className={cn(
            "h-full rounded transition-[width]",
            remaining <= 8 ? "bg-red-500/80" : "bg-brand",
          )}
          style={{
            width: `${Math.min(100, (sceneChunk / SCENE_BUDGET) * 100)}%`,
          }}
        />
      </div>

      <div className="mt-3 flex items-center gap-1.5">
        {snapshot.paused ? (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => resume()}
            leadingIcon={<Icon name="play" />}
            className="flex-1"
          >
            Resume
          </Button>
        ) : (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => pause()}
            leadingIcon={<Icon name="pause" />}
            className="flex-1"
          >
            Pause
          </Button>
        )}
        <IconButton
          icon="reset"
          label="Reset"
          tone="danger"
          onClick={onReset}
        />
      </div>
    </Panel>
  );
}
