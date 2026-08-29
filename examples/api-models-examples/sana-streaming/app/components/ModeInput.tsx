"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { Button, Panel, SegmentedToggle } from "./ui";
import type { SanaMode } from "../lib/types";
import { startGeneration } from "../lib/state";
import { Playback } from "./Playback";
import { SeedField } from "./SeedField";
import { VideoPicker } from "./VideoPicker";
import { WebcamSource } from "./WebcamSource";

// The Input panel. Phase-aware on the model's `started` flag:
//
//   - Setup (!started): pick the source (webcam / video), choose a clip in
//     video mode, set the seed, and Start.
//   - Live (started): the Start control is replaced by Playback.
//
// In webcam mode the self-view lives here and stays mounted across the start
// transition, so the camera keeps streaming `camera` while generation runs. In
// video mode the clip preview lives in the stage instead; this panel only picks
// it. Both sources stream into the same `camera` track, which is the only path
// the model has from v2.0.0 on, so Start is just `start`.
export function ModeInput({
  started,
  paused,
  mode,
  modelSeed,
  hasVideoUrl,
  onModeChange,
  onSelectVideo,
  onTrack,
  onReset,
}: {
  started: boolean;
  paused: boolean;
  mode: SanaMode;
  modelSeed: number;
  hasVideoUrl: boolean;
  onModeChange: (m: SanaMode) => void;
  onSelectVideo: (url: string, name: string) => void;
  onTrack: (track: MediaStreamTrack | null) => void;
  onReset: () => void;
}) {
  const { start, status } = useSanaStreaming();
  const ready = status === "ready";

  const handleModeChange = (m: SanaMode) => {
    if (started) return; // source is fixed once a run has started — reset to switch
    onModeChange(m);
  };

  const onStart = () => {
    void startGeneration({ start });
  };

  return (
    <Panel label="Input">
      {!started && (
        <SegmentedToggle
          aria-label="Input source"
          value={mode}
          onChange={handleModeChange}
          options={[
            { value: "webcam", label: "Webcam" },
            { value: "video", label: "Video" },
          ]}
        />
      )}

      {mode === "webcam" ? (
        <div className={started ? "" : "mt-3"}>
          <WebcamSource onTrack={onTrack} />
        </div>
      ) : (
        !started && (
          <div className="mt-3">
            <VideoPicker onSelect={onSelectVideo} />
          </div>
        )
      )}

      {started ? (
        <div className="mt-3">
          <Playback paused={paused} onReset={onReset} />
        </div>
      ) : (
        <div className="mt-3 flex flex-col gap-3">
          <Button
            variant="primary"
            size="md"
            className="w-full"
            data-testid="start"
            disabled={!ready || (mode === "video" && !hasVideoUrl)}
            onClick={onStart}
          >
            Start
          </Button>
          <SeedField modelSeed={modelSeed} />
          {!ready ? (
            <p className="text-xs text-zinc-500">connect to start</p>
          ) : (
            mode === "video" &&
            !hasVideoUrl && (
              <p className="text-xs text-zinc-500">pick a clip to stream</p>
            )
          )}
        </div>
      )}
    </Panel>
  );
}
