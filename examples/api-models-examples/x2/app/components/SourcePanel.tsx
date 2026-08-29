"use client";

import { useEffect, useRef, useState } from "react";
import { useX2 } from "@reactor-models/x2";
import type { X2SourceMode } from "@/app/lib/types";
import { Button, Panel, SegmentedToggle, Switch } from "./ui";
import { ImagePicker } from "./ImagePicker";
import { VideoPicker } from "./VideoPicker";
import { WebcamSource } from "./WebcamSource";

// The Source panel. X2 has no explicit start command: generation begins on
// its own once a non-empty prompt is set AND source frames are arriving, so
// this panel only picks what streams into the `source` track.
//
// Every mode only *produces* a MediaStreamTrack; the controller's
// useSourcePublisher is the single owner of the `source` slot, so mode
// switches swap tracks on one publisher instead of racing two.
//
//   - Webcam mode renders the self-view here (WebcamSource) and hands its
//     track up via onTrack. It stays mounted across the start transition
//     so the camera keeps streaming while generation runs.
//   - Video and image modes only pick media here; the stage plays/repeats
//     it and hands the captured track up the same way. Image mode streams
//     a still image as a constant feed — the drag-to-animate setup: prompt
//     it, then steer the subject with the pointer.
//   - Generating: the source is fixed for the run — Reset stops generation
//     (and clears prompt, reference image, and pointer) to change it.
export function SourcePanel({
  generating,
  keepBacklog,
  mode,
  onModeChange,
  onSelectVideo,
  onSelectImage,
  onTrack,
}: {
  generating: boolean;
  /** The model-reported keep_backlog policy (from the state_update snapshot). */
  keepBacklog: boolean;
  mode: X2SourceMode;
  onModeChange: (m: X2SourceMode) => void;
  onSelectVideo: (url: string, name: string) => void;
  onSelectImage: (url: string, name: string) => void;
  onTrack: (track: MediaStreamTrack | null) => void;
}) {
  const { reset, setKeepBacklog, status } = useX2();
  const ready = status === "ready";

  // `keepBacklog` is the model's echoed policy (via state_update): false drops
  // stale source frames so the edit tracks "now" (right for webcam); true
  // consumes every frame in order for smoother motion at the cost of growing
  // delay (right for clips and drag-to-animate). It's the source of truth, but
  // it only arrives after a round trip and takes effect from the next block, so
  // driving the switch straight off it leaves it dead for a beat after a click.
  //
  // `pending` holds the value the user just asked for and drives the switch
  // immediately (optimistic). It clears when the echo confirms it, or after a
  // timeout that falls back to the echoed truth — the command send is
  // fire-and-forget (it never rejects, and a rejected command comes back as a
  // separate command_error banner), so without the timeout a request the model
  // never applied would leave the switch stuck showing a lie.
  const [pending, setPending] = useState<boolean | null>(null);
  const pendingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const displayKeepBacklog = pending ?? keepBacklog;

  useEffect(() => {
    if (pending !== null && keepBacklog === pending) {
      setPending(null);
      if (pendingTimer.current) clearTimeout(pendingTimer.current);
    }
  }, [keepBacklog, pending]);

  useEffect(() => {
    return () => {
      if (pendingTimer.current) clearTimeout(pendingTimer.current);
    };
  }, []);

  const toggleKeepBacklog = (value: boolean) => {
    setPending(value);
    if (pendingTimer.current) clearTimeout(pendingTimer.current);
    pendingTimer.current = setTimeout(() => setPending(null), 4000);
    setKeepBacklog({ keep_backlog: value }).catch(console.error);
  };

  const handleModeChange = (m: X2SourceMode) => {
    if (generating) return; // source is fixed once a run has started — reset to switch
    onModeChange(m);
  };

  return (
    <Panel label="Source">
      {!generating && (
        <SegmentedToggle
          aria-label="Input source"
          value={mode}
          onChange={handleModeChange}
          options={[
            { value: "webcam", label: "Webcam" },
            { value: "video", label: "Video" },
            { value: "image", label: "Image" },
          ]}
        />
      )}

      {mode === "webcam" ? (
        <div className={generating ? "" : "mt-3"}>
          <WebcamSource onTrack={onTrack} />
        </div>
      ) : mode === "video" ? (
        !generating && (
          <div className="mt-3">
            <VideoPicker onSelect={onSelectVideo} />
          </div>
        )
      ) : (
        !generating && (
          <div className="mt-3">
            <ImagePicker onSelect={onSelectImage} />
          </div>
        )
      )}

      <div className="mt-3 flex items-start justify-between gap-3">
        <span className="min-w-0 text-xs text-zinc-400">
          Keep frame backlog
          <span className="block text-zinc-600">
            {pending !== null
              ? "Applying…"
              : displayKeepBacklog
                ? "Every frame: smoother motion, growing delay."
                : "Newest frames: bounded latency."}
          </span>
        </span>
        <Switch
          aria-label="Keep frame backlog"
          checked={displayKeepBacklog}
          disabled={!ready}
          onChange={toggleKeepBacklog}
        />
      </div>

      {generating ? (
        <div className="mt-3 flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            disabled={!ready}
            onClick={() => reset().catch(console.error)}
          >
            Reset
          </Button>
          <span className="text-xs text-zinc-500">
            Editing — reset to change the source
          </span>
        </div>
      ) : (
        <p className="mt-3 text-xs text-zinc-500">
          {ready
            ? "Generation starts once a prompt is set and frames are streaming."
            : "Connect, then set a prompt to start."}
        </p>
      )}
    </Panel>
  );
}
