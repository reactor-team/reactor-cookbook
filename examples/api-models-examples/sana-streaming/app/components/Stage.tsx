"use client";

import { SanaStreamingMainVideoView } from "@reactor-models/sana-streaming";
import type { SanaMode, SanaState } from "../lib/types";
import { VideoSource } from "./VideoSource";

// The stage.
//   - webcam mode: a single pane of the model's edited output (the webcam
//     self-view lives in the Input panel).
//   - video mode: side by side — the edited output on the left, the clip you're
//     streaming in on the right. The right pane owns the published `camera`
//     track, so it is literally the frames the model edits — the two panes
//     share one source and can't drift.
export function Stage({
  state,
  mode,
  videoUrl,
  cleared,
  onTrack,
}: {
  state: SanaState;
  mode: SanaMode;
  videoUrl: string | null;
  // After a reset the WebRTC <video> would freeze on the last received frame
  // (the model emits nothing new) - cover it until generation runs again.
  cleared: boolean;
  onTrack: (track: MediaStreamTrack | null) => void;
}) {
  const statusRow = (
    <div className="absolute bottom-0 inset-x-0 flex min-w-0 items-center gap-3 bg-gradient-to-t from-black/60 p-3 font-mono text-xs text-zinc-400">
      <span className="shrink-0">
        {state.running ? (state.paused ? "paused" : "streaming") : "idle"}
      </span>
      <span className="shrink-0">chunk {state.currentChunk}</span>
      {state.currentPrompt && (
        <span className="min-w-0 flex-1 truncate">"{state.currentPrompt}"</span>
      )}
    </div>
  );

  const editedPane = (label: string) => (
    <>
      <SanaStreamingMainVideoView
        videoObjectFit="contain"
        className="absolute inset-0 h-full w-full"
      />
      {cleared && (
        <div
          data-testid="stage-cleared"
          className="absolute inset-0 bg-black"
        />
      )}
      <span className="absolute top-2 left-2 max-w-[80%] truncate rounded bg-black/60 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
        {label}
      </span>
    </>
  );

  if (mode === "video") {
    return (
      <section
        data-testid="stage-split"
        className="relative flex min-h-0 flex-col gap-3 max-lg:h-[40vh] lg:flex-1 lg:flex-row"
      >
        <div
          data-testid="pane-input"
          className="relative w-full min-h-0 flex-1 overflow-hidden rounded-lg border border-zinc-800 bg-black"
        >
          {videoUrl ? (
            <VideoSource
              videoUrl={videoUrl}
              started={state.started}
              running={state.running}
              paused={state.paused}
              onTrack={onTrack}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center">
              <p className="text-xs text-zinc-500">
                Select a video to stream into the model.
              </p>
            </div>
          )}
          <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-xs text-zinc-400">
            original
          </span>
        </div>
        <div
          data-testid="pane-edited"
          className="relative w-full min-h-0 flex-1 overflow-hidden rounded-lg border border-zinc-800 bg-black"
        >
          {editedPane(state.currentPrompt ?? "edited")}
        </div>
        {statusRow}
      </section>
    );
  }

  return (
    <section className="relative flex-1 min-h-[40vh] lg:min-h-0 overflow-hidden rounded-lg border border-zinc-800 bg-black">
      {editedPane(state.currentPrompt ?? "edited")}
      {statusRow}
    </section>
  );
}
