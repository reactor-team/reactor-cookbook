"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useLtx2 } from "@reactor-models/ltx2";
import type { ReactorStatus } from "@reactor-team/js-sdk";
import { statusLine } from "@/app/lib/machine";
import { type Ltx2UiState } from "@/app/lib/types";

// The stage owns its own <video> rather than using the generated
// <Ltx2MainVideoView>, for two reasons:
//
//   1. AUDIO. This model's voice is half the product, so the element has to
//      carry `main_audio` alongside `main_video` in one MediaStream — the two
//      tracks share a sample clock and must not be played by separate
//      elements, or they drift apart.
//   2. TTFF. requestVideoFrameCallback fires only when a *new* frame is
//      composited, so it measures first-frame latency at the display rather
//      than inferring it from a message. The stall watch reads the same
//      callback: when it goes quiet mid-run, the stream has died.
//
// If you don't need either, <Ltx2MainVideoView /> is a one-liner.

// How long the display may go without a newly-composited frame, mid-run,
// before the stream is declared stalled. The snapshot and the media transport
// fail independently, so `generating` can stay true after the frames have
// died; frame age at the display is the only client-side signal there is.
// Eight seconds is the threshold the production demo runs.
const STALL_AFTER_MS = 8_000;
export function Stage({
  status,
  ui,
  hasTake,
  onFirstFrame,
}: {
  status: ReactorStatus;
  ui: Ltx2UiState;
  /** The composited frame belongs to a take that is still current. */
  hasTake: boolean;
  onFirstFrame: () => void;
}) {
  const { tracks } = useLtx2();
  const videoTrack = tracks["main_video"] ?? null;
  const audioTrack = tracks["main_audio"] ?? null;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [needsAudioUnlock, setNeedsAudioUnlock] = useState(false);
  const lastFrameAt = useRef(0);
  const [stalled, setStalled] = useState(false);

  const mediaStream = useMemo(() => {
    const list: MediaStreamTrack[] = [];
    if (videoTrack) list.push(videoTrack);
    if (audioTrack) list.push(audioTrack);
    return list.length ? new MediaStream(list) : null;
  }, [videoTrack, audioTrack]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    el.srcObject = mediaStream;
    if (!mediaStream) return;
    el.play().catch(() => {
      // Autoplay with audio needs a user gesture. The model's voice matters
      // here, so we ask for the gesture rather than falling back to muted.
      setNeedsAudioUnlock(true);
    });
  }, [mediaStream]);

  // Report every newly-composited frame; the app shell decides which one
  // counts (only frames after `generation_started` arms the measurement).
  useEffect(() => {
    const el = videoRef.current;
    if (!el || !("requestVideoFrameCallback" in el)) return;
    let handle = 0;
    let cancelled = false;
    const onFrame = () => {
      if (cancelled) return;
      lastFrameAt.current = performance.now();
      onFirstFrame();
      handle = el.requestVideoFrameCallback(onFrame);
    };
    handle = el.requestVideoFrameCallback(onFrame);
    return () => {
      cancelled = true;
      el.cancelVideoFrameCallback?.(handle);
    };
  }, [onFirstFrame, mediaStream]);

  // The stall watch. Armed only while the snapshot says frames should be
  // flowing: a run is on, not paused, and past warm-up (`seconds_sent > 0`,
  // the same signal statusLine reads, since nothing composites while the
  // leading window denoises). Re-seeding the frame clock on arm means a slow
  // warm-up or the idle gap between takes can never trip it.
  const streaming = ui.generating && !ui.paused && ui.secondsSent > 0;
  useEffect(() => {
    if (!streaming) {
      setStalled(false);
      return;
    }
    lastFrameAt.current = performance.now();
    const timer = setInterval(
      () =>
        setStalled(performance.now() - lastFrameAt.current > STALL_AFTER_MS),
      1000,
    );
    return () => clearInterval(timer);
  }, [streaming]);

  // No take has run since the session was last cleared, so there is nothing
  // on the frame worth showing.
  //
  // `hasTake` is the ONLY input here on purpose. It flips on
  // `generation_started` and off on `reset`/disconnect — both discrete model
  // events — so it cannot get stuck the way a derived progress signal can. A
  // previous version also covered the frame during warm-up, which meant a
  // progress field that never advanced left a black rectangle over a take
  // that was streaming. Covering the stage is the one thing that must never
  // depend on a signal that might not arrive.
  const stageIsEmpty = !hasTake;
  const line = statusLine(status, ui);
  const connecting = status === "connecting" || status === "waiting";

  return (
    // The frame is aspect-locked at every breakpoint and centered in whatever
    // the sidebar leaves it, rather than stretched to fill that space with the
    // video letterboxing inside. Same picture either way, but the border then
    // hugs the picture instead of enclosing 150px of black above and below it
    // — leftover room reads as layout, which it is, rather than as bars.
    //
    // Width is the limiting dimension in any normal desktop window, so the fit
    // is exact. `max-h-full` is the guard for a short, wide window, where the
    // box clamps and object-contain pillarboxes instead.
    <div className="flex min-h-0 items-center justify-center lg:flex-1">
      <div className="relative aspect-[640/352] max-h-full w-full overflow-hidden rounded-xl border border-edge bg-black">
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full object-contain"
          playsInline
          autoPlay
        />

        {/* The status line goes in exactly one of two places, never both: over
            the frame while the stage is empty, and in the caption row below
            once there is a take to look at. Printing it over a frame is the
            case to avoid — "take complete" across the subject's face is not a
            status, it is graffiti.

            The `bg-black` is also what clears the stage. The <video> holds a
            live WebRTC stream that keeps compositing between takes, so the
            last frame of a take stays up indefinitely — including through a
            `reset`, which is supposed to leave nothing behind, and through the
            warm-up of the next take, where the previous face would otherwise
            sit under a "warming up" caption.

            Covering the frame rather than detaching `srcObject` is
            deliberate: re-attaching the stream for the next take risks
            dropping its first composited frames, which is exactly what the
            TTFF measurement above depends on. */}
        {stageIsEmpty && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black">
            {connecting && (
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-brand/60 border-t-transparent" />
            )}
            <span className="px-4 text-center text-sm text-zinc-500">
              {line}
            </span>
          </div>
        )}

        {/* The one failure the model cannot report about itself: the stream
            died but the session, and so its snapshot, is still up. `stop`
            stays in `valid_commands` for the whole run and the conditions
            survive it, so recovery is one Stop and one Start away. Surfaced
            rather than auto-recovered, like every other failure in this
            app. */}
        {stalled && !stageIsEmpty && (
          <div className="absolute bottom-3 left-3 rounded-md border border-red-500/40 bg-black/70 px-2.5 py-1 font-mono text-[11px] text-red-400">
            Stalled: no new frame in {STALL_AFTER_MS / 1000}s. Press Stop, then
            start again.
          </div>
        )}

        {/* Audio unlock. The track arrives seconds after the Connect click,
            by which point the browser's transient activation has lapsed, so
            play() with sound is refused and a second gesture is genuinely
            required. Measured, not assumed: priming the element during the
            click does not help, because it has no source to play yet.
            So this stays — but as a chip in the corner, not a wall over the
            stage, and only when play() actually rejected. */}
        {needsAudioUnlock && (
          <button
            onClick={() => {
              videoRef.current?.play().then(
                () => setNeedsAudioUnlock(false),
                () => {},
              );
            }}
            className="absolute bottom-3 right-3 rounded-md border border-brand/40 bg-black/70 px-2.5 py-1 font-mono text-[11px] text-brand-light hover:border-brand"
          >
            Enable audio
          </button>
        )}
      </div>
    </div>
  );
}
