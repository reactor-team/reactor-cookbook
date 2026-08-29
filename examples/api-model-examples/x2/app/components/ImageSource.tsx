"use client";

import { useEffect, useRef, useState } from "react";

// captureStream is widely supported on canvas but still missing from some
// DOM lib typings; declare the surface we use.
type CapturableCanvas = HTMLCanvasElement & {
  captureStream?: (frameRate?: number) => MediaStream;
};

// The model's native pacing; matches the demo clips (fps: 24).
const STREAM_FPS = 24;

// The streamed frame is a fixed 16:9 canvas. The model picks its output
// resolution bucket from the source stream's aspect ratio, so pinning the
// canvas to 16:9 pins the whole run — preview pane, source stream, and
// edited output — to the same landscape shape regardless of the picked
// image's own aspect.
const FRAME_WIDTH = 1280;
const FRAME_HEIGHT = 720;

// Still image -> a constant `source` stream. The image is drawn to a canvas
// and captured with canvas.captureStream(24); the draw loop keeps repainting
// the same frame so the capturer keeps emitting it — from the model's side
// this is indistinguishable from a video of a motionless scene, which is
// exactly the drag-to-animate setup: a still subject the pointer brings to
// life. This component *produces* the track; useSourcePublisher publishes
// it. The same canvas is the "original" pane in the stage, so what you see
// is what the model receives.
export function ImageSource({
  imageUrl,
  onTrack,
}: {
  imageUrl: string;
  onTrack: (track: MediaStreamTrack | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;
    let cancelled = false;
    let drawTimer: ReturnType<typeof setInterval> | null = null;
    let captured: MediaStreamTrack | null = null;
    setError(null);

    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      canvas.width = FRAME_WIDTH;
      canvas.height = FRAME_HEIGHT;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setError("Canvas 2D is unavailable in this browser.");
        return;
      }
      // Cover-crop the image into the 16:9 frame (center crop, no
      // distortion, no letterbox bars for the model to "edit") — the same
      // resize-and-crop the model applies to source frames anyway.
      const scale = Math.max(
        FRAME_WIDTH / img.naturalWidth,
        FRAME_HEIGHT / img.naturalHeight,
      );
      const cropWidth = FRAME_WIDTH / scale;
      const cropHeight = FRAME_HEIGHT / scale;
      const cropX = (img.naturalWidth - cropWidth) / 2;
      const cropY = (img.naturalHeight - cropHeight) / 2;
      const draw = () =>
        ctx.drawImage(
          img,
          cropX,
          cropY,
          cropWidth,
          cropHeight,
          0,
          0,
          FRAME_WIDTH,
          FRAME_HEIGHT,
        );
      draw();

      const capture = (canvas as CapturableCanvas).captureStream;
      if (!capture) {
        setError("This browser can't stream a canvas.");
        return;
      }
      const track =
        capture.call(canvas, STREAM_FPS).getVideoTracks()[0] ?? null;
      if (!track) return;
      track.contentHint = "detail"; // hold resolution; adapt framerate
      captured = track;
      onTrack(track);

      // captureStream only emits when the canvas repaints, so keep marking
      // it dirty at the stream rate to repeat the frame indefinitely.
      drawTimer = setInterval(draw, 1000 / STREAM_FPS);
    };
    img.onerror = () => {
      if (!cancelled) setError("Could not load the image.");
    };
    img.src = imageUrl;

    return () => {
      cancelled = true;
      if (drawTimer) clearInterval(drawTimer);
      captured?.stop();
      onTrack(null);
    };
  }, [imageUrl, onTrack]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-contain"
      />
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-4 text-center">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}
    </>
  );
}
