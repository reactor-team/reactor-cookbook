"use client";

import { useEffect, useRef, useState } from "react";

// captureStream is widely supported but still missing from some DOM lib
// typings; declare the surface we use (plus Firefox's mozCaptureStream).
type CapturableVideo = HTMLVideoElement & {
  captureStream?: () => MediaStream;
  mozCaptureStream?: () => MediaStream;
};

// Video file -> a `camera` track. Rather than uploading the clip, we play it in
// a <video> element and grab its frames with captureStream(); this component
// *produces* that track (a single owner publishes it — see useCameraPublisher).
// The same element is the "your video" pane in the stage, so what you see is
// literally what the model receives (no separate playback clock, so no drift).
//
// Playback is gated on the model's run state: paused at frame 0 while set up
// (a still poster that also seeds the captured track), playing from the start
// once generation begins, and pausing/resuming in step with the model.
export function VideoSource({
  videoUrl,
  started,
  running,
  paused,
  onTrack,
}: {
  videoUrl: string;
  started: boolean;
  running: boolean;
  paused: boolean;
  onTrack: (track: MediaStreamTrack | null) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);

  // Capture the clip's frames into a track once metadata is available.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoUrl) return;
    let cancelled = false;
    let captured: MediaStreamTrack | null = null;

    const startCapture = () => {
      if (cancelled) return;
      const cap = v as CapturableVideo;
      const capture = cap.captureStream ?? cap.mozCaptureStream;
      if (!capture) {
        setError("This browser can't stream a video file.");
        return;
      }
      const track = capture.call(v).getVideoTracks()[0] ?? null;
      if (!track) return;
      track.contentHint = "detail"; // hold resolution; adapt framerate
      captured = track;
      onTrack(track);
    };

    if (v.readyState >= 1 /* HAVE_METADATA */) startCapture();
    else v.addEventListener("loadedmetadata", startCapture, { once: true });

    return () => {
      cancelled = true;
      v.removeEventListener("loadedmetadata", startCapture);
      captured?.stop();
      onTrack(null);
    };
  }, [videoUrl, onTrack]);

  // Drive playback off the model's run state: from the top once started,
  // pause/resume in step, back to the poster frame when not started.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!started) {
      v.pause();
      try {
        v.currentTime = 0;
      } catch {
        // currentTime can throw before metadata loads; ignored.
      }
      return;
    }
    if (running && !paused) v.play().catch(() => {});
    else v.pause();
  }, [started, running, paused, videoUrl]);

  return (
    <>
      <video
        ref={videoRef}
        src={videoUrl}
        loop
        muted
        playsInline
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
