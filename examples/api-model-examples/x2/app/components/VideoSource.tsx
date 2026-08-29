"use client";

import { useEffect, useRef, useState } from "react";

// captureStream is widely supported but still missing from some DOM lib
// typings; declare the surface we use (plus Firefox's mozCaptureStream).
type CapturableVideo = HTMLVideoElement & {
  captureStream?: () => MediaStream;
  mozCaptureStream?: () => MediaStream;
};

// Video file -> a `source` track. Rather than uploading the clip, we play it
// in a <video> element and grab its frames with captureStream(); this
// component *produces* that track (a single owner publishes it — see
// useSourcePublisher). The same element is the "original" pane in the stage,
// so what you see is literally what the model receives.
//
// The clip plays and loops from the moment it is selected: X2 consumes source
// frames and starts generating on its own once a prompt is set and frames
// are arriving, so a continuously-playing source is exactly the contract the
// model expects.
export function VideoSource({
  videoUrl,
  onTrack,
}: {
  videoUrl: string;
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

  return (
    <>
      <video
        ref={videoRef}
        src={videoUrl}
        autoPlay
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
