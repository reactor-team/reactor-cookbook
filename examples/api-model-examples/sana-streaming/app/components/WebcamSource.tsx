"use client";

import { useEffect, useRef, useState } from "react";
import { Button, errorMessage } from "./ui";

// Webcam -> a `camera` track. This component only *produces* the track (and a
// small self-view in the Input panel); useCameraPublisher publishes it. It
// stays mounted across the start transition in webcam mode, so the camera keeps
// streaming while generation runs.
//
// We own the MediaStreamTrack so we can set contentHint = "detail": the model
// needs a stable camera resolution, but Chrome's encoder ramps resolution at
// stream start and on bandwidth dips. "detail" holds it steady and trades
// framerate instead.
export function WebcamSource({
  onTrack,
}: {
  onTrack: (track: MediaStreamTrack | null) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [track, setTrack] = useState<MediaStreamTrack | null>(null);
  const [denied, setDenied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);

  // Acquire the camera; re-acquire when retryKey increments.
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    setDenied(false);
    setError(null);
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 360 },
            facingMode: "user",
          },
        });
      } catch (e) {
        if (cancelled) return;
        if (e instanceof DOMException && e.name === "NotAllowedError") {
          setDenied(true);
        } else if (e instanceof DOMException && e.name === "NotReadableError") {
          setError("Camera is in use by another application.");
        } else {
          setError(errorMessage(e));
        }
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const videoTrack = stream.getVideoTracks()[0];
      videoTrack.contentHint = "detail"; // hold resolution; adapt framerate
      if (videoRef.current) videoRef.current.srcObject = stream;
      setTrack(videoTrack);
      onTrack(videoTrack);
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      onTrack(null);
    };
  }, [retryKey, onTrack]);

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-hidden rounded-md border border-zinc-800 bg-black">
        {/* Local self-view (muted preview of the published camera) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="aspect-video w-full object-cover"
        />
      </div>

      {denied && (
        <div className="flex items-center gap-2">
          <p className="text-xs text-red-400">
            Camera access denied. Allow camera for this site and retry.
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setRetryKey((k) => k + 1)}
          >
            Retry
          </Button>
        </div>
      )}
      {error && !denied && (
        <p className="text-xs text-red-400">Camera error: {error}</p>
      )}
      {!track && !denied && !error && (
        <p className="text-xs text-zinc-500">acquiring camera…</p>
      )}
    </div>
  );
}
