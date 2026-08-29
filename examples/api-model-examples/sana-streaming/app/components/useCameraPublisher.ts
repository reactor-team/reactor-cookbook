"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { useEffect, useState } from "react";
import { errorMessage } from "./ui";

// Single owner of the `camera` slot. Whichever input source is mounted (webcam
// in the sidebar, or the video source in the stage) only *produces* a track and
// hands it here; this hook publishes whichever track is current. It always
// unpublishes before publishing, so switching sources can't race into
// "publisher slot already taken" — the prior source's slot is freed first.
// Lives in the workspace so it stays mounted no matter which source is active.
export function useCameraPublisher(
  track: MediaStreamTrack | null,
): string | null {
  const { publish, unpublish, status } = useSanaStreaming();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status !== "ready") return;
    let cancelled = false;
    (async () => {
      try {
        await unpublish("camera").catch(() => {});
        if (cancelled || !track) return;
        await publish("camera", track);
        if (!cancelled) setError(null);
      } catch (e) {
        if (!cancelled) setError(errorMessage(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [track, status, publish, unpublish]);

  // Free the camera slot when the workspace unmounts.
  useEffect(() => () => void unpublish("camera").catch(() => {}), [unpublish]);

  return error;
}
