"use client";

import { useEffect, useRef, useState } from "react";
import { useX2 } from "@reactor-models/x2";
import { errorMessage } from "./ui";

// The single owner of the `source` slot. Every input mode (webcam, video,
// image) only *produces* a MediaStreamTrack and hands it here; this hook is
// the only code that calls publish/unpublish on `source`, so mode switches
// can never race two publishers against the same transceiver.
//
// Reconciliation is serialized: one loop at a time compares the desired
// track (latest from the producer) with the published one and converges.
// Switching tracks does NOT unpublish first — the SDK's publishTrack sends
// the `publish_track` control request and then replaceTrack()s the new
// media onto the existing RTCRtpSender without renegotiating, so a direct
// re-publish is the safe path. Unpublish only happens when the producer
// goes away entirely (no track) or on unmount.
export function useSourcePublisher(
  track: MediaStreamTrack | null,
): string | null {
  const { publish, unpublish, status } = useX2();
  const [error, setError] = useState<string | null>(null);

  // What the producer currently wants on the wire.
  const desiredRef = useRef<MediaStreamTrack | null>(null);
  // What this hook believes is on the wire. Cleared when the transport
  // goes away (status leaves "ready") so the next ready re-publishes.
  const publishedRef = useRef<MediaStreamTrack | null>(null);
  // True while a reconcile loop is in flight — guarantees a single writer.
  const busyRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    desiredRef.current = track;
    if (status !== "ready") {
      publishedRef.current = null;
      return;
    }

    const reconcile = async () => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        // Converge on the latest desired track. The loop re-reads
        // desiredRef each pass, so a switch that lands mid-publish is
        // picked up by the same loop instead of spawning a second one.
        while (
          mountedRef.current &&
          desiredRef.current !== publishedRef.current
        ) {
          const want = desiredRef.current;
          try {
            if (want) {
              await publish("source", want);
            } else {
              await unpublish("source");
            }
            publishedRef.current = want;
            setError(null);
          } catch (e) {
            // Leave publishedRef as-is and stop; the next track/status
            // change retries. Never spin on a failing transport.
            setError(errorMessage(e));
            break;
          }
        }
      } finally {
        busyRef.current = false;
      }
    };
    void reconcile();
  }, [track, status, publish, unpublish]);

  // Mount flag for the reconcile loop. Mount-once on purpose: tying this to
  // any dep (e.g. `unpublish`, whose identity changes when the provider
  // recreates its store on auth arrival) would flip it to false mid-session
  // and permanently kill publishing.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Free the slot if this hook owns it — on unmount, and when the store is
  // torn down and rebuilt (new `unpublish` identity means the old transport
  // this hook published on is going away).
  useEffect(
    () => () => {
      if (publishedRef.current) {
        publishedRef.current = null;
        void unpublish("source").catch(() => {});
      }
    },
    [unpublish],
  );

  return error;
}
