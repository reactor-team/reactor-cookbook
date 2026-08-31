"use client";

import { useState } from "react";
import {
  ClipDownloadButton,
  ClipPlayer,
  RecordingError,
  useReactor,
  type Clip,
} from "@reactor-team/js-sdk";

// Model-agnostic "Snap clip" panel.
//
// Captures the last `durationSeconds` of the live session and pops a
// modal with the SDK's built-in <ClipPlayer> preview and a download
// button. Drops into the sidebar of any example app — it does not
// depend on the typed model package at all, only @reactor-team/js-sdk.
//
// Recording is a base-SDK feature: it works the same way for every
// model with recording enabled, and the typed model packages
// (@reactor-models/helios, @reactor-models/lingbot, …) do not
// re-export the recording surface. So this is the one place in the
// example apps where importing directly from @reactor-team/js-sdk is
// idiomatic, not a smell. Drop the file in unchanged when you scaffold
// a new model example.
//
// `<ClipPlayer>` and `<ClipDownloadButton>` auto-inherit the JWT
// resolver from `<HeliosProvider getJwt={…} />` via React context
// (`@reactor-team/js-sdk` ≥ 2.10.1). No `getJwt` prop needed here.
// The one case where you would still pass it explicitly is when the
// clip UI renders through a portal outside the provider subtree (e.g.
// a Sonner toast living in `app/layout.tsx`) — capture the resolver
// with `reactor.getJwtResolver()` at action time and thread it down.
//
// As of `@reactor-team/js-sdk` ≥ 2.11.1, `requestClip` /
// `requestRecording` / `downloadClipAsFile` are exposed directly on
// the React store, so the recording surface is reachable via
// `useReactor((s) => s.requestClip)` without the `s.internal.reactor`
// escape hatch. The clip components also accept `onError` / `onSuccess`
// callbacks — this panel routes both player and download failures
// into its inline error line, and clears that line when a download
// completes.
export interface SnapClipProps {
  /** Length of the snap, in seconds. Default 10. */
  durationSeconds?: number;
  /**
   * Suggested filename for the saved MP4. Default
   * `reactor-clip-<unix-seconds>.mp4`.
   */
  filename?: string;
  /** Optional override for the button label. */
  label?: string;
}

export function SnapClip({
  durationSeconds = 10,
  filename,
  label,
}: SnapClipProps) {
  const { status, requestClip } = useReactor((s) => ({
    status: s.status,
    requestClip: s.requestClip,
  }));

  const [clip, setClip] = useState<Clip | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (status !== "ready") return null;

  async function snap() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const c = await requestClip(durationSeconds);
      setClip(c);
    } catch (e) {
      setError(
        e instanceof RecordingError
          ? `${e.code}: ${e.reason}`
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setBusy(false);
    }
  }

  const downloadName =
    filename ?? `reactor-clip-${Math.floor(Date.now() / 1000)}.mp4`;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <span className="text-[10px] uppercase tracking-wider text-zinc-500">
        Capture
      </span>
      <button
        onClick={snap}
        disabled={busy}
        className="mt-2 w-full rounded-md bg-brand px-3 py-2 text-sm font-medium text-brand-fg hover:opacity-90 disabled:opacity-40"
      >
        {busy ? "Capturing…" : (label ?? `Snap last ${durationSeconds}s`)}
      </button>
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {clip && (
        <ClipModal
          clip={clip}
          filename={downloadName}
          onClose={() => setClip(null)}
          onError={(e) => setError(e.message)}
          onDownloaded={() => setError(null)}
        />
      )}
    </div>
  );
}

function ClipModal({
  clip,
  filename,
  onClose,
  onError,
  onDownloaded,
}: {
  clip: Clip;
  filename: string;
  onClose: () => void;
  onError: (error: Error) => void;
  onDownloaded: () => void;
}) {
  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="flex w-full max-w-2xl flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4 shadow-xl"
      >
        <div className="flex items-center justify-between gap-3">
          <span className="text-[10px] uppercase tracking-wider text-zinc-500">
            Clip · {clip.kind}
          </span>
          <button
            onClick={onClose}
            className="rounded-md border border-zinc-800 px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
          >
            Close
          </button>
        </div>

        <ClipPlayer
          clip={clip}
          onError={onError}
          className="w-full overflow-hidden rounded-lg border border-zinc-800"
        />

        <div className="flex justify-end">
          <ClipDownloadButton
            clip={clip}
            filename={filename}
            onSuccess={onDownloaded}
            onError={onError}
          />
        </div>
      </div>
    </div>
  );
}
