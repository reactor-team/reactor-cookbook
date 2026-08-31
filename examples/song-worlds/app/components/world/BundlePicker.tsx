"use client";

import { useEffect, useState } from "react";
import { GLASS_PANEL_ERROR, GLASS_PANEL_INTERACTIVE } from "../../lib/ui/glass";
import type { BundleListEntry } from "../../lib/world/types";

/**
 * Song selection. The app no longer analyzes audio itself — it consumes
 * bundles ({song_id}.meta.json + {song_id}.features.parquet) produced by
 * the offline extractor and listed by /api/bundles from the configured
 * output dirs. Bundles that failed validation (e.g. an unrecognized
 * pipeline_version) are shown disabled with their error, not hidden.
 *
 * FIXTURE_PREFIX hides synthetic/test/stress/edge-case bundles from this
 * picker — display only, the API still returns them and extraction is
 * unaffected. These exist to exercise the pipeline (synthetic_*, the
 * bare "test"/"test2_90s", stress_*, edge_*), not as real songs a
 * listener would pick.
 */
const FIXTURE_PREFIX = /^(synthetic_|test|stress_|edge_)/i;

export function BundlePicker({
  onPick,
  disabled,
}: {
  onPick: (bundle: BundleListEntry) => void;
  disabled?: boolean;
}) {
  const [bundles, setBundles] = useState<BundleListEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/bundles")
      .then(async (res) => {
        const data = (await res.json()) as {
          bundles?: BundleListEntry[];
          error?: string;
        };
        if (cancelled) return;
        if (!res.ok || !data.bundles) {
          setError(data.error ?? `Bundle list failed: ${res.status}`);
          return;
        }
        setBundles(data.bundles.filter((b) => !FIXTURE_PREFIX.test(b.songId)));
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className={`w-full p-4 text-sm text-red-300 ${GLASS_PANEL_ERROR}`}>
        {error}
      </div>
    );
  }
  if (!bundles) {
    return (
      <div className="flex justify-center py-8">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/15 border-t-brand" />
      </div>
    );
  }
  if (bundles.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-zinc-400">
        No songs found. Run the extractor pipeline, or point
        EXTRACTOR_OUTPUT_DIRS at its output folder.
      </p>
    );
  }

  return (
    <div className="w-full space-y-2">
      <p className="text-[10px] uppercase tracking-wider text-zinc-400">
        Choose a song
      </p>
      {bundles.map((b) => (
        <button
          key={b.id}
          disabled={disabled || !!b.error}
          onClick={() => onPick(b)}
          className={`flex w-full items-center justify-between gap-3 p-3.5 text-left text-sm ${
            b.error
              ? "cursor-not-allowed rounded-2xl border border-red-400/20 bg-red-500/10 opacity-70 backdrop-blur-xl"
              : GLASS_PANEL_INTERACTIVE
          } ${disabled ? "pointer-events-none opacity-50" : ""}`}
        >
          <div className="min-w-0">
            <span className="font-medium text-zinc-100">{b.songId}</span>
            <span className="ml-2 text-xs text-zinc-500">{b.dir}</span>
            {b.error && (
              <p className="mt-1 truncate text-xs text-red-300" title={b.error}>
                {b.error}
              </p>
            )}
          </div>
          {!b.error && (
            <div className="shrink-0 font-mono text-[11px] text-zinc-400">
              {formatDuration(b.durationSec)} ·{" "}
              {b.bpm !== null ? `${Math.round(b.bpm)} bpm` : "no beat grid"} ·{" "}
              {b.structureBackend ?? "no structure"} · {b.sectionCount} sec
            </div>
          )}
        </button>
      ))}
    </div>
  );
}

function formatDuration(sec: number): string {
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
