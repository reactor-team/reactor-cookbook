"use client";

/**
 * Client-side access to the extractor's dense per-frame curves for the
 * PLAYBACK live-effects layer (Step C).
 *
 * The curves are fetched once (from /api/bundles/features, which reads
 * the untouched parquet) before playback starts, then indexed by
 * playback time every animation frame:
 *
 *   row = floor(t * frameRate)   // frameRate = meta.target_frame_rate (60)
 *
 * Playback time and the extractor's frame grid share the same time base
 * (t=0 = start of the source file), so no offset is applied. This layer
 * never calls the LLM and never sends anything to Reactor.
 */

export interface FeatureFrames {
  frameRate: number;
  rowCount: number;
  /** Curve value at playback time t (seconds), 0 if column missing/OOB. */
  valueAt(t: number, column: string): number;
  has(column: string): boolean;
}

interface FeaturesResponse {
  frameRate: number;
  rowCount: number;
  availableColumns: string[];
  columns: Record<string, number[]>;
}

export async function fetchFeatureFrames(bundleId: string): Promise<FeatureFrames> {
  const res = await fetch(
    `/api/bundles/features?id=${encodeURIComponent(bundleId)}`,
  );
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `Feature fetch failed: ${res.status}`);
  }
  const data = (await res.json()) as FeaturesResponse;
  const columns = new Map<string, Float32Array>();
  for (const [name, values] of Object.entries(data.columns)) {
    columns.set(name, Float32Array.from(values));
  }

  return {
    frameRate: data.frameRate,
    rowCount: data.rowCount,
    has: (column) => columns.has(column),
    valueAt(t: number, column: string): number {
      const curve = columns.get(column);
      if (!curve || t < 0) return 0;
      const row = Math.floor(t * data.frameRate);
      return row < curve.length ? curve[row] : 0;
    },
  };
}
