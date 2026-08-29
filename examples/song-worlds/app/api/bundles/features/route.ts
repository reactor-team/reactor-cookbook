import { NextResponse } from "next/server";
import { BundleError, loadFeatures, loadMeta, resolveBundle } from "../../../lib/server/bundle";
import { CLIP_SECONDS } from "../../../lib/server/config";

/**
 * GET /api/bundles/features?id={dir}/{song_id}&columns=a,b,c
 *
 * Dense per-frame curves for the PLAYBACK live-effects layer (Step C).
 * The client indexes them by playback time — row = floor(t * frameRate)
 * with frameRate = meta.target_frame_rate (60) — for frame-synced
 * pulses. This path never touches the LLM and never talks to Reactor.
 *
 * Columns are matched by name against the parquet's own schema; unknown
 * names are simply absent from the response. This is a fresh read of
 * the untouched parquet — Stage 1's earlier read did not consume it.
 */

/** Default curve set for the effects overlay. */
const DEFAULT_COLUMNS = [
  "frame_time_sec",
  "full_mix_rms",
  "drums_onset_envelope",
  "bass_rms",
  "full_mix_spectral_centroid",
];

const MAX_COLUMNS = 16;

export async function GET(request: Request) {
  const url = new URL(request.url);
  const id = url.searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing ?id" }, { status: 400 });

  const columnsParam = url.searchParams.get("columns");
  const want = (
    columnsParam ? columnsParam.split(",").map((c) => c.trim()).filter(Boolean) : DEFAULT_COLUMNS
  ).slice(0, MAX_COLUMNS);

  try {
    const bundle = await resolveBundle(id);
    const meta = await loadMeta(bundle);
    // Playback stops at CLIP_SECONDS, so the effects layer only needs
    // the same window (trims the JSON payload for long songs too).
    const table = await loadFeatures(bundle, want, meta.target_frame_rate, CLIP_SECONDS);

    const columns: Record<string, number[]> = {};
    for (const [name, values] of table.columns) {
      // float32 precision is plenty for visuals; trim JSON size.
      columns[name] = Array.from(values, (v) => Math.round(v * 1e6) / 1e6);
    }

    return NextResponse.json({
      frameRate: meta.target_frame_rate,
      rowCount: table.rowCount,
      availableColumns: table.allColumnNames,
      columns,
    });
  } catch (e) {
    if (e instanceof BundleError) {
      return NextResponse.json({ error: e.message }, { status: e.status });
    }
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Features failed" },
      { status: 500 },
    );
  }
}
