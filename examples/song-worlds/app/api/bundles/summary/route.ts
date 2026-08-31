import { NextResponse } from "next/server";
import { BundleError, loadFeatures, loadMeta, resolveBundle } from "../../../lib/server/bundle";
import { CLIP_SECONDS } from "../../../lib/server/config";
import { SUMMARY_COLUMNS, buildMusicalSummary } from "../../../lib/server/summarize";

/**
 * GET /api/bundles/summary?id={dir}/{song_id}
 *
 * Runs Stage 1 (deterministic reduction, no LLM) and returns the
 * MusicalSummary as inspectable JSON, so its quality can be checked
 * independently of the LLM. The same function feeds /api/compose.
 */
export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing ?id" }, { status: 400 });

  try {
    const bundle = await resolveBundle(id);
    const meta = await loadMeta(bundle);
    // Only the first CLIP_SECONDS of the song are used, end to end.
    const features = await loadFeatures(
      bundle,
      SUMMARY_COLUMNS,
      meta.target_frame_rate,
      CLIP_SECONDS,
    );
    const summary = buildMusicalSummary(meta, features, CLIP_SECONDS);
    return NextResponse.json({ summary });
  } catch (e) {
    if (e instanceof BundleError) {
      return NextResponse.json({ error: e.message }, { status: e.status });
    }
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Summary failed" },
      { status: 500 },
    );
  }
}
