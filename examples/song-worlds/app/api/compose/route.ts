import { NextResponse } from "next/server";
import { BundleError, loadFeatures, loadMeta, resolveBundle } from "../../lib/server/bundle";
import { CLIP_SECONDS } from "../../lib/server/config";
import { SUMMARY_COLUMNS, buildMusicalSummary } from "../../lib/server/summarize";
import { ComposeError, composeWithNemotron } from "../../lib/server/compose";
import { loadSeedCatalog } from "../../lib/server/seedCatalog";
import type { CompositionResult, RhythmData, SeedRef } from "../../lib/world/types";

/**
 * POST /api/compose  { bundleId: "{dir}/{song_id}" }
 *
 * The full offline composition pass, before playback ever starts:
 *
 *   read extractor bundle → Stage 1 deterministic reduction (no LLM)
 *                         → Stage 2 one Nemotron call
 *                         → sanitized, anchor-aligned WorldEvent series
 *
 * Runs server-side because NVIDIA_NEMO_KEY is a secret and must never
 * reach browser JS. The LLM is never called again after this — playback
 * only executes the returned events.
 *
 * All timestamps in the response are seconds on the extractor's time
 * base (t=0 = start of the extracted song); timeOffsetSec carries any
 * future sub-window offset explicitly (always 0 today).
 */
export async function POST(request: Request) {
  let bundleId: string;
  try {
    const body = (await request.json()) as { bundleId?: string };
    if (typeof body.bundleId !== "string" || !body.bundleId) {
      return NextResponse.json({ error: "Missing bundleId" }, { status: 400 });
    }
    bundleId = body.bundleId;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const bundle = await resolveBundle(bundleId);
    const meta = await loadMeta(bundle);
    // Stage 1 reads the parquet (read-only — the file and its dense
    // curves remain fully available to the playback features route).
    // Only the first CLIP_SECONDS of the song are summarized/composed;
    // the window starts at t=0 so timeOffsetSec stays 0.
    const features = await loadFeatures(
      bundle,
      SUMMARY_COLUMNS,
      meta.target_frame_rate,
      CLIP_SECONDS,
    );
    const summary = buildMusicalSummary(meta, features, CLIP_SECONDS);

    // Dump the Stage 1 result so it can be inspected independently of
    // the LLM (also available standalone at /api/bundles/summary).
    console.info(`[stage1] MusicalSummary for ${summary.songId}:`);
    console.info(JSON.stringify(summary, null, 2));

    // The seed-image catalog is the menu Nemotron chooses from; every
    // event comes back with a validated seedId.
    const seedCatalog = await loadSeedCatalog();
    const composed = await composeWithNemotron(summary, seedCatalog);

    // Distinct seeds actually used, in order of first use — playback
    // pre-uploads exactly these before starting.
    const seeds: SeedRef[] = [];
    for (const event of composed.events) {
      if (seeds.some((s) => s.id === event.seedId)) continue;
      const entry = seedCatalog.find((s) => s.id === event.seedId);
      if (!entry) continue; // can't happen post-sanitize; belt-and-braces
      seeds.push({
        id: entry.id,
        category: entry.category,
        oneLiner: entry.oneLiner,
        imageUrl: `/api/seeds/image?id=${encodeURIComponent(entry.id)}`,
      });
    }

    // Raw rhythm grid for the live layer — clipped to the same window as
    // the summary/playback (CLIP_SECONDS from t=0). The extractor's exact
    // beats/downbeats/chords, forwarded rather than re-derived on the
    // client. low_confidence_rhythm rides along so the client knows when
    // NOT to trust the grid (fall back to energy curves).
    const inWindow = (t: number) => t >= 0 && t < CLIP_SECONDS;
    const rhythm: RhythmData = {
      bpm: meta.tempo.bpm,
      beats: meta.beats.filter(inWindow),
      downbeats: meta.downbeats.filter(inWindow),
      lowConfidenceRhythm: meta.tempo.low_confidence_rhythm,
      downbeatPhaseUncertain: meta.tempo.downbeat_phase_uncertain,
      chords: (meta.chords ?? []).filter((c) => c.start < CLIP_SECONDS),
    };

    const result: CompositionResult = {
      events: composed.events,
      seeds,
      source: "nemotron",
      interpretation: composed.interpretation,
      summary,
      rhythm,
      timeOffsetSec: 0,
      tokenUsage: composed.tokenUsage,
      model: composed.model,
    };
    return NextResponse.json(result);
  } catch (e) {
    if (e instanceof BundleError || e instanceof ComposeError) {
      return NextResponse.json({ error: e.message }, { status: e.status });
    }
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Composition failed" },
      { status: 500 },
    );
  }
}
