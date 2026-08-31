import { NextResponse } from "next/server";
import { listBundles } from "../../lib/server/bundle";

/**
 * GET /api/bundles — list every extractor bundle found in the configured
 * output dirs (EXTRACTOR_OUTPUT_DIRS). Bundles whose meta fails to load
 * or validate (e.g. unknown pipeline_version) appear with an `error`
 * field instead of being silently hidden.
 */
export async function GET() {
  try {
    return NextResponse.json({ bundles: await listBundles() });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to list bundles" },
      { status: 500 },
    );
  }
}
