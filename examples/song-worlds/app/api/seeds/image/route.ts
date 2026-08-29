import fsp from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { SEED_IMAGES_DIR } from "../../../lib/server/config";
import { SEED_MIME, getSeed } from "../../../lib/server/seedCatalog";

/**
 * GET /api/seeds/image?id={seedId}
 *
 * Serves one seed image's bytes. Playback fetches this to upload the
 * blob to Reactor (uploadFile → FileRef), and mock mode renders it as
 * the section backdrop. The id is resolved through the catalog — never
 * joined into a path directly — so only real catalog entries are
 * servable (no traversal). Whole-Buffer response on purpose: seeds are
 * small, and Node streams crash the process on aborted requests (see
 * api/bundles/audio/route.ts).
 */
export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing ?id" }, { status: 400 });

  const seed = await getSeed(id);
  if (!seed) {
    return NextResponse.json({ error: `Unknown seed id: ${id}` }, { status: 404 });
  }

  try {
    const buffer = await fsp.readFile(path.join(SEED_IMAGES_DIR, seed.filename));
    const contentType =
      SEED_MIME[path.extname(seed.filename).toLowerCase()] ??
      "application/octet-stream";
    return new NextResponse(buffer as unknown as BodyInit, {
      headers: {
        "Content-Type": contentType,
        "Content-Length": String(buffer.byteLength),
        // The library is a curated folder that changes rarely; let the
        // browser cache across a session.
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Seed image read failed" },
      { status: 500 },
    );
  }
}
