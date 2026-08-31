import fsp from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { BundleError, loadMeta, resolveAudioPath, resolveBundle } from "../../../lib/server/bundle";

/**
 * GET /api/bundles/audio?id={dir}/{song_id}
 *
 * Serves the bundle's source audio file (meta.source_file, located via
 * the configured audio search dirs) for the <audio> element, with Range
 * support so seeking works. Playback time in this file IS the extractor
 * time base — the extractor analyzed the same full file from t=0.
 *
 * Reads the requested byte range into a Buffer rather than piping a
 * fs.ReadStream through a web ReadableStream: these clips are capped at
 * CLIP_SECONDS (a few MB at most), and a Node Readable has no clean
 * cancellation story when the <audio> element aborts a request mid-seek —
 * both a raw `stream as unknown as ReadableStream` cast and Node's own
 * `Readable.toWeb()` throw "Controller is already closed" from inside the
 * stream lifecycle (outside any try/catch) when that happens, which
 * crashes the whole dev server process. A Buffer has no controller state
 * to corrupt, so an aborted client connection just discards the response.
 */

const MIME: Record<string, string> = {
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".m4a": "audio/mp4",
};

export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing ?id" }, { status: 400 });

  try {
    const bundle = await resolveBundle(id);
    const meta = await loadMeta(bundle);
    const audioPath = await resolveAudioPath(meta);
    if (!audioPath) {
      return NextResponse.json(
        {
          error:
            `Source audio "${meta.source_file}" not found in the configured ` +
            `audio dirs (EXTRACTOR_AUDIO_DIRS)`,
        },
        { status: 404 },
      );
    }

    const stat = await fsp.stat(audioPath);
    const contentType = MIME[path.extname(audioPath).toLowerCase()] ?? "application/octet-stream";
    const range = request.headers.get("range");

    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      const start = m?.[1] ? parseInt(m[1], 10) : 0;
      const end = m?.[2] ? Math.min(parseInt(m[2], 10), stat.size - 1) : stat.size - 1;
      if (start >= stat.size || start > end) {
        return new NextResponse(null, {
          status: 416,
          headers: { "Content-Range": `bytes */${stat.size}` },
        });
      }
      const length = end - start + 1;
      const handle = await fsp.open(audioPath, "r");
      let chunk: Buffer;
      try {
        chunk = Buffer.alloc(length);
        await handle.read(chunk, 0, length, start);
      } finally {
        await handle.close();
      }
      return new NextResponse(chunk as unknown as BodyInit, {
        status: 206,
        headers: {
          "Content-Type": contentType,
          "Content-Length": String(length),
          "Content-Range": `bytes ${start}-${end}/${stat.size}`,
          "Accept-Ranges": "bytes",
        },
      });
    }

    const fileBuffer = await fsp.readFile(audioPath);
    return new NextResponse(fileBuffer as unknown as BodyInit, {
      headers: {
        "Content-Type": contentType,
        "Content-Length": String(stat.size),
        "Accept-Ranges": "bytes",
      },
    });
  } catch (e) {
    if (e instanceof BundleError) {
      return NextResponse.json({ error: e.message }, { status: e.status });
    }
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Audio failed" },
      { status: 500 },
    );
  }
}
