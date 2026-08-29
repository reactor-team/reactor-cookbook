import { NextResponse } from "next/server";
import { getJob, startExtraction } from "../../lib/server/extractJob";

/**
 * POST /api/extract  (multipart form, field "file")
 *   Saves the uploaded song and runs the offline extractor on it as a
 *   subprocess. Returns { jobId, songId } immediately; extraction takes
 *   minutes (GPU pipeline) and is polled via GET.
 *
 * GET /api/extract?job={jobId}
 *   Job status: { status, songId, logTail, elapsedSec, error? }.
 *   When status is "done" the bundle appears in GET /api/bundles.
 */

const MAX_UPLOAD_BYTES = 60 * 1024 * 1024;
const AUDIO_EXT = /\.(mp3|wav|flac|ogg|m4a)$/i;

export async function POST(request: Request) {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: "Expected multipart form data" }, { status: 400 });
  }
  const file = form.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Missing 'file' field" }, { status: 400 });
  }
  if (!AUDIO_EXT.test(file.name)) {
    return NextResponse.json(
      { error: "Unsupported file type — mp3/wav/flac/ogg/m4a only" },
      { status: 415 },
    );
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json({ error: "File too large (60MB max)" }, { status: 413 });
  }

  try {
    const data = Buffer.from(await file.arrayBuffer());
    const job = await startExtraction(file.name, data);
    return NextResponse.json({ jobId: job.id, songId: job.songId });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Upload failed" },
      { status: 409 },
    );
  }
}

export async function GET(request: Request) {
  const jobId = new URL(request.url).searchParams.get("job");
  if (!jobId) return NextResponse.json({ error: "Missing ?job" }, { status: 400 });
  const job = getJob(jobId);
  if (!job) return NextResponse.json({ error: "Unknown job" }, { status: 404 });
  return NextResponse.json({
    status: job.status,
    songId: job.songId,
    logTail: job.logTail,
    elapsedSec: Math.round(((job.finishedAt ?? Date.now()) - job.startedAt) / 1000),
    error: job.error,
  });
}
