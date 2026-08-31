import "server-only";
import ffmpegPath from "ffmpeg-static";
import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import {
  CLIP_SECONDS,
  EXTRACTOR_BIN,
  EXTRACTOR_CONFIG,
  EXTRACTOR_OUT_DIR_NAME,
  EXTRACTOR_ROOT,
  UPLOAD_DIR,
} from "./config";

/**
 * Upload → extraction jobs. The app still does NO audio analysis of its
 * own: an uploaded song is trimmed to CLIP_SECONDS (everything past that
 * is never summarized, composed, or played, so there's no reason to make
 * the GPU pipeline chew through the rest of the song), saved to
 * UPLOAD_DIR, and the offline extractor (D:\PROJECTS\Research\audio-extractor,
 * package name `beatlens`) is run on the trimmed file as a subprocess via
 * its `beatlens` console-script entrypoint — not as a server
 * (`beatlens-serve`), since this all runs locally on the same machine.
 * When the pipeline finishes, its bundle lands in the configured output
 * dir and the normal bundle loader picks it up — exactly as if it had
 * been extracted by hand.
 *
 * Extraction is GPU-heavy (stem separation + structure model) and takes
 * minutes per song; jobs run one at a time and are polled by the UI.
 *
 * Job state lives on globalThis so Next.js dev-mode module reloads
 * don't lose running jobs. This is a prototype-grade in-memory store —
 * jobs don't survive a server restart (the subprocess would be orphaned
 * anyway; re-upload to retry).
 */

export interface ExtractJob {
  id: string;
  songId: string;
  fileName: string;
  status: "running" | "done" | "error";
  startedAt: number;
  finishedAt?: number;
  /** Last few lines of pipeline output, for the UI. */
  logTail: string[];
  error?: string;
}

const store = globalThis as unknown as {
  __extractJobs?: Map<string, ExtractJob>;
  __extractRunning?: boolean;
};
const jobs = (store.__extractJobs ??= new Map<string, ExtractJob>());

export function getJob(id: string): ExtractJob | undefined {
  return jobs.get(id);
}

/**
 * Filesystem-safe slug, mirroring the extractor's `make_song_id()`
 * (schema.py — lowercase, [^a-z0-9]+ -> "_", strip leading/trailing "_").
 * That function re-derives song_id from whatever filename we hand it, with
 * no length cap. We truncate to keep filenames sane, but truncation can
 * expose a NEW trailing "_" that wasn't there before the cut — so the
 * trailing-underscore strip must run again AFTER truncating, or our
 * songId (used to save the file and to look up the extractor's output)
 * disagrees with the song_id the extractor computes from that same
 * filename, and the app never finds the bundle it just produced.
 */
export function slugify(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return (
    slug
      .slice(0, 60)
      .replace(/^_+|_+$/g, "") || "song"
  );
}

const LOG_TAIL_LINES = 15;

/**
 * Trims `srcPath` down to the first CLIP_SECONDS via ffmpeg (re-encoded to
 * match the output extension, so mismatched containers/codecs on upload
 * never confuse the extractor). Only ever shrinks — clips already shorter
 * than CLIP_SECONDS pass through ffmpeg unchanged in length.
 */
function trimToClip(srcPath: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      ffmpegPath as string,
      ["-y", "-i", srcPath, "-t", String(CLIP_SECONDS), destPath],
      { windowsHide: true },
    );
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => reject(new Error(`Failed to start ffmpeg: ${err.message}`)));
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg trim failed (code ${code}): ${stderr.slice(-500)}`));
    });
  });
}

export async function startExtraction(
  originalName: string,
  data: Buffer,
): Promise<ExtractJob> {
  if (store.__extractRunning) {
    throw new Error(
      "An extraction is already running — wait for it to finish (the pipeline is GPU-heavy).",
    );
  }
  if (!fs.existsSync(EXTRACTOR_BIN)) {
    throw new Error(
      `Extractor binary not found at ${EXTRACTOR_BIN} — is the extractor project set up (pip install -e . in its venv)?`,
    );
  }

  const ext = path.extname(originalName).toLowerCase() || ".mp3";
  const songId = slugify(path.basename(originalName, path.extname(originalName)));
  const savedName = `${songId}${ext}`;

  await fsp.mkdir(UPLOAD_DIR, { recursive: true });
  const rawPath = path.join(UPLOAD_DIR, `${songId}.raw${ext}`);
  const savedPath = path.join(UPLOAD_DIR, savedName);
  await fsp.writeFile(rawPath, data);
  try {
    // Only the first CLIP_SECONDS of any song are ever summarized, composed,
    // or played (see config.ts) — so that's all that should go through the
    // GPU-heavy extractor too.
    await trimToClip(rawPath, savedPath);
  } finally {
    await fsp.unlink(rawPath).catch(() => {});
  }

  const job: ExtractJob = {
    id: `${songId}-${Date.now().toString(36)}`,
    songId,
    fileName: savedName,
    status: "running",
    startedAt: Date.now(),
    logTail: [],
  };
  jobs.set(job.id, job);
  store.__extractRunning = true;

  const outDir = path.isAbsolute(EXTRACTOR_OUT_DIR_NAME)
    ? EXTRACTOR_OUT_DIR_NAME
    : path.join(EXTRACTOR_ROOT, EXTRACTOR_OUT_DIR_NAME);

  const child = spawn(
    EXTRACTOR_BIN,
    [savedPath, "--config", EXTRACTOR_CONFIG, "--out", outDir],
    { cwd: EXTRACTOR_ROOT, windowsHide: true },
  );

  const pushLog = (chunk: Buffer) => {
    for (const line of chunk.toString().split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      job.logTail.push(trimmed);
      if (job.logTail.length > LOG_TAIL_LINES) job.logTail.shift();
    }
  };
  child.stdout.on("data", pushLog);
  child.stderr.on("data", pushLog);

  child.on("error", (err) => {
    job.status = "error";
    job.error = `Failed to start extractor: ${err.message}`;
    job.finishedAt = Date.now();
    store.__extractRunning = false;
  });

  child.on("close", (code) => {
    job.finishedAt = Date.now();
    store.__extractRunning = false;
    const metaPath = path.join(outDir, `${songId}.meta.json`);
    if (code === 0 && fs.existsSync(metaPath)) {
      job.status = "done";
    } else {
      job.status = "error";
      job.error =
        code === 0
          ? `Pipeline exited cleanly but ${songId}.meta.json was not produced`
          : `Extractor exited with code ${code}`;
    }
    console.info(
      `[extract] ${songId}: ${job.status} in ${Math.round((job.finishedAt - job.startedAt) / 1000)}s`,
    );
  });

  return job;
}
