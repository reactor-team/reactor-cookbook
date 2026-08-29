import path from "node:path";

/**
 * Server-only configuration for the extractor-bundle → Nemotron pipeline.
 * Everything here is overridable via env so the disk paths can become an
 * HTTP fetch later without rearchitecting (swap the bundle loader, keep
 * the config surface).
 *
 * This module must never be imported from client components — it reads
 * process.env server secrets' metadata and local filesystem paths.
 */

/**
 * Root of the offline extractor project (BeatLens —
 * https://github.com/DhruvaMyakeri/BeatLens or wherever you cloned it).
 * Default assumes it's cloned as a sibling of this app's folder — see
 * the README's "Extracting your own songs" section. Point EXTRACTOR_ROOT
 * at your actual clone via .env if it lives somewhere else.
 */
export const EXTRACTOR_ROOT = process.env.EXTRACTOR_ROOT ?? "../beatlens";

/**
 * Directories scanned for `{song_id}.meta.json` + `{song_id}.features.parquet`
 * pairs — full bundles produced by `beatlens song.mp3 --out <dir>`.
 * Semicolon-separated in env; relative entries resolve against
 * EXTRACTOR_ROOT. `data/bundles` (app-local, gitignored) is always
 * scanned too, so you can drop a bundle straight into this repo without
 * touching env config at all — see the README.
 */
export const EXTRACTOR_OUTPUT_DIRS: string[] = (
  process.env.EXTRACTOR_OUTPUT_DIRS ?? ["outputs"].join(";")
)
  .split(";")
  .map((d) => d.trim())
  .filter(Boolean)
  .map((d) => (path.isAbsolute(d) ? d : path.join(EXTRACTOR_ROOT, d)))
  .concat([path.join(process.cwd(), "data", "bundles")]);

/**
 * Where to look for the bundle's `source_file` audio for playback (the
 * meta stores only a basename). Searched in order; relative entries
 * resolve against EXTRACTOR_ROOT. `data/songs` (app-local, gitignored —
 * see the README) is always searched too, so a song and the bundle
 * BeatLens produced from it can live side by side in this repo without
 * any env config.
 */
export const AUDIO_SEARCH_DIRS: string[] = (
  process.env.EXTRACTOR_AUDIO_DIRS ?? [".", "uploads"].join(";")
)
  .split(";")
  .map((d) => d.trim())
  .filter(Boolean)
  .map((d) => (path.isAbsolute(d) ? d : path.join(EXTRACTOR_ROOT, d)))
  .concat([path.join(process.cwd(), "data", "songs")]);

// ── Upload → extraction (runs the offline extractor as a subprocess) ──

/** Where uploaded songs are saved before extraction. */
export const UPLOAD_DIR =
  process.env.UPLOAD_DIR ?? path.join(EXTRACTOR_ROOT, "uploads");

/**
 * The extractor's own `beatlens` console-script entrypoint (installed into
 * its venv via `pip install -e .` — see beatlens/pyproject.toml
 * `[project.scripts]`). Invoked directly as `beatlens <audio> --out <dir>`,
 * matching the README; we do NOT run it as a server (`beatlens-serve`) and
 * we do NOT invoke it via `python -m afe.pipeline` (stale module path from
 * before the package was renamed to `beatlens`).
 */
export const EXTRACTOR_BIN =
  process.env.EXTRACTOR_BIN ??
  path.join(EXTRACTOR_ROOT, ".venv", "Scripts", "beatlens.exe");

/** Extractor config passed via --config (relative to EXTRACTOR_ROOT). */
export const EXTRACTOR_CONFIG =
  process.env.EXTRACTOR_CONFIG ?? path.join("config", "default.yaml");

/** Which output dir extraction writes to (must be in EXTRACTOR_OUTPUT_DIRS
 *  so the finished bundle shows up in the picker). */
export const EXTRACTOR_OUT_DIR_NAME = process.env.EXTRACTOR_OUT_DIR ?? "outputs";

/**
 * Pipeline versions this app knows how to read. Fail loudly on anything
 * else rather than silently misreading a future format (fact #7).
 */
export const COMPATIBLE_PIPELINE_MAJOR_VERSIONS = [0];

/**
 * Only the first CLIP_SECONDS of every song are used — summarized,
 * composed and played. The window always starts at the extractor's t=0,
 * so CompositionResult.timeOffsetSec stays 0 and every timestamp keeps
 * the extractor time base. (If a non-zero window start is ever added,
 * carry it through timeOffsetSec explicitly.)
 */
export const CLIP_SECONDS = Number(process.env.CLIP_SECONDS ?? 90);

// ── Seed image library (image-conditioned generation) ──

/**
 * Folder holding the curated seed images: per seed, an image file
 * (.jpg/.png) plus a same-basename .md description. The catalog is
 * scanned from this folder — never hardcode seed ids/filenames in logic,
 * so adding/removing seeds is a folder change, not a code change.
 */
export const SEED_IMAGES_DIR =
  process.env.SEED_IMAGES_DIR ?? path.join(process.cwd(), "seed_images");

// ── Stage 1 (deterministic reduction) knobs ──

/** Minimum spacing between notable moments (non-maximum suppression). */
export const NOTABLE_MOMENT_MIN_SPACING_SEC = Number(
  process.env.NOTABLE_MOMENT_MIN_SPACING_SEC ?? 2,
);

/** Cap on the notable-moments list so the summary stays page-sized. */
export const NOTABLE_MOMENT_MAX_COUNT = Number(
  process.env.NOTABLE_MOMENT_MAX_COUNT ?? 12,
);

/** Percentiles that map to 0/1 for energy/brightness normalization —
 *  NOT naive min-max, so one loud drop can't flatten every section. */
export const NORM_LOW_PERCENTILE = 5;
export const NORM_HIGH_PERCENTILE = 95;

/** Energy-derived fallback sectioning (structure stage disabled). */
export const FALLBACK_SECTION_MIN_LENGTH_SEC = 8;
export const FALLBACK_SECTION_SHIFT_THRESHOLD = 0.18;

// ── Stage 2 (Nemotron) config ──

export const NEMOTRON_BASE_URL =
  process.env.NEMOTRON_BASE_URL ?? "https://integrate.api.nvidia.com/v1";

/**
 * Model id from build.nvidia.com — a config value, never hardcoded in
 * logic. Confirm against the account's available models if calls 404.
 */
export const NEMOTRON_MODEL =
  process.env.NEMOTRON_MODEL ?? "nvidia/nemotron-3-ultra-550b-a55b";

/** NVIDIA's recommended sampling for Nemotron reasoning models. */
export const NEMOTRON_TEMPERATURE = Number(
  process.env.NEMOTRON_TEMPERATURE ?? 1.0,
);
export const NEMOTRON_TOP_P = Number(process.env.NEMOTRON_TOP_P ?? 0.95);
export const NEMOTRON_MAX_TOKENS = Number(
  process.env.NEMOTRON_MAX_TOKENS ?? 16384,
);
/** Cap on the reasoning trace so it can't starve the final JSON answer. */
export const NEMOTRON_REASONING_BUDGET = Number(
  process.env.NEMOTRON_REASONING_BUDGET ?? 8192,
);

/** Reads the key at call time; NEVER export the value to client code. */
export function getNemotronApiKey(): string | undefined {
  return process.env.NVIDIA_NEMO_KEY;
}
