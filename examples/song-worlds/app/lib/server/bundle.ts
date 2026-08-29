import "server-only";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import {
  asyncBufferFromFile,
  parquetMetadataAsync,
  parquetReadObjects,
  parquetSchema,
} from "hyparquet";
import {
  AUDIO_SEARCH_DIRS,
  COMPATIBLE_PIPELINE_MAJOR_VERSIONS,
  EXTRACTOR_OUTPUT_DIRS,
} from "./config";
import type { BundleListEntry, ExtractorMeta } from "../world/types";

/**
 * Extractor-bundle loader. The app does no audio analysis of its own —
 * it consumes `{song_id}.meta.json` + `{song_id}.features.parquet`
 * produced by the offline extractor (D:\PROJECTS\Research\audio-extractor).
 *
 * Reads straight from disk today; the configurable EXTRACTOR_OUTPUT_DIRS
 * seam is the single place to swap in an HTTP fetch later.
 *
 * Correctness rules enforced here:
 *  - pipeline_version is checked against a known-compatible major range;
 *    unknown versions fail loudly instead of being silently misread.
 *  - Parquet columns are discovered from the parquet's own schema and
 *    matched BY NAME ({stem}_{feature}); indices are never hardcoded.
 *  - meta.json nested nulls (loudness.integrated_lufs, tempo.bpm,
 *    model_versions.structure) are preserved as null, never coerced.
 */

export class BundleError extends Error {
  constructor(
    message: string,
    readonly status: number = 400,
  ) {
    super(message);
  }
}

export interface FeatureTable {
  /** Column name → dense per-frame values (one row per frame). */
  columns: Map<string, Float64Array>;
  rowCount: number;
  /** All column names present in the parquet schema. */
  allColumnNames: string[];
  frameRate: number;
}

interface ResolvedBundle {
  id: string;
  songId: string;
  dir: string;
  metaPath: string;
  parquetPath: string;
}

// ── Discovery ──

async function scanDir(dir: string): Promise<ResolvedBundle[]> {
  let names: string[];
  try {
    names = await fsp.readdir(dir);
  } catch {
    return []; // configured dir may not exist — fine
  }
  const out: ResolvedBundle[] = [];
  for (const name of names) {
    if (!name.endsWith(".meta.json")) continue;
    const songId = name.slice(0, -".meta.json".length);
    const parquetPath = path.join(dir, `${songId}.features.parquet`);
    if (!fs.existsSync(parquetPath)) continue; // meta without features → not a bundle
    out.push({
      id: `${path.basename(dir)}/${songId}`,
      songId,
      dir,
      metaPath: path.join(dir, name),
      parquetPath,
    });
  }
  return out;
}

async function resolveAll(): Promise<ResolvedBundle[]> {
  const groups = await Promise.all(EXTRACTOR_OUTPUT_DIRS.map(scanDir));
  const seen = new Set<string>();
  const out: ResolvedBundle[] = [];
  for (const b of groups.flat()) {
    // First configured dir wins on id collision (same dir basename + song).
    if (seen.has(b.id)) continue;
    seen.add(b.id);
    out.push(b);
  }
  return out;
}

export async function resolveBundle(id: string): Promise<ResolvedBundle> {
  const all = await resolveAll();
  const found = all.find((b) => b.id === id);
  if (!found) throw new BundleError(`Unknown bundle id: ${id}`, 404);
  return found;
}

// ── meta.json ──

function validateMeta(raw: unknown, metaPath: string): ExtractorMeta {
  if (typeof raw !== "object" || raw === null) {
    throw new BundleError(`${metaPath}: meta.json is not an object`, 422);
  }
  const m = raw as Record<string, unknown>;

  // Fail loudly on unrecognized pipeline major versions (fact #7).
  const version = m.pipeline_version;
  if (typeof version !== "string") {
    throw new BundleError(`${metaPath}: missing pipeline_version`, 422);
  }
  const major = Number(version.split(".")[0]);
  if (!COMPATIBLE_PIPELINE_MAJOR_VERSIONS.includes(major)) {
    throw new BundleError(
      `${metaPath}: unsupported pipeline_version "${version}" ` +
        `(this app understands major version(s) ` +
        `${COMPATIBLE_PIPELINE_MAJOR_VERSIONS.join(", ")}). ` +
        `Refusing to silently misread a newer format.`,
      422,
    );
  }

  // Structural checks on the NESTED shape — these fields must exist as
  // objects; their inner values may legitimately be null.
  const loudness = m.loudness as Record<string, unknown> | undefined;
  const tempo = m.tempo as Record<string, unknown> | undefined;
  const modelVersions = m.model_versions as Record<string, unknown> | undefined;
  if (
    typeof m.song_id !== "string" ||
    typeof m.duration_sec !== "number" ||
    typeof m.target_frame_rate !== "number" ||
    typeof loudness !== "object" ||
    loudness === null ||
    typeof tempo !== "object" ||
    tempo === null ||
    typeof modelVersions !== "object" ||
    modelVersions === null ||
    !Array.isArray(m.beats) ||
    !Array.isArray(m.sections)
  ) {
    throw new BundleError(
      `${metaPath}: meta.json does not match the extractor schema ` +
        `(expected nested loudness/tempo/model_versions objects)`,
      422,
    );
  }

  // Null tri-state checks: null is a VALID value for these — only reject
  // types that are neither number nor null.
  const lufs = loudness.integrated_lufs;
  if (lufs !== null && typeof lufs !== "number") {
    throw new BundleError(`${metaPath}: loudness.integrated_lufs must be number|null`, 422);
  }
  const bpm = tempo.bpm;
  if (bpm !== null && typeof bpm !== "number") {
    throw new BundleError(`${metaPath}: tempo.bpm must be number|null`, 422);
  }
  const structure = modelVersions.structure ?? null;
  if (structure !== null && typeof structure !== "string") {
    throw new BundleError(`${metaPath}: model_versions.structure must be string|null`, 422);
  }

  return raw as ExtractorMeta;
}

export async function loadMeta(bundle: ResolvedBundle): Promise<ExtractorMeta> {
  let text: string;
  try {
    text = await fsp.readFile(bundle.metaPath, "utf-8");
  } catch (e) {
    throw new BundleError(
      `Failed to read ${bundle.metaPath}: ${e instanceof Error ? e.message : e}`,
      500,
    );
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new BundleError(`${bundle.metaPath}: invalid JSON`, 422);
  }
  return validateMeta(raw, bundle.metaPath);
}

// ── features.parquet ──

/**
 * Read the requested columns by name. Column names come from the
 * parquet's own schema, not from meta.json (fact #5): callers pass
 * `want` name lists and get back only the names that actually exist.
 * This is a pure read — the parquet on disk is never mutated, so the
 * raw curves stay fully available for playback.
 *
 * `maxSeconds` truncates the read to the first floor(maxSeconds ×
 * frameRate) rows (the CLIP_SECONDS window, starting at t=0).
 */
export async function loadFeatures(
  bundle: ResolvedBundle,
  want: string[],
  frameRate: number,
  maxSeconds?: number,
): Promise<FeatureTable> {
  const file = await asyncBufferFromFile(bundle.parquetPath);
  const metadata = await parquetMetadataAsync(file);
  const schema = parquetSchema(metadata);
  const allColumnNames = schema.children.map((c) => c.element.name);
  const available = new Set(allColumnNames);

  const columnsToRead = want.filter((w) => available.has(w));
  if (columnsToRead.length === 0) {
    throw new BundleError(
      `${bundle.parquetPath}: none of the requested columns exist. ` +
        `Requested: ${want.join(", ")}`,
      422,
    );
  }

  const totalRows = Number(metadata.num_rows);
  const rowEnd =
    maxSeconds !== undefined
      ? Math.min(totalRows, Math.max(1, Math.floor(maxSeconds * frameRate)))
      : totalRows;

  const rows = (await parquetReadObjects({
    file,
    columns: columnsToRead,
    rowStart: 0,
    rowEnd,
  })) as Record<string, number>[];

  const columns = new Map<string, Float64Array>();
  for (const name of columnsToRead) {
    const arr = new Float64Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const v = rows[i][name];
      arr[i] = Number.isFinite(v) ? v : 0;
    }
    columns.set(name, arr);
  }

  return { columns, rowCount: rows.length, allColumnNames, frameRate };
}

// ── audio ──

/** Locate the bundle's source audio file for playback (meta stores only
 *  a basename; search the configured audio dirs). */
export async function resolveAudioPath(meta: ExtractorMeta): Promise<string | null> {
  for (const dir of AUDIO_SEARCH_DIRS) {
    const candidate = path.join(dir, meta.source_file);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

// ── listing ──

export async function listBundles(): Promise<BundleListEntry[]> {
  const bundles = await resolveAll();
  const entries: BundleListEntry[] = [];
  for (const b of bundles) {
    try {
      const meta = await loadMeta(b);
      entries.push({
        id: b.id,
        songId: b.songId,
        dir: path.basename(b.dir),
        durationSec: meta.duration_sec,
        bpm: meta.tempo.bpm,
        structureBackend: meta.model_versions.structure ?? null,
        sectionCount: meta.sections.length,
      });
    } catch (e) {
      entries.push({
        id: b.id,
        songId: b.songId,
        dir: path.basename(b.dir),
        durationSec: 0,
        bpm: null,
        structureBackend: null,
        sectionCount: 0,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }
  entries.sort((a, b) => a.id.localeCompare(b.id));
  return entries;
}
