import "server-only";
import fsp from "node:fs/promises";
import path from "node:path";
import { SEED_IMAGES_DIR } from "./config";
import type { SeedCategory } from "../world/types";

/**
 * Seed-image catalog: the curated library that image-conditioned
 * generation draws from. Scanned once from SEED_IMAGES_DIR and cached
 * (deterministic — same folder, same catalog); everything downstream
 * (the LLM's menu, seedId validation, image serving) flows from this
 * scan, so the folder is the single source of truth.
 *
 * On-disk format (confirmed against the real folder): per seed, an
 * image file (.jpg/.jpeg/.png) plus a same-basename .md whose body
 * contains a `**One-liner:**` line and richer description sections.
 *
 * Categories are NOT stored on disk — they are inferred at scan time
 * from the markdown's keywords (deterministic keyword heuristic, party
 * terms checked before collage terms so e.g. a "digital collage" disco
 * scene stays party). Any seed's .md may pin its category explicitly
 * with a `**Category:** <category>` line, which overrides the heuristic
 * — still a folder change, not a code change.
 */

export interface SeedEntry {
  /** Filesystem-safe id = image basename without extension. */
  id: string;
  /** Image filename inside SEED_IMAGES_DIR (extension varies). */
  filename: string;
  category: SeedCategory;
  /** The md's `**One-liner:**` line (fallback: first paragraph). */
  oneLiner: string;
  /** Full markdown description — the LLM's fine-grained descriptor. */
  markdown: string;
}

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"];

export const SEED_MIME: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
};

// ── category inference ──

/** Checked first: a party scene often also carries collage/painting
 *  keywords, but dance-floor energy is the stronger mood signal. */
const PARTY_TERMS = [
  "disco",
  "party",
  "danc", // dance / dancing / dancers
  "nightclub",
  "nightlife",
  "crowd",
  "concert",
  "celebration",
  "rave",
  "festival",
  "dj",
];

const COLLAGE_IMPRESSIONIST_TERMS = [
  "collage",
  "impressionis", // impressionism / impressionist / post-impressionist
  "van gogh",
  "pixel art",
  "torn paper",
  "mixed media",
  "scrapbook",
  "impasto",
];

const CATEGORIES: SeedCategory[] = [
  "surreal-landscape",
  "party-psychedelic",
  "collage-impressionist",
];

function inferCategory(markdown: string): SeedCategory {
  const text = markdown.toLowerCase();

  // Explicit override line wins: `**Category:** party-psychedelic`
  const override = /\*\*category:\*\*\s*([a-z-]+)/.exec(text)?.[1];
  if (override && (CATEGORIES as string[]).includes(override)) {
    return override as SeedCategory;
  }

  if (PARTY_TERMS.some((t) => text.includes(t))) return "party-psychedelic";
  if (COLLAGE_IMPRESSIONIST_TERMS.some((t) => text.includes(t))) {
    return "collage-impressionist";
  }
  return "surreal-landscape";
}

function extractOneLiner(markdown: string): string {
  const m = /\*\*One-liner:\*\*\s*(.+)/.exec(markdown);
  if (m) return m[1].trim();
  // Fallback: first non-heading, non-empty line.
  for (const line of markdown.split(/\r?\n/)) {
    const t = line.trim();
    if (t && !t.startsWith("#")) return t;
  }
  return "";
}

// ── scan + cache ──

// globalThis so Next.js dev-mode module reloads don't rescan (same
// prototype-grade pattern as the extract job store). Restart the server
// to pick up seed folder changes.
const store = globalThis as unknown as {
  __seedCatalog?: Promise<SeedEntry[]>;
};

async function scanCatalog(): Promise<SeedEntry[]> {
  let names: string[];
  try {
    names = await fsp.readdir(SEED_IMAGES_DIR);
  } catch {
    return []; // Missing folder → empty catalog; callers decide severity.
  }

  const byBasename = new Map<string, { image?: string; md?: string }>();
  for (const name of names) {
    const ext = path.extname(name).toLowerCase();
    const base = name.slice(0, -ext.length);
    const entry = byBasename.get(base) ?? {};
    if (IMAGE_EXTENSIONS.includes(ext)) entry.image = name;
    else if (ext === ".md") entry.md = name;
    byBasename.set(base, entry);
  }

  const seeds: SeedEntry[] = [];
  for (const [base, files] of byBasename) {
    // A seed is the PAIR — an image without a description (or vice
    // versa) is skipped, loudly, rather than half-loaded.
    if (!files.image || !files.md) {
      console.warn(
        `[seeds] skipping "${base}": missing ${files.image ? ".md description" : "image file"}`,
      );
      continue;
    }
    const markdown = await fsp.readFile(
      path.join(SEED_IMAGES_DIR, files.md),
      "utf-8",
    );
    seeds.push({
      id: base,
      filename: files.image,
      category: inferCategory(markdown),
      oneLiner: extractOneLiner(markdown),
      markdown,
    });
  }

  seeds.sort((a, b) => a.id.localeCompare(b.id));
  const counts = CATEGORIES.map(
    (c) => `${c}=${seeds.filter((s) => s.category === c).length}`,
  ).join(" ");
  console.info(`[seeds] catalog: ${seeds.length} seeds (${counts})`);
  return seeds;
}

export function loadSeedCatalog(): Promise<SeedEntry[]> {
  return (store.__seedCatalog ??= scanCatalog());
}

export async function getSeed(id: string): Promise<SeedEntry | undefined> {
  const catalog = await loadSeedCatalog();
  return catalog.find((s) => s.id === id);
}

// ── seedId validation (compose-time trust boundary helper) ──

/**
 * Resolve whatever seedId the LLM returned to a real catalog entry —
 * mirroring how timestamps snap to anchors. Exact id wins; then a
 * normalized match (case/separator slips); then unique substring
 * containment (truncations); then the previous event's seed (scene
 * continuity beats a random jump); finally the first catalog seed.
 * Every repair is logged.
 */
export function resolveSeedId(
  requested: unknown,
  catalog: SeedEntry[],
  previousSeedId: string | null,
): string {
  const fallback = previousSeedId ?? catalog[0].id;
  if (typeof requested !== "string" || !requested.trim()) {
    console.warn(`[seeds] event missing seedId — using "${fallback}"`);
    return fallback;
  }
  const raw = requested.trim();
  if (catalog.some((s) => s.id === raw)) return raw;

  const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const wanted = normalize(raw);
  const normalized = catalog.find((s) => normalize(s.id) === wanted);
  if (normalized) {
    console.warn(`[seeds] snapped seedId "${raw}" → "${normalized.id}"`);
    return normalized.id;
  }

  const containing = catalog.filter(
    (s) => normalize(s.id).includes(wanted) || wanted.includes(normalize(s.id)),
  );
  if (containing.length === 1) {
    console.warn(`[seeds] snapped seedId "${raw}" → "${containing[0].id}"`);
    return containing[0].id;
  }

  console.warn(`[seeds] unknown seedId "${raw}" — using "${fallback}"`);
  return fallback;
}
