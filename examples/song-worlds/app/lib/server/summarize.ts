import "server-only";
import {
  FALLBACK_SECTION_MIN_LENGTH_SEC,
  FALLBACK_SECTION_SHIFT_THRESHOLD,
  NORM_HIGH_PERCENTILE,
  NORM_LOW_PERCENTILE,
  NOTABLE_MOMENT_MAX_COUNT,
  NOTABLE_MOMENT_MIN_SPACING_SEC,
} from "./config";
import type { FeatureTable } from "./bundle";
import type {
  ExtractorChord,
  ExtractorMeta,
  ExtractorSection,
  LabelRegime,
  MusicalSummary,
  NotableMoment,
  SectionDigest,
} from "../world/types";

/**
 * Stage 1 — deterministic reduction. Pure code, NO LLM.
 *
 * Reduces the extractor bundle (sparse meta + dense 60 FPS curves) to a
 * compact, page-sized MusicalSummary. The raw curves are never sent to
 * the LLM — a multi-minute song is ~16k rows × 124 columns, which blows
 * the context window and makes reasoning worse. Reduce first, LLM second.
 *
 * This module only READS the FeatureTable it is given (which itself is a
 * read-only view of the parquet); the parquet on disk is untouched and
 * its curves stay available for frame-synced playback effects.
 *
 * All timestamps produced here are seconds on the extractor's time base.
 */

/** Columns Stage 1 wants, matched by name against the parquet schema.
 *  Missing ones degrade gracefully (e.g. full_mix_novelty only exists
 *  with the novelty structure backend). */
export const SUMMARY_COLUMNS = [
  "frame_time_sec",
  "full_mix_rms",
  "full_mix_spectral_centroid",
  "full_mix_onset_envelope",
  "full_mix_novelty", // optional — novelty backend only
  "full_mix_hpss_ratio", // harmonic-vs-percussive balance
  "full_mix_spectral_flux", // timbral motion / restlessness
  "full_mix_spectral_bandwidth", // spectral width / fullness
  "drums_rms",
  "bass_rms",
  "vocals_spectral_flatness", // tonal stems have no rms; see stem energy below
  "drums_onset_envelope",
];

// ── small numeric helpers ──

function percentile(sorted: Float64Array, p: number): number {
  if (sorted.length === 0) return 0;
  const idx = ((sorted.length - 1) * p) / 100;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/** Percentile normalizer: song's p5 → 0, p95 → 1, clipped outside.
 *  NOT naive min-max — one loud drop must not flatten every other
 *  section into the bottom of the range. */
function makePercentileNorm(values: Float64Array): (v: number) => number {
  const sorted = Float64Array.from(values).sort();
  const lo = percentile(sorted, NORM_LOW_PERCENTILE);
  const hi = percentile(sorted, NORM_HIGH_PERCENTILE);
  const span = hi - lo;
  if (span <= 1e-12) return () => 0; // flat curve (e.g. silence)
  return (v: number) => Math.min(1, Math.max(0, (v - lo) / span));
}

function mean(arr: Float64Array, from: number, to: number): number {
  let s = 0;
  let n = 0;
  for (let i = from; i < to && i < arr.length; i++) {
    s += arr[i];
    n++;
  }
  return n > 0 ? s / n : 0;
}

function max(arr: Float64Array, from: number, to: number): number {
  let m = -Infinity;
  for (let i = from; i < to && i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m === -Infinity ? 0 : m;
}

const round = (v: number, places = 3) => {
  const f = 10 ** places;
  return Math.round(v * f) / f;
};

// ── energy-derived fallback sections (structure stage disabled) ──

/**
 * When model_versions.structure is null the sections list is empty; we
 * derive a coarse structural map from the full-mix energy curve instead
 * of producing an empty result: segment wherever a smoothed energy level
 * sustains a shift past a threshold, with a minimum section length.
 * Labels are identity-style ("S1", "S2", …) — they carry no musical
 * function, and the LLM is told so via labelRegime "energy-derived".
 */
export function deriveSectionsFromEnergy(
  rms: Float64Array,
  frameRate: number,
  durationSec: number,
): ExtractorSection[] {
  if (rms.length === 0) return [{ start: 0, end: durationSec, label: "S1" }];
  const norm = makePercentileNorm(rms);

  // Smooth to ~1s resolution so per-beat wiggle doesn't split sections.
  const win = Math.max(1, Math.round(frameRate));
  const nBlocks = Math.ceil(rms.length / win);
  const blockLevel = new Float64Array(nBlocks);
  for (let b = 0; b < nBlocks; b++) {
    blockLevel[b] = norm(mean(rms, b * win, (b + 1) * win));
  }

  const minBlocks = Math.max(1, Math.round(FALLBACK_SECTION_MIN_LENGTH_SEC));
  const boundaries: number[] = [0];
  let refLevel = blockLevel[0];
  let lastBoundary = 0;
  for (let b = 1; b < nBlocks; b++) {
    if (
      b - lastBoundary >= minBlocks &&
      Math.abs(blockLevel[b] - refLevel) >= FALLBACK_SECTION_SHIFT_THRESHOLD
    ) {
      // Require the shift to sustain for 2 blocks (~2s) — a single
      // transient block is an accent, not a section change.
      const next = b + 1 < nBlocks ? blockLevel[b + 1] : blockLevel[b];
      if (Math.abs(next - refLevel) >= FALLBACK_SECTION_SHIFT_THRESHOLD) {
        boundaries.push(b);
        lastBoundary = b;
        refLevel = blockLevel[b];
        continue;
      }
    }
    // Drift the reference slowly so gradual builds eventually trigger.
    refLevel = refLevel * 0.98 + blockLevel[b] * 0.02;
  }

  const sections: ExtractorSection[] = [];
  for (let i = 0; i < boundaries.length; i++) {
    const startBlock = boundaries[i];
    const endBlock = i + 1 < boundaries.length ? boundaries[i + 1] : nBlocks;
    const start = round((startBlock * win) / frameRate);
    const end = round(Math.min((endBlock * win) / frameRate, durationSec));
    // Degenerate tail (rounding can leave a near-zero-length section at
    // the end) — extend the previous section instead.
    if (end - start < 1 && sections.length > 0) {
      sections[sections.length - 1].end = end;
      continue;
    }
    sections.push({ start, end, label: `S${sections.length + 1}` });
  }
  if (sections.length > 0) sections[sections.length - 1].end = round(durationSec);
  return sections;
}

// ── notable moments ──

interface Candidate extends NotableMoment {
  score: number;
}

/** Non-maximum suppression by time: strongest first, drop anything
 *  within minSpacing of an already-kept moment, so an attack and its
 *  secondary peak can't yield near-duplicate anchors. */
function suppress(cands: Candidate[], minSpacing: number, cap: number): NotableMoment[] {
  const kept: Candidate[] = [];
  for (const c of [...cands].sort((a, b) => b.score - a.score)) {
    if (kept.some((k) => Math.abs(k.time - c.time) < minSpacing)) continue;
    kept.push(c);
    if (kept.length >= cap) break;
  }
  return kept
    .sort((a, b) => a.time - b.time)
    .map(({ time, kind, strength }) => ({
      time: round(time, 2),
      kind,
      strength: round(strength, 2),
    }));
}

function findNotableMoments(
  features: FeatureTable,
  frameRate: number,
): NotableMoment[] {
  const rms = features.columns.get("full_mix_rms");
  if (!rms || rms.length === 0) return [];
  const norm = makePercentileNorm(rms);
  const cands: Candidate[] = [];

  // Energy jumps/drops: delta of ~1s-block means, normalized.
  const win = Math.max(1, Math.round(frameRate / 2)); // 0.5s blocks
  const nBlocks = Math.floor(rms.length / win);
  const level = new Float64Array(nBlocks);
  for (let b = 0; b < nBlocks; b++) level[b] = norm(mean(rms, b * win, (b + 1) * win));
  for (let b = 2; b < nBlocks; b++) {
    const delta = level[b] - level[b - 2]; // change over ~1s
    const time = (b * win) / frameRate;
    if (delta > 0.25) {
      cands.push({ time, kind: "energy-jump", strength: Math.min(1, delta), score: delta });
    } else if (delta < -0.25) {
      cands.push({ time, kind: "energy-drop", strength: Math.min(1, -delta), score: -delta * 0.9 });
    }
  }

  // Loudest moment of the song (skipped for all-silent audio, where
  // "loudest" is meaningless).
  let peakIdx = 0;
  for (let i = 1; i < rms.length; i++) if (rms[i] > rms[peakIdx]) peakIdx = i;
  if (rms[peakIdx] > 1e-9) {
    cands.push({
      time: peakIdx / frameRate,
      kind: "loudest",
      strength: 1,
      score: 1.2, // always survives suppression against nearby jumps
    });
  }

  // full_mix_novelty peaks (novelty backend only) — normalized 0–1
  // boundary-strength curve; its peaks are natural scene-change points.
  const novelty = features.columns.get("full_mix_novelty");
  if (novelty && novelty.length > 0) {
    for (let i = 2; i < novelty.length - 2; i++) {
      const v = novelty[i];
      if (v < 0.5) continue;
      if (v >= novelty[i - 1] && v >= novelty[i + 1] && v > novelty[i - 2] && v > novelty[i + 2]) {
        cands.push({
          time: i / frameRate,
          kind: "novelty-peak",
          strength: Math.min(1, v),
          score: v * 1.1, // high-value driver when present
        });
      }
    }
  }

  return suppress(cands, NOTABLE_MOMENT_MIN_SPACING_SEC, NOTABLE_MOMENT_MAX_COUNT);
}

// ── per-section digests ──

function digestSections(
  sections: ExtractorSection[],
  features: FeatureTable,
  frameRate: number,
): SectionDigest[] {
  const rms = features.columns.get("full_mix_rms") ?? new Float64Array(0);
  const centroid = features.columns.get("full_mix_spectral_centroid") ?? new Float64Array(0);
  const onset = features.columns.get("full_mix_onset_envelope") ?? new Float64Array(0);
  const drums = features.columns.get("drums_rms") ?? new Float64Array(0);
  const bass = features.columns.get("bass_rms") ?? new Float64Array(0);
  // Tonal stems carry no rms column; vocal presence is inferred from
  // spectral flatness (voiced content → low flatness). See dominance.
  const vocalFlatness = features.columns.get("vocals_spectral_flatness") ?? new Float64Array(0);

  const normEnergy = makePercentileNorm(rms);
  const normBright = makePercentileNorm(centroid);
  const normOnset = makePercentileNorm(onset);

  // Song-wide stem baselines for relative dominance.
  const songDrums = mean(drums, 0, drums.length);
  const songBass = mean(bass, 0, bass.length);
  const songRms = mean(rms, 0, rms.length);

  // Onset threshold for a crude onsets-per-second estimate: count local
  // maxima of the onset envelope above its p75.
  const sortedOnset = Float64Array.from(onset).sort();
  const onsetThreshold = percentile(sortedOnset, 75);

  return sections.map((sec) => {
    const from = Math.max(0, Math.floor(sec.start * frameRate));
    const to = Math.min(rms.length, Math.ceil(sec.end * frameRate));
    const lenSec = Math.max(1e-6, sec.end - sec.start);

    const energyMean = normEnergy(mean(rms, from, to));
    const energyPeak = normEnergy(max(rms, from, to));
    const brightness = normBright(mean(centroid, from, to));

    // Onset rate: local peaks above threshold, per second.
    let onsetCount = 0;
    for (let i = Math.max(1, from); i < to - 1 && i < onset.length - 1; i++) {
      if (onset[i] > onsetThreshold && onset[i] >= onset[i - 1] && onset[i] > onset[i + 1]) {
        onsetCount++;
      }
    }
    const onsetRate = onsetCount / lenSec;

    // Trend: start-vs-end thirds of the section's energy.
    const third = Math.max(1, Math.floor((to - from) / 3));
    const startE = normEnergy(mean(rms, from, from + third));
    const endE = normEnergy(mean(rms, to - third, to));
    const trend: SectionDigest["trend"] =
      endE - startE > 0.12 ? "building" : startE - endE > 0.12 ? "dropping" : "steady";

    // Dominance: per-stem energy in this section relative to the song.
    const secDrums = mean(drums, from, to);
    const secBass = mean(bass, from, to);
    const secRms = mean(rms, from, to);
    const secVocalFlat = mean(vocalFlatness, from, to);
    let dominance: SectionDigest["dominance"] = "balanced";
    if (songRms <= 1e-9 || secRms / songRms < 0.35) {
      // Includes the all-silent song: no energy anywhere → sparse.
      dominance = "sparse";
    } else {
      const drumsRel = songDrums > 1e-9 ? secDrums / songDrums : 0;
      const bassRel = songBass > 1e-9 ? secBass / songBass : 0;
      // Low spectral flatness in the vocal stem = tonal/voiced content.
      const vocalPresent = vocalFlatness.length > 0 && secVocalFlat < 0.1;
      if (drumsRel > 1.25 && drumsRel > bassRel) dominance = "drum-heavy";
      else if (bassRel > 1.25) dominance = "bass-driven";
      else if (vocalPresent) dominance = "vocal-led";
    }

    return {
      start: round(sec.start, 2),
      end: round(sec.end, 2),
      label: sec.label,
      energyMean: round(energyMean),
      energyPeak: round(energyPeak),
      brightness: round(brightness),
      onsetRate: round(onsetRate, 1),
      trend,
      dominance,
    };
  });
}

// ── label regime (fact #3) ──

function detectLabelRegime(meta: ExtractorMeta): {
  regime: LabelRegime;
  derived: boolean;
} {
  const backend = meta.model_versions.structure ?? null;
  if (backend === null || meta.sections.length === 0) {
    return { regime: "energy-derived", derived: true };
  }
  if (backend.startsWith("songformer")) return { regime: "semantic", derived: false };
  // novelty_ssm_v1 and anything unknown: safest to treat labels as
  // identity-only — never invent musical function from a label.
  return { regime: "identity", derived: false };
}

// ── cross-song identity (ABSOLUTE, not per-song normalized) ──

/** Tonic mode from the key estimate. Handles "C major"/"A minor" words,
 *  "{root}:min|maj", and trailing-m note forms ("Am", "F#m"). */
function deriveKeyMode(keyEstimate: string): "major" | "minor" | "unknown" {
  const k = keyEstimate.toLowerCase().trim();
  if (/\bminor\b|:min|(^|\s)min\b/.test(k)) return "minor";
  if (/\bmajor\b|:maj|(^|\s)maj\b/.test(k)) return "major";
  if (/^[a-g][#b]?m\b/.test(k)) return "minor"; // "am", "f#m"
  if (/^[a-g][#b]?\b/.test(k)) return "major"; // bare note ⇒ major
  return "unknown";
}

/** "A:min" → "Am", "F:maj" → "F", "G:7" → "G7". */
function formatChord(label: string): string {
  const [root, qual] = label.split(":");
  if (!qual || qual === "maj") return root;
  if (qual === "min") return `${root}m`;
  return `${root}${qual}`;
}

/**
 * Factual harmonic digest from the chord track (within the clip window):
 * minor-vs-major balance by sounding duration + the most-present chords.
 * This is cross-song-comparable — a melancholic minor-key song and an
 * anthemic major-key song produce visibly different strings, which the
 * per-song-normalized section metrics never could.
 */
function deriveHarmonicCharacter(
  chords: ExtractorChord[] | undefined,
  clipSeconds: number,
): string {
  if (!chords || chords.length === 0) return "";
  const durByLabel = new Map<string, number>();
  let majDur = 0;
  let minDur = 0;
  for (const c of chords) {
    if (c.start >= clipSeconds) continue;
    if (!c.chord_label || c.chord_label === "N") continue;
    const dur = Math.max(0, Math.min(c.end, clipSeconds) - c.start);
    if (dur <= 0) continue;
    durByLabel.set(c.chord_label, (durByLabel.get(c.chord_label) ?? 0) + dur);
    const qual = c.chord_label.split(":")[1] ?? "";
    if (qual === "min") minDur += dur;
    else if (qual === "maj") majDur += dur;
  }
  if (durByLabel.size === 0) return "";
  const top = [...durByLabel.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([label]) => formatChord(label));
  const majMin = majDur + minDur;
  const prefix = majMin > 0 ? `${Math.round((minDur / majMin) * 100)}% minor · ` : "";
  return `${prefix}${top.join(", ")}`;
}

/** ABSOLUTE brightness bucket from the mean spectral centroid (Hz).
 *  Fixed thresholds (typical music centroid ~1–4 kHz) so the label is
 *  comparable across songs, unlike the normalized per-section brightness. */
function deriveBrightnessLabel(
  centroid: Float64Array,
): "dark" | "warm" | "bright" | "brilliant" {
  const m = mean(centroid, 0, centroid.length);
  if (m < 1200) return "dark";
  if (m < 2200) return "warm";
  if (m < 3200) return "bright";
  return "brilliant";
}

/**
 * Harmonic-vs-percussive balance from mean hpss_ratio (fraction of energy
 * that is harmonic; higher = more tonal/sustained, lower = more drum-
 * driven). Thresholds are heuristic (a tonal ballad sits ~0.86, a beat-
 * driven track drops well below) — tune against a few contrasting songs.
 */
function deriveHarmonicPercussiveBalance(
  hpss: Float64Array,
): "harmonic-dominant" | "balanced" | "percussion-driven" {
  if (hpss.length === 0) return "balanced"; // column absent → neutral
  const m = mean(hpss, 0, hpss.length);
  if (m >= 0.82) return "harmonic-dominant";
  if (m >= 0.62) return "balanced";
  return "percussion-driven";
}

/** Timbral restlessness from mean spectral flux (rate of spectral change
 *  frame to frame). Heuristic thresholds around the observed p50 (~0.10). */
function deriveSpectralMotion(
  flux: Float64Array,
): "static" | "flowing" | "restless" {
  if (flux.length === 0) return "flowing"; // column absent → neutral
  const m = mean(flux, 0, flux.length);
  if (m < 0.06) return "static";
  if (m < 0.13) return "flowing";
  return "restless";
}

/** Spectral width/fullness from mean spectral bandwidth (Hz). Heuristic
 *  thresholds around the observed range (~2400–3700 Hz). */
function deriveSpectralWidth(
  bandwidth: Float64Array,
): "narrow" | "full" | "wide" {
  if (bandwidth.length === 0) return "full"; // column absent → neutral
  const m = mean(bandwidth, 0, bandwidth.length);
  if (m < 2600) return "narrow";
  if (m < 3400) return "full";
  return "wide";
}

// ── entry point ──

/**
 * `clipSeconds` limits the summary to the song's first N seconds (the
 * caller must have loaded `features` over the same window so the
 * percentile normalization sees only in-window values). The window
 * starts at t=0, so all timestamps remain on the extractor time base.
 */
export function buildMusicalSummary(
  meta: ExtractorMeta,
  features: FeatureTable,
  clipSeconds?: number,
): MusicalSummary {
  const frameRate = meta.target_frame_rate;
  const { regime, derived } = detectLabelRegime(meta);
  const durationSec =
    clipSeconds !== undefined
      ? Math.min(meta.duration_sec, clipSeconds)
      : meta.duration_sec;

  const sections: ExtractorSection[] = (
    derived
      ? deriveSectionsFromEnergy(
          features.columns.get("full_mix_rms") ?? new Float64Array(0),
          frameRate,
          durationSec,
        )
      : meta.sections
  )
    // Clip the structural map to the window: drop sections past the
    // cap, clamp the one straddling it.
    .filter((s) => s.start < durationSec)
    .map((s) => (s.end > durationSec ? { ...s, end: durationSec } : s));

  return {
    songId: meta.song_id,
    durationSec: round(durationSec, 2),
    bpm: meta.tempo.bpm, // may be null — preserved, never coerced
    beatConfidence: round(meta.tempo.beat_confidence, 2),
    lowConfidenceRhythm: meta.tempo.low_confidence_rhythm,
    downbeatPhaseUncertain: meta.tempo.downbeat_phase_uncertain,
    keyEstimate: meta.key_estimate,
    keyMode: deriveKeyMode(meta.key_estimate),
    harmonicCharacter: deriveHarmonicCharacter(meta.chords, durationSec),
    brightnessLabel: deriveBrightnessLabel(
      features.columns.get("full_mix_spectral_centroid") ?? new Float64Array(0),
    ),
    harmonicPercussiveBalance: deriveHarmonicPercussiveBalance(
      features.columns.get("full_mix_hpss_ratio") ?? new Float64Array(0),
    ),
    spectralMotion: deriveSpectralMotion(
      features.columns.get("full_mix_spectral_flux") ?? new Float64Array(0),
    ),
    spectralWidth: deriveSpectralWidth(
      features.columns.get("full_mix_spectral_bandwidth") ?? new Float64Array(0),
    ),
    integratedLufs: meta.loudness.integrated_lufs, // may be null
    dynamicRangeLufs: round(meta.loudness.dynamic_range_lufs, 2),
    labelRegime: regime,
    structureBackend: meta.model_versions.structure ?? null,
    sectionsDerivedFromEnergy: derived,
    sections: digestSections(sections, features, frameRate),
    notableMoments: findNotableMoments(features, frameRate),
  };
}
