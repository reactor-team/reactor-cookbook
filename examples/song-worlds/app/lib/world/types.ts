/**
 * Shared types for the Song World app.
 *
 * The pipeline (one direction, three stages):
 *
 *   extractor bundle ({song_id}.meta.json + {song_id}.features.parquet)
 *     → Stage 1: deterministic reduction → MusicalSummary   (server, no LLM)
 *     → Stage 2: one Nemotron call       → WorldEvent[]     (server)
 *     → Stage 3: playback executes the fixed event series against Reactor
 *
 * Single time base: every timestamp in this file — beats, sections,
 * notable moments, world events — is seconds from the extractor's t=0.
 * If a sub-window of a song is ever played, CompositionResult.timeOffsetSec
 * carries the offset explicitly; today it is always 0 (full song).
 *
 * Nothing that happens during playback mutates the event series. The dense
 * parquet curves stay available at playback time for frame-synced live
 * effects (row = floor(t * target_frame_rate)); that path never calls
 * the LLM and never talks to Reactor.
 */

// ── Extractor bundle (mirrors the real meta.json — fields are NESTED) ──

export interface ExtractorSection {
  start: number;
  end: number;
  label: string;
}

export interface ExtractorChord {
  start: number;
  end: number;
  chord_label: string;
}

/**
 * The song's rhythm grid + harmony, forwarded raw to the client for the
 * live layer (beat-synced movement, effects pulses). Distinct from the
 * MusicalSummary digest: these are exact per-event timestamps, not
 * reduced statistics. `lowConfidenceRhythm` says whether the beat grid
 * is trustworthy — when true, consumers must fall back to energy curves.
 */
export interface RhythmData {
  bpm: number | null;
  beats: number[];
  downbeats: number[];
  lowConfidenceRhythm: boolean;
  downbeatPhaseUncertain: boolean;
  chords: ExtractorChord[];
}

export interface ExtractorMeta {
  song_id: string;
  source_file: string;
  duration_sec: number;
  sample_rate: number;
  /** Rows-per-second of the features parquet (60 by default). */
  target_frame_rate: number;
  loudness: {
    /** null = unmeasurable (digital silence / clip < 0.4s) — NOT "0 LUFS". */
    integrated_lufs: number | null;
    applied_gain_db: number;
    dynamic_range_lufs: number;
  };
  tempo: {
    /** null = no real beat grid exists (< 4 beats found). */
    bpm: number | null;
    beat_confidence: number;
    /** true → don't beat-sync; anchor on sections + energy shifts. */
    low_confidence_rhythm: boolean;
    /** true → beat grid ok for pacing, but bar/downbeat phase is suspect. */
    downbeat_phase_uncertain: boolean;
  };
  beats: number[];
  downbeats: number[];
  sections: ExtractorSection[];
  /** Template-matched chords: {start,end,chord_label} where chord_label
   *  is "{root}:{maj|min|7}" or "N" (no chord). May be absent on older
   *  bundles / when chord detection produced nothing. */
  chords?: ExtractorChord[];
  key_estimate: string;
  stems_processed: string[];
  pipeline_version: string;
  model_versions: {
    separation?: string | null;
    rhythm?: string | null;
    /** "songformer_hf" (semantic labels) | "novelty_ssm_v1" (identity
     *  labels) | null (structure stage disabled → sections is empty). */
    structure?: string | null;
    [key: string]: string | null | undefined;
  };
}

/** How to interpret section labels — passed to the LLM explicitly.
 *  - "semantic": songformer labels (intro/verse/chorus/…) carry musical
 *    function; the LLM may reason about it.
 *  - "identity": novelty labels (A/B/C…) only mean "this part resembles
 *    that part" — NOT verse/chorus. The LLM must not invent function.
 *  - "energy-derived": structure stage was disabled; sections were
 *    derived by this app from the energy curves. Identity semantics. */
export type LabelRegime = "semantic" | "identity" | "energy-derived";

// ── Stage 1 output: the compact MusicalSummary (what the LLM sees) ──

export interface SectionDigest {
  start: number;
  end: number;
  label: string;
  /** Percentile-normalized (song's p5→0, p95→1, clipped) mean energy. */
  energyMean: number;
  /** Same normalization, peak energy within the section. */
  energyPeak: number;
  /** Relative brightness 0..1 (percentile-normalized spectral centroid). */
  brightness: number;
  /** Detected onsets per second — rhythmic density/busyness. */
  onsetRate: number;
  /** Energy trajectory within the section. */
  trend: "building" | "steady" | "dropping";
  /** Which stem dominates: relative per-stem energy heuristic. */
  dominance: "drum-heavy" | "bass-driven" | "vocal-led" | "sparse" | "balanced";
}

export interface NotableMoment {
  time: number;
  kind: "energy-jump" | "energy-drop" | "loudest" | "novelty-peak";
  /** Relative salience 0..1 within this song. */
  strength: number;
}

export interface MusicalSummary {
  songId: string;
  durationSec: number;
  /** null = no beat grid — never divide by this. */
  bpm: number | null;
  beatConfidence: number;
  lowConfidenceRhythm: boolean;
  downbeatPhaseUncertain: boolean;
  keyEstimate: string;
  /** Tonic mode parsed from keyEstimate — the single strongest mood
   *  distinguisher between songs (minor reads darker/introspective, major
   *  brighter/open). "unknown" when the estimate is unparseable/atonal. */
  keyMode: "major" | "minor" | "unknown";
  /** Factual harmonic digest from the chord track: major/minor balance +
   *  the most common chords, e.g. "68% minor · Am, F, C, G". Empty when no
   *  chord data. Cross-song-comparable, unlike the normalized section
   *  metrics — lets the LLM tell a melancholic song from an anthemic one. */
  harmonicCharacter: string;
  /** ABSOLUTE spectral brightness bucket (mean centroid in Hz, fixed
   *  thresholds — NOT per-song normalized). Distinguishes a dark, bassy
   *  song from an airy, trebly one across songs, which the normalized
   *  per-section `brightness` cannot. */
  brightnessLabel: "dark" | "warm" | "bright" | "brilliant";
  /** Harmonic-vs-percussive energy balance (mean hpss_ratio). Separates a
   *  tonal, sustained, chord-driven song from a punchy, drum-driven one —
   *  one of the strongest identity signals for whether the world should
   *  flow or hit. */
  harmonicPercussiveBalance: "harmonic-dominant" | "balanced" | "percussion-driven";
  /** How fast the timbre shifts frame to frame (mean spectral flux) —
   *  the song's textural restlessness, from held/sustained to churning. */
  spectralMotion: "static" | "flowing" | "restless";
  /** Spectral width/fullness (mean spectral bandwidth) — a thin, focused
   *  mix vs a wide, dense, enveloping one. */
  spectralWidth: "narrow" | "full" | "wide";
  /** null = no loudness reference (not "measured as silent"). */
  integratedLufs: number | null;
  dynamicRangeLufs: number;
  labelRegime: LabelRegime;
  /** Raw model_versions.structure for provenance (null = disabled). */
  structureBackend: string | null;
  /** True when sections came from the energy-derived fallback. */
  sectionsDerivedFromEnergy: boolean;
  sections: SectionDigest[];
  notableMoments: NotableMoment[];
}

// ── Seed image library (image-conditioned generation) ──

/** Coarse mood tag for a seed image. Inferred at catalog-scan time from
 *  the seed's markdown keywords (overridable per-seed with an explicit
 *  `**Category:**` line in its .md). */
export type SeedCategory =
  | "surreal-landscape"
  | "party-psychedelic"
  | "collage-impressionist";

/** Client-facing reference to one seed image from the catalog — enough
 *  to display it and upload it to the world model. The full markdown
 *  descriptor stays server-side (only the LLM needs it). */
export interface SeedRef {
  id: string;
  category: SeedCategory;
  oneLiner: string;
  /** App URL serving the seed's image bytes (GET /api/seeds/image). */
  imageUrl: string;
}

// ── Stage 2 output: the timed world-event series ──

/** One pre-authored step of the world score. Produced once by the
 *  composition pass and never modified during playback. */
export interface WorldEvent {
  /** Seconds, extractor time base (same t=0 as beats/sections). */
  timestamp: number;
  /** Seed image (validated against the catalog) that anchors this
   *  stretch of the world — uploaded to the model as the reference
   *  image; the prompt complements it, never replaces it. */
  seedId: string;
  /** Scene steer sent with the seed: how the seed's world moves,
   *  lights, and evolves for this section — not a from-scratch scene. */
  prompt: string;
  transition: "cut" | "morph";
}

/** Back-compat alias — the kept playback components use this name. */
export type TimelineEntry = WorldEvent;

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface CompositionResult {
  events: WorldEvent[];
  /** The distinct seeds referenced by `events`, in order of first use —
   *  playback pre-uploads all of these before starting so mid-song seed
   *  swaps never wait on an upload. */
  seeds: SeedRef[];
  source: "nemotron";
  /** One-line emotional reading the LLM composed around. */
  interpretation: string;
  /** The Stage 1 summary the events were composed from (debug panel). */
  summary: MusicalSummary;
  /** Raw beat grid + harmony (within the clip) for the live layer —
   *  beat-synced movement and effects pulses read exact timestamps here
   *  instead of re-deriving them from curves. */
  rhythm: RhythmData;
  /** Offset of playback t=0 relative to extractor t=0. Always 0 today
   *  (full song); carried explicitly per the single-time-base rule. */
  timeOffsetSec: number;
  /** Per-song token counts — sanity check that the summary is compact. */
  tokenUsage: TokenUsage | null;
  /** Model id that produced the events. */
  model: string;
}

// ── Bundle listing (what /api/bundles returns) ──

export interface BundleListEntry {
  /** Opaque id, "{dirName}/{songId}" — pass back to the bundle APIs. */
  id: string;
  songId: string;
  /** Basename of the output dir the bundle came from. */
  dir: string;
  durationSec: number;
  bpm: number | null;
  structureBackend: string | null;
  sectionCount: number;
  /** Populated instead of the fields above when the meta failed to
   *  load/validate (e.g. unknown pipeline_version) — shown in the UI. */
  error?: string;
}

// ── Mock renderer ──

/** Parameters the mock renderer derives from an event's prompt text.
 *  Only used in mock mode — the real model consumes the prompt verbatim. */
export interface MockVisualParams {
  hueA: number;
  hueB: number;
  speed: number;
  turbulence: number;
  brightness: number;
}
