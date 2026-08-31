"use client";

/**
 * Beat clock — plays back the extractor's EXACT beat/downbeat grid
 * (meta.json `beats[]` / `downbeats[]`, forwarded via CompositionResult
 * .rhythm) against playback time, so the live layer locks to the real
 * detected beats instead of re-deriving them from an onset curve.
 *
 * Two consumers:
 *  - EffectsOverlay: pulse exactly on each beat, harder on downbeats.
 *  - movement: phrase-level (downbeat) emphasis.
 *
 * `consume(t)` assumes monotonically increasing t (forward playback) and
 * reports beats crossed since the previous call — call it once per frame.
 * Call reset() on replay/seek.
 */
export class BeatClock {
  private beatIdx = 0;
  private downbeatIdx = 0;

  constructor(
    private readonly beats: number[],
    private readonly downbeats: number[],
  ) {}

  get hasGrid(): boolean {
    return this.beats.length > 0;
  }

  reset(): void {
    this.beatIdx = 0;
    this.downbeatIdx = 0;
  }

  /**
   * Advance the clock to time `t` (seconds). Returns how many beats and
   * downbeats were crossed since the last call — normally {0,0} or a
   * single {1, 0|1}. A crossed beat that is also a downbeat reports in
   * both counts, so callers can render a downbeat as a stronger accent.
   */
  consume(t: number): { beats: number; downbeats: number } {
    let beats = 0;
    let downbeats = 0;
    while (this.beatIdx < this.beats.length && this.beats[this.beatIdx] <= t) {
      beats++;
      this.beatIdx++;
    }
    while (
      this.downbeatIdx < this.downbeats.length &&
      this.downbeats[this.downbeatIdx] <= t
    ) {
      downbeats++;
      this.downbeatIdx++;
    }
    return { beats, downbeats };
  }
}
