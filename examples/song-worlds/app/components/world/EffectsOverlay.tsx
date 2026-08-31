"use client";

import { useEffect, useRef } from "react";
import type { FeatureFrames } from "../../lib/world/featureFrames";
import { BeatClock } from "../../lib/world/beatClock";

/**
 * Step C — live local effects. The ONLY layer that can react to the
 * audio PER FRAME (the Lingbot model itself is chunk-paced ~1s, so exact
 * per-beat activity has to live here, composited on top of the world).
 * Bloom pulses, particle bursts, a color punch, and a soft vignette.
 *
 * Transient trigger, in preference order:
 *  1. The extractor's EXACT beat grid (meta.json beats[]/downbeats[],
 *     forwarded via CompositionResult.rhythm) played by a BeatClock —
 *     pulse ON each real beat, harder on downbeats. Magnitude comes from
 *     the drums_onset_envelope curve at that instant.
 *  2. drums_onset_envelope threshold (when no trustworthy beat grid —
 *     e.g. low_confidence_rhythm — but curves exist).
 *  3. Web Audio AnalyserNode bass band (no curves at all).
 *
 * Intentionally decoupled: it never sends anything to Reactor and never
 * touches the composed event series (beyond borrowing an accent hue).
 */

interface Burst {
  x: number;
  y: number;
  angle: number;
  speed: number;
  life: number; // 1 → 0
  size: number;
}

export function EffectsOverlay({
  analyserRef,
  framesRef,
  beats,
  downbeats,
  getPlaybackTime,
  accentHue,
  active,
}: {
  analyserRef: React.RefObject<AnalyserNode | null>;
  /** Frame-synced extractor curves; null until fetched / on failure. */
  framesRef: React.RefObject<FeatureFrames | null>;
  /** Exact beat/downbeat timestamps (null when the grid isn't trusted,
   *  → fall back to curve/analyser detection). */
  beats: number[] | null;
  downbeats: number[] | null;
  /** Current playback time in seconds (extractor time base). */
  getPlaybackTime: () => number;
  /** Hue borrowed from the current score event so pulses feel native. */
  accentHue: number;
  /** Only run the analysis loop while the song is actually playing. */
  active: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hueRef = useRef(accentHue);
  hueRef.current = accentHue;
  const getTimeRef = useRef(getPlaybackTime);
  getTimeRef.current = getPlaybackTime;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let lastTime = performance.now();
    let bloom = 0; // 0..1, decays after each transient
    let punch = 0; // color punch alpha
    let emaLevel = 0; // running average for the transient threshold
    let runningMax = 1e-9; // slow-decay normalizer for raw curve units
    let lastBeatAt = 0;
    const bursts: Burst[] = [];
    const freq = new Uint8Array(1024);
    // Preferred trigger: the extractor's real beat grid. Null → fall back
    // to curve/analyser detection below. Rebuilt per play via deps.
    const beatClock =
      beats && downbeats && beats.length > 0 ? new BeatClock(beats, downbeats) : null;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    /** Returns the current 0..1 transient level from the best source. */
    const sampleLevel = (): number => {
      const frames = framesRef.current;
      if (frames && frames.has("drums_onset_envelope")) {
        // Frame-synced path: extractor curve at row floor(t * frameRate).
        const t = getTimeRef.current();
        const raw = frames.valueAt(t, "drums_onset_envelope");
        runningMax = Math.max(raw, runningMax * 0.9995);
        return runningMax > 1e-9 ? raw / runningMax : 0;
      }
      const analyser = analyserRef.current;
      if (!analyser) return 0;
      // Fallback: bass band ≈ first ~250Hz of bins (fftSize 2048 @ 44.1k).
      analyser.getByteFrequencyData(freq as Uint8Array<ArrayBuffer>);
      let bass = 0;
      const bassBins = 12;
      for (let i = 0; i < bassBins; i++) bass += freq[i];
      return bass / (bassBins * 255);
    };

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      const { width: w, height: h } = canvas;
      ctx.clearRect(0, 0, w, h);

      if (active) {
        // Magnitude (0..1) always from the best available curve/analyser.
        const level = sampleLevel();
        emaLevel = emaLevel * 0.95 + level * 0.05;
        const t = getTimeRef.current();

        // hit: 0 none, 1 beat, 2 downbeat. From the real grid if present,
        // else the curve-threshold fallback.
        let hit = 0;
        if (beatClock) {
          const crossed = beatClock.consume(t);
          hit = crossed.downbeats > 0 ? 2 : crossed.beats > 0 ? 1 : 0;
        } else if (
          level > 0.15 &&
          level > emaLevel * 1.35 &&
          now - lastBeatAt > 220 // refractory window
        ) {
          hit = 1;
        }

        if (hit) {
          lastBeatAt = now;
          const strong = hit === 2; // downbeats punch harder
          bloom = Math.min(1, (strong ? 0.7 : 0.45) + level * 0.6);
          punch = Math.min(0.55, (strong ? 0.22 : 0.14) + level * 0.4);
          const cx = w * (0.3 + Math.random() * 0.4);
          const cy = h * (0.3 + Math.random() * 0.4);
          const n = (strong ? 16 : 10) + Math.floor(level * 18);
          const spd = strong ? 340 : 260;
          for (let i = 0; i < n; i++) {
            bursts.push({
              x: cx,
              y: cy,
              angle: Math.random() * Math.PI * 2,
              speed: 60 + Math.random() * spd * Math.max(level, 0.4),
              life: 1,
              size: 1 + Math.random() * (strong ? 3.2 : 2.5),
            });
          }
        }
      }

      const hue = hueRef.current;

      // Bloom pulse — full-frame radial glow that flashes on transients.
      if (bloom > 0.01) {
        const g = ctx.createRadialGradient(
          w / 2, h / 2, 0,
          w / 2, h / 2, Math.max(w, h) * 0.7,
        );
        g.addColorStop(0, `hsla(${hue}, 90%, 70%, ${bloom * 0.28})`);
        g.addColorStop(1, "hsla(0, 0%, 100%, 0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
        bloom *= Math.exp(-dt * 6);
      }

      // Color punch — brief tinted wash over everything.
      if (punch > 0.01) {
        ctx.fillStyle = `hsla(${hue}, 100%, 60%, ${punch * 0.35})`;
        ctx.fillRect(0, 0, w, h);
        punch *= Math.exp(-dt * 8);
      }

      // Particle bursts.
      for (let i = bursts.length - 1; i >= 0; i--) {
        const b = bursts[i];
        b.life -= dt * 1.6;
        if (b.life <= 0) {
          bursts.splice(i, 1);
          continue;
        }
        b.x += Math.cos(b.angle) * b.speed * dt;
        b.y += Math.sin(b.angle) * b.speed * dt;
        b.speed *= Math.exp(-dt * 2);
        ctx.fillStyle = `hsla(${hue}, 95%, 75%, ${b.life * 0.8})`;
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.size * b.life, 0, Math.PI * 2);
        ctx.fill();
      }

      // Constant soft vignette so the stage reads as a framed world.
      const v = ctx.createRadialGradient(
        w / 2, h / 2, Math.min(w, h) * 0.45,
        w / 2, h / 2, Math.max(w, h) * 0.75,
      );
      v.addColorStop(0, "rgba(0,0,0,0)");
      v.addColorStop(1, "rgba(0,0,0,0.45)");
      ctx.fillStyle = v;
      ctx.fillRect(0, 0, w, h);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [analyserRef, framesRef, beats, downbeats, active]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}
