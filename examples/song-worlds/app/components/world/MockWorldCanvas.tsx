"use client";

import { useEffect, useRef } from "react";
import { visualsFromPrompt } from "../../lib/world/promptVisuals";
import type { MockVisualParams, TimelineEntry } from "../../lib/world/types";

/**
 * Mock-mode stand-in for the Reactor video stream: the active event's
 * SEED IMAGE rendered as the world backdrop (what the real model would
 * be conditioned on), overlaid with a procedural particle field +
 * gradient blobs, all driven by the SAME pre-composed score the real
 * model would receive. When the score executor "sends" a scene in mock
 * mode, `activeEntry` changes and this canvas cuts (instant) or morphs
 * (crossfade) to that entry's seed image + derived palette/motion — so
 * the seed-selection logic is testable end-to-end without a live
 * session.
 *
 * `lookRef` is the mouse-look camera (yaw/pitch). It only offsets the
 * parallax layers — identical in spirit to how the real video plane is
 * panned — and never influences the world content.
 *
 * REAL-API SWAP POINT: in live mode this component simply isn't
 * rendered; <WorldStage> shows the Reactor <video> instead.
 */

interface Particle {
  x: number;
  y: number;
  z: number; // 0..1 depth for parallax
  vx: number;
  vy: number;
  size: number;
}

const MORPH_SECONDS = 2.5;

export function MockWorldCanvas({
  activeEntry,
  lookRef,
}: {
  activeEntry: TimelineEntry | null;
  lookRef: React.RefObject<{ yaw: number; pitch: number }>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const entryRef = useRef<TimelineEntry | null>(null);
  entryRef.current = activeEntry;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let lastTime = performance.now();
    let lastPrompt = "";
    let lastSeedId = "";

    // Seed image backdrop: current + outgoing for the morph crossfade.
    // URLs are derived from seedId (same endpoint the real session
    // uploads from), so no extra plumbing is needed.
    let seedImg: HTMLImageElement | null = null;
    let prevSeedImg: HTMLImageElement | null = null;
    let seedFade = 1; // 0→1 crossfade progress toward seedImg

    const loadSeed = (seedId: string, transition: "cut" | "morph") => {
      const img = new Image();
      img.src = `/api/seeds/image?id=${encodeURIComponent(seedId)}`;
      img.onload = () => {
        prevSeedImg = transition === "morph" ? seedImg : null;
        seedImg = img;
        seedFade = transition === "cut" ? 1 : 0;
      };
      img.onerror = () => console.warn(`[mock world] seed image failed: ${seedId}`);
    };

    /** Cover-fit draw with camera parallax (deep layer — moves least). */
    const drawSeed = (
      img: HTMLImageElement,
      alpha: number,
      w: number,
      h: number,
      look: { yaw: number; pitch: number },
    ) => {
      const overscan = 1.14; // oversized plane so looking never runs out of pixels
      const scale = Math.max(w / img.width, h / img.height) * overscan;
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (w - dw) / 2 - look.yaw * 60;
      const dy = (h - dh) / 2 - look.pitch * 45;
      ctx.globalAlpha = alpha;
      ctx.drawImage(img, dx, dy, dw, dh);
      ctx.globalAlpha = 1;
    };

    const idle: MockVisualParams = {
      hueA: 230,
      hueB: 280,
      speed: 0.12,
      turbulence: 0.1,
      brightness: 0.25,
    };
    let current = { ...idle };
    let from = { ...idle };
    let target = { ...idle };
    let morphT = 1; // 1 = settled on target

    const particles: Particle[] = Array.from({ length: 170 }, () => ({
      x: Math.random(),
      y: Math.random(),
      z: 0.2 + Math.random() * 0.8,
      vx: (Math.random() - 0.5) * 0.02,
      vy: (Math.random() - 0.5) * 0.02,
      size: 0.5 + Math.random() * 2,
    }));
    // Blob phase offsets so the gradients drift independently.
    const blobs = Array.from({ length: 5 }, (_, i) => ({
      seed: i * 1.7 + 0.9,
      depth: 0.15 + (i / 5) * 0.5,
    }));

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const lerpHue = (a: number, b: number, t: number) => {
      let d = ((b - a + 540) % 360) - 180;
      return (a + d * t + 360) % 360;
    };

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      const t = now / 1000;

      // Pick up score changes: morph eases over MORPH_SECONDS, cut snaps.
      const entry = entryRef.current;
      const prompt = entry?.prompt ?? "";
      if (prompt !== lastPrompt) {
        lastPrompt = prompt;
        const next = prompt ? visualsFromPrompt(prompt) : idle;
        if (entry?.transition === "cut") {
          current = { ...next };
          morphT = 1;
        } else {
          from = { ...current };
          morphT = 0;
        }
        target = next;
      }
      // Seed backdrop follows the score the same way: cut = swap,
      // morph = crossfade. (Mirrors the real session's set_image swap.)
      const seedId = entry?.seedId ?? "";
      if (seedId && seedId !== lastSeedId) {
        lastSeedId = seedId;
        loadSeed(seedId, entry?.transition === "cut" ? "cut" : "morph");
      }
      if (seedFade < 1) seedFade = Math.min(1, seedFade + dt / MORPH_SECONDS);
      if (morphT < 1) {
        morphT = Math.min(1, morphT + dt / MORPH_SECONDS);
        const e = morphT * morphT * (3 - 2 * morphT); // smoothstep
        current = {
          hueA: lerpHue(from.hueA, target.hueA, e),
          hueB: lerpHue(from.hueB, target.hueB, e),
          speed: lerp(from.speed, target.speed, e),
          turbulence: lerp(from.turbulence, target.turbulence, e),
          brightness: lerp(from.brightness, target.brightness, e),
        };
      }

      const { width: w, height: h } = canvas;
      const look = lookRef.current ?? { yaw: 0, pitch: 0 };

      // Background: black base, then the seed image (the world's visual
      // basis), then a translucent hue wash so the prompt-derived
      // palette still tints the scene on top of the seed.
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);
      if (prevSeedImg && seedFade < 1) drawSeed(prevSeedImg, 1 - seedFade, w, h, look);
      if (seedImg) drawSeed(seedImg, seedFade, w, h, look);
      const bgLight = 4 + current.brightness * 6;
      const washAlpha = seedImg ? 0.32 : 1;
      ctx.fillStyle = `hsla(${current.hueA}, 40%, ${bgLight}%, ${washAlpha})`;
      ctx.fillRect(0, 0, w, h);

      // Gradient blobs — deep parallax layer.
      ctx.globalCompositeOperation = "lighter";
      for (const blob of blobs) {
        const drift = current.speed * 0.25;
        const bx =
          w *
          (0.5 +
            0.38 * Math.sin(t * drift + blob.seed * 2.4) +
            0.12 * Math.sin(t * drift * 2.7 * (1 + current.turbulence) + blob.seed));
        const by =
          h *
          (0.5 +
            0.34 * Math.cos(t * drift * 0.8 + blob.seed * 3.1) +
            0.1 * Math.cos(t * drift * 2.2 + blob.seed * 5));
        // Camera parallax: deeper layers move less.
        const px = bx - look.yaw * 220 * blob.depth;
        const py = by - look.pitch * 160 * blob.depth;
        const r = Math.max(w, h) * (0.22 + 0.1 * Math.sin(t * 0.3 + blob.seed));
        const hue = blob.seed % 2 < 1 ? current.hueA : current.hueB;
        const light = 30 + current.brightness * 30;
        const g = ctx.createRadialGradient(px, py, 0, px, py, r);
        g.addColorStop(0, `hsla(${hue}, 75%, ${light}%, ${0.16 + current.brightness * 0.1})`);
        g.addColorStop(1, "hsla(0, 0%, 0%, 0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
      }

      // Particle field — near parallax layer.
      for (const p of particles) {
        const jitter = current.turbulence * 0.06;
        p.vx += (Math.random() - 0.5) * jitter * dt;
        p.vy += (Math.random() - 0.5) * jitter * dt;
        const cap = 0.05 * (0.3 + current.speed);
        p.vx = Math.max(-cap, Math.min(cap, p.vx));
        p.vy = Math.max(-cap, Math.min(cap, p.vy));
        p.x = (p.x + p.vx * dt * current.speed * 8 + 1) % 1;
        p.y = (p.y + p.vy * dt * current.speed * 8 + 1) % 1;

        const sx =
          ((p.x * w - look.yaw * 380 * p.z) % (w + 40) + (w + 40)) % (w + 40) - 20;
        const sy =
          ((p.y * h - look.pitch * 260 * p.z) % (h + 40) + (h + 40)) % (h + 40) - 20;
        const hue = p.z > 0.6 ? current.hueB : current.hueA;
        const alpha = (0.25 + current.brightness * 0.5) * p.z;
        ctx.fillStyle = `hsla(${hue}, 80%, ${55 + current.brightness * 25}%, ${alpha})`;
        ctx.beginPath();
        ctx.arc(sx, sy, p.size * p.z * 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [lookRef]);

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />;
}
