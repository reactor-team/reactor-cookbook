"use client";

import { useEffect, useRef, useState } from "react";
import { useX2 } from "@reactor-models/x2";

// Model docs: the pointer is sampled once per generated block, so ~30 Hz is
// plenty. Sends are throttled with a trailing timer so the last position of a
// fast gesture always lands.
const SEND_INTERVAL_MS = 33;

// Drag-to-steer overlay for the edited output pane. While the user holds the
// pointer down, its position is streamed to the model as `set_pointer`
// (x, y, active) and the edited subject follows the drag; releasing sends
// active=false. Coordinates are normalized to the *output frame*, which the
// pane letterboxes (object-fit: contain) — outputAspect (from
// generation_started's width/height) is used to map from the pane's box to
// the visible video content, so 0..1 means the frame, not the letterbox.
export function PointerOverlay({
  outputAspect,
  enabled,
}: {
  /** width / height of the negotiated output, or null before generation. */
  outputAspect: number | null;
  enabled: boolean;
}) {
  const { setPointer, status } = useX2();
  const ref = useRef<HTMLDivElement>(null);
  const [dot, setDot] = useState<{ left: number; top: number } | null>(null);

  const draggingRef = useRef(false);
  const lastSentAtRef = useRef(0);
  const trailingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingRef = useRef<{ x: number; y: number } | null>(null);

  const active = enabled && status === "ready";

  useEffect(() => {
    return () => {
      if (trailingRef.current) clearTimeout(trailingRef.current);
    };
  }, []);

  // End a drag that's still active on unmount/disable so the model doesn't
  // keep steering toward a stale point.
  useEffect(() => {
    if (active) return;
    if (draggingRef.current) {
      draggingRef.current = false;
      setDot(null);
      setPointer({ active: false }).catch(() => {});
    }
  }, [active, setPointer]);

  // Map a pointer event to output-frame coordinates (0,0 = top left, clamped
  // to 0..1) plus the dot's position within the pane, accounting for the
  // letterbox object-fit: contain introduces when the pane and the output
  // frame have different aspect ratios.
  function locate(e: React.PointerEvent): {
    x: number;
    y: number;
    left: number;
    top: number;
  } | null {
    const el = ref.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;

    let cx = rect.left;
    let cy = rect.top;
    let cw = rect.width;
    let ch = rect.height;
    const aspect = outputAspect;
    if (aspect && aspect > 0) {
      const paneAspect = rect.width / rect.height;
      if (paneAspect > aspect) {
        cw = rect.height * aspect;
        cx = rect.left + (rect.width - cw) / 2;
      } else {
        ch = rect.width / aspect;
        cy = rect.top + (rect.height - ch) / 2;
      }
    }

    const x = Math.min(1, Math.max(0, (e.clientX - cx) / cw));
    const y = Math.min(1, Math.max(0, (e.clientY - cy) / ch));
    return {
      x,
      y,
      left: cx - rect.left + x * cw,
      top: cy - rect.top + y * ch,
    };
  }

  function send(x: number, y: number, isActive: boolean) {
    setPointer({ x, y, active: isActive }).catch(() => {});
  }

  function sendThrottled(x: number, y: number) {
    const now = Date.now();
    const elapsed = now - lastSentAtRef.current;
    if (elapsed >= SEND_INTERVAL_MS) {
      lastSentAtRef.current = now;
      send(x, y, true);
      return;
    }
    pendingRef.current = { x, y };
    if (trailingRef.current) return;
    trailingRef.current = setTimeout(() => {
      trailingRef.current = null;
      const p = pendingRef.current;
      pendingRef.current = null;
      if (p && draggingRef.current) {
        lastSentAtRef.current = Date.now();
        send(p.x, p.y, true);
      }
    }, SEND_INTERVAL_MS - elapsed);
  }

  function onPointerDown(e: React.PointerEvent) {
    if (!active) return;
    const p = locate(e);
    if (!p) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    draggingRef.current = true;
    setDot({ left: p.left, top: p.top });
    lastSentAtRef.current = Date.now();
    send(p.x, p.y, true);
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!draggingRef.current) return;
    const p = locate(e);
    if (!p) return;
    setDot({ left: p.left, top: p.top });
    sendThrottled(p.x, p.y);
  }

  function onPointerEnd(e: React.PointerEvent) {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (trailingRef.current) {
      clearTimeout(trailingRef.current);
      trailingRef.current = null;
    }
    pendingRef.current = null;
    const p = locate(e);
    setDot(null);
    send(p?.x ?? 0.5, p?.y ?? 0.5, false);
  }

  return (
    <div
      ref={ref}
      data-testid="pointer-overlay"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      className={
        "absolute inset-0 touch-none select-none " +
        (active ? "cursor-crosshair" : "pointer-events-none")
      }
    >
      {/* The drag marker: a press ripple (one-shot — the marker mounts on
          pointerdown and unmounts on release, so the animation runs once per
          drag), a brand-glow ring that follows the gesture, and a center dot
          on the exact coordinate being streamed. */}
      {dot && (
        <span
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: dot.left, top: dot.top }}
        >
          <span className="absolute -inset-4 animate-pointer-press rounded-full border-2 border-brand" />
          <span className="block h-7 w-7 rounded-full border-2 border-white/90 bg-brand/20 shadow-[0_0_16px_2px] shadow-brand/50" />
          <span className="absolute top-1/2 left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white" />
        </span>
      )}
    </div>
  );
}
