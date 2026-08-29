"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FRAME_HEIGHT, FRAME_WIDTH } from "@/app/lib/types";
import { Button } from "./ui";

// Crop step for avatar uploads.
//
// Why this exists: the model fits whatever you upload to its 640×352 canvas,
// which is a wide 16:9-ish frame. Hand it a normal portrait photo — taller
// than it is wide — and the fit decapitates the subject. Since the avatar
// image is the one thing that defines the face for the whole take, getting it
// wrong is not a subtle degradation, it is a headless video.
//
// So the picked file goes through this modal and only the framed pixels are
// uploaded. The frame is the largest 640:352 region that fits the image: tall
// images drag vertically, extra-wide ones horizontally. The default framing
// comes from the browser's FaceDetector where available, and otherwise from
// top-center, since faces sit in the top third of portraits.

const ASPECT = FRAME_WIDTH / FRAME_HEIGHT;
const JPEG_QUALITY = 0.92;

// FaceDetector is a Chrome-only experimental API, absent from lib.dom.
interface DetectedFaceBox {
  boundingBox: { x: number; y: number; width: number; height: number };
}
interface FaceDetectorLike {
  detect(source: CanvasImageSource): Promise<DetectedFaceBox[]>;
}
declare global {
  interface Window {
    FaceDetector?: new (options?: {
      maxDetectedFaces?: number;
      fastMode?: boolean;
    }) => FaceDetectorLike;
  }
}

interface Frame {
  /** Frame size and offset, all in natural image pixels. */
  width: number;
  height: number;
  x: number;
  y: number;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

/** Largest 640:352 frame that fits, positioned at top-center. */
function fallbackFrame(imgW: number, imgH: number): Frame {
  const wide = imgW / imgH > ASPECT;
  const width = wide ? imgH * ASPECT : imgW;
  const height = wide ? imgH : imgW / ASPECT;
  // Faces live in the top third of portraits: aim the frame's center there.
  const y = clamp(imgH / 3 - height / 2, 0, imgH - height);
  const x = clamp((imgW - width) / 2, 0, imgW - width);
  return { width, height, x, y };
}

async function detectFaceFrame(img: HTMLImageElement): Promise<Frame | null> {
  if (typeof window === "undefined" || !window.FaceDetector) return null;
  try {
    const detector = new window.FaceDetector({
      maxDetectedFaces: 1,
      fastMode: true,
    });
    const faces = await detector.detect(img);
    const box = faces[0]?.boundingBox;
    if (!box) return null;
    const base = fallbackFrame(img.naturalWidth, img.naturalHeight);
    return {
      ...base,
      x: clamp(
        box.x + box.width / 2 - base.width / 2,
        0,
        img.naturalWidth - base.width,
      ),
      y: clamp(
        box.y + box.height / 2 - base.height / 2,
        0,
        img.naturalHeight - base.height,
      ),
    };
  } catch {
    return null; // detector present but failed — fallback wins
  }
}

export function CropModal({
  file,
  onConfirm,
  onCancel,
}: {
  file: File;
  onConfirm: (blob: Blob, name: string) => void;
  onCancel: () => void;
}) {
  // Created in an effect (not useState) so StrictMode's simulated unmount
  // revokes and recreates the URL as a pair — a state-held URL would survive
  // the remount already revoked, and the image would never load.
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [displayScale, setDisplayScale] = useState(1);
  const [busy, setBusy] = useState(false);
  const dragState = useRef<{
    startX: number;
    startY: number;
    frame: Frame;
  } | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onImageLoad = useCallback(async () => {
    const img = imgRef.current;
    if (!img) return;
    setDisplayScale(img.clientWidth / img.naturalWidth);
    const detected = await detectFaceFrame(img);
    setFrame(detected ?? fallbackFrame(img.naturalWidth, img.naturalHeight));
  }, []);

  // Keep the display scale honest if the modal resizes. The crop rectangle is
  // positioned in display pixels but stored in natural ones, so a stale scale
  // slides it off the face it was framing.
  //
  // Keyed on `objectUrl` rather than mounting once, because the <img> is only
  // rendered once the URL exists: on the first pass `imgRef.current` is still
  // null and there is nothing to observe. The image element is also the right
  // thing to watch rather than the container — `max-h-[60vh] max-w-full` means
  // its size tracks the viewport, and it is its own clientWidth the scale is
  // derived from.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const observer = new ResizeObserver(() => {
      if (img.naturalWidth > 0) {
        setDisplayScale(img.clientWidth / img.naturalWidth);
      }
    });
    observer.observe(img);
    return () => observer.disconnect();
  }, [objectUrl]);

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (!frame) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragState.current = { startX: e.clientX, startY: e.clientY, frame };
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const drag = dragState.current;
    const img = imgRef.current;
    if (!drag || !img) return;
    const dx = (e.clientX - drag.startX) / displayScale;
    const dy = (e.clientY - drag.startY) / displayScale;
    setFrame({
      ...drag.frame,
      x: clamp(drag.frame.x + dx, 0, img.naturalWidth - drag.frame.width),
      y: clamp(drag.frame.y + dy, 0, img.naturalHeight - drag.frame.height),
    });
  }

  function onPointerUp() {
    dragState.current = null;
  }

  async function confirm() {
    const img = imgRef.current;
    if (!img || !frame || busy) return;
    setBusy(true);
    try {
      // Crop at native resolution, upscaling only when the framed region is
      // smaller than the model's canvas.
      const scaleUp = Math.max(
        1,
        FRAME_WIDTH / frame.width,
        FRAME_HEIGHT / frame.height,
      );
      const outW = Math.round(frame.width * scaleUp);
      const outH = Math.round(frame.height * scaleUp);
      const canvas = document.createElement("canvas");
      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas 2D context unavailable");
      ctx.drawImage(
        img,
        frame.x,
        frame.y,
        frame.width,
        frame.height,
        0,
        0,
        outW,
        outH,
      );
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
      );
      if (!blob) throw new Error("Cropping produced no image");
      const base = file.name.replace(/\.[^.]+$/, "") || "avatar";
      onConfirm(blob, `${base}-crop.jpg`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6">
      <div className="flex max-w-3xl flex-col gap-3 rounded-xl border border-edge bg-surface p-4">
        <div className="flex items-baseline justify-between gap-6">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">
            Frame the avatar
          </p>
          <p className="font-mono text-[10px] uppercase tracking-wider text-brand/70">
            {FRAME_WIDTH}×{FRAME_HEIGHT} — drag to frame
          </p>
        </div>

        <div className="relative overflow-hidden rounded-lg bg-black">
          {objectUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              ref={imgRef}
              src={objectUrl}
              alt="Portrait to crop"
              onLoad={() => void onImageLoad()}
              className="block max-h-[60vh] w-auto max-w-full select-none"
              draggable={false}
            />
          )}
          {frame && (
            <div
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              className="absolute cursor-grab touch-none border-2 border-brand active:cursor-grabbing"
              style={{
                left: frame.x * displayScale,
                top: frame.y * displayScale,
                width: frame.width * displayScale,
                height: frame.height * displayScale,
                boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.65)",
              }}
            >
              <span className="absolute -top-5 left-0 font-mono text-[9px] uppercase tracking-wider text-brand-light">
                take frame
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2">
          <Button variant="secondary" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => void confirm()}
            disabled={!frame || busy}
          >
            {busy ? "Cropping…" : "Use this framing"}
          </Button>
        </div>
      </div>
    </div>
  );
}
