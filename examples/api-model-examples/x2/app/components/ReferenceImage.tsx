"use client";

import { useEffect, useRef, useState } from "react";
import { useX2 } from "@reactor-models/x2";
import { PRESET_IMAGES } from "@/app/lib/clips";
import { Panel, cn, EYEBROW, FOCUS_RING, errorMessage } from "./ui";

// The runtime feeds the reference image to the model as a driving video, and
// its GStreamer pipeline scrambles the color channels when a streamed frame's
// width or height isn't a multiple of 4 — an arbitrarily-sized upload shears
// the output into desaturated streaks, worst right after a mid-run swap. Floor
// each dimension to the nearest multiple of 4 and center-crop to it before
// upload (shaves at most 3px per side, no visible change). The demo presets
// are already mod-4, so this is a no-op for them; it only rescues odd-sized
// user uploads. Best-effort: any failure returns the original file so the
// upload still goes through.
async function toMod4(file: File): Promise<Blob> {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("decode failed"));
      image.src = url;
    });
    const w = Math.max(4, Math.floor(img.naturalWidth / 4) * 4);
    const h = Math.max(4, Math.floor(img.naturalHeight / 4) * 4);
    if (w === img.naturalWidth && h === img.naturalHeight) return file;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    // Center-crop the shaved pixels rather than scaling, so nothing distorts.
    ctx.drawImage(
      img,
      (img.naturalWidth - w) / 2,
      (img.naturalHeight - h) / 2,
      w,
      h,
      0,
      0,
      w,
      h,
    );
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
    return blob ?? file;
  } catch {
    return file;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Reference-image panel for character/object insertion or swap. Picking an
// image uploads it via the SDK's presigned-URL protocol (uploadFile) and then
// sends set_reference_image with the returned FileRef; the model answers with
// reference_image_accepted (or command_error for an undecodable file).
//
// The reference conditions a run from its first block. Setting it while
// generating restarts the stream automatically (generation_stopped with
// reason `reference_image_changed`, then a fresh generation_started) — no
// manual reset needed. The parent keys this component on the reset nonce: a
// user reset clears the reference server-side, and the remount drops the
// local preview in step.
export function ReferenceImage({
  generating,
  hasReference,
  accepted,
}: {
  generating: boolean;
  hasReference: boolean;
  accepted: { width: number; height: number } | null;
}) {
  const { uploadFile, setReferenceImage, status } = useX2();
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = status === "ready";

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function apply(file: File) {
    if (busy || !ready) return;
    setBusy(true);
    setError(null);
    try {
      // Normalize to mod-4 dimensions first (see toMod4): the runtime streams
      // the reference as a driving video and scrambles the color channels on
      // any frame whose width or height isn't a multiple of 4.
      const ref = await uploadFile(await toMod4(file), { name: file.name });
      await setReferenceImage({ reference_image: ref });
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function pick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (inputRef.current) inputRef.current.value = "";
    if (file) await apply(file);
  }

  // Demo refs ship as static assets; fetch one back into a File so it goes
  // through the exact same upload path as a user-picked image.
  async function pickPreset(preset: (typeof PRESET_IMAGES)[number]) {
    if (busy || !ready) return;
    setError(null);
    try {
      const blob = await (await fetch(preset.src)).blob();
      await apply(new File([blob], `${preset.id}.jpg`, { type: "image/jpeg" }));
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <Panel label="Reference image">
      <div className="mb-2 flex flex-wrap gap-1.5">
        {PRESET_IMAGES.map((preset) => (
          <button
            key={preset.id}
            type="button"
            disabled={!ready || busy}
            onClick={() => void pickPreset(preset)}
            className={cn(
              "rounded-md border border-zinc-700 p-1 transition hover:border-zinc-500 disabled:opacity-40 disabled:pointer-events-none",
              FOCUS_RING,
            )}
            aria-label={`Use demo reference: ${preset.label}`}
            title={preset.label}
          >
            {/* Plain <img>: tiny demo thumbnails, no next/image pipeline needed. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preset.src}
              alt={preset.label}
              className="h-10 w-10 rounded object-cover"
            />
          </button>
        ))}
      </div>
      <label
        className={cn(
          "flex cursor-pointer items-center gap-3 rounded-md border border-dashed border-zinc-700 p-3 transition hover:border-zinc-500",
          (!ready || busy) && "pointer-events-none opacity-40",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => void pick(e)}
          className="hidden"
        />
        {previewUrl ? (
          // Plain <img>: the preview is a transient blob URL, not an asset
          // next/image could optimize.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Reference"
            className="h-14 w-14 shrink-0 rounded border border-zinc-800 object-cover"
          />
        ) : (
          <span className="flex h-14 w-14 shrink-0 items-center justify-center rounded border border-zinc-800 text-lg text-zinc-600">
            +
          </span>
        )}
        <span className="min-w-0">
          <span className={EYEBROW}>
            {busy
              ? "uploading…"
              : previewUrl
                ? "replace image"
                : "add an image"}
          </span>
          <span className="mt-0.5 block text-xs text-zinc-500">
            Character or object to insert / swap in.
          </span>
        </span>
      </label>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {accepted && !error && hasReference && (
        <p className="mt-2 text-xs text-zinc-500">
          Accepted ({accepted.width}×{accepted.height}).{" "}
          {generating
            ? "Conditioning the current run — replacing it restarts the stream."
            : "Will condition the next run."}
        </p>
      )}
    </Panel>
  );
}
