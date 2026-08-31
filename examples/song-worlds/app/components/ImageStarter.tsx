"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import {
  useHelios,
  useHeliosState,
  type HeliosStateMessage,
} from "@reactor-models/helios";
import { IMAGE_SCENES, type Scene } from "../lib/prompts";

// Image-to-video inputs. There are two distinct flows here:
//
//   1. Curated scene (image + prompt known at the same time):
//        uploadFile → setConditioning({ prompt, image }) → start
//
//      `setConditioning` (Helios 0.9.0+) bundles both pieces of
//      conditioning into a single data-channel message. The model
//      validates, VAE-encodes, and commits both atomically before the
//      next command on its queue is dispatched, so the following
//      `start` can NEVER see partial state. This replaces the older
//      "setImage → wait for image_accepted → setPrompt → start" dance
//      that the example used before `set_conditioning` existed.
//
//   2. Custom upload (user types their own prompt later, in
//      PromptComposer):
//        uploadFile → setImage
//
//      We only stamp the image here; the user's prompt arrives via
//      PromptComposer's `setPrompt + start`. The single-consumer model
//      queue means there's no race in this path either — by the time
//      the human has typed a prompt and clicked Start, the upload has
//      long since been processed.
//
// The SDK lifts FileRef values out of the event params into the
// `uploads` envelope automatically, so `image: ref` reads as if it
// were a regular field.

// Setup-phase panel. Hidden once generation has started.
export function ImageStarter() {
  const { status, uploadFile, setImage, setConditioning, start } = useHelios();
  const [snapshot, setSnapshot] = useState<HeliosStateMessage | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useHeliosState((msg) => setSnapshot(msg));

  useEffect(() => {
    if (status !== "ready") setSnapshot(null);
  }, [status]);

  // Hide once we're generating — but keep rendering (in disabled form)
  // when the user is just not connected, so the page doesn't go blank
  // after disconnect.
  if (status === "ready" && snapshot?.started) return null;

  const ready = status === "ready";

  // Example image: the prompt is part of the curated scene, so we
  // launch the full thing (image + prompt + start) in one click.
  //
  // `setConditioning` is the atomic prompt+image command added in
  // Helios 0.9.0. The model handles validate → decode → VAE-encode →
  // commit as a single transaction, then emits `prompt_accepted`,
  // `image_accepted`, `conditions_ready(True, True)`, and a fresh
  // state snapshot in one go. Because both pieces of conditioning ride
  // on a single message, the following `start` can't slip past them
  // on the wire — we don't have to await any acknowledgment here.
  async function startFromExample(scene: Scene & { imageUrl: string }) {
    setBusy(scene.label);
    try {
      const blob = await fetch(scene.imageUrl).then((r) => r.blob());
      const ref = await uploadFile(blob, { name: `${scene.id}.jpg` });
      await setConditioning({ prompt: scene.initial.text, image: ref });
      await start();
    } finally {
      setBusy(null);
    }
  }

  // Custom upload: we ONLY change the image. The user types their own
  // prompt in the textarea above and clicks Start — at that point
  // PromptComposer fires `setPrompt + start` and the model picks up
  // the image we set here.
  async function uploadCustomImage(file: File) {
    setBusy(file.name);
    try {
      const ref = await uploadFile(file);
      await setImage({ image: ref });
    } finally {
      setBusy(null);
    }
  }

  const customImageSet = snapshot?.image_set === true;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        Or start from an image
      </label>

      <div className="mt-2 grid grid-cols-2 gap-2">
        {IMAGE_SCENES.map((scene) => (
          <button
            key={scene.id}
            disabled={!ready || busy !== null}
            onClick={() => startFromExample(scene)}
            className="group relative aspect-video overflow-hidden rounded-md border border-zinc-800 bg-zinc-950 text-left hover:border-brand disabled:opacity-40 disabled:hover:border-zinc-800"
          >
            <Image
              src={scene.imageUrl}
              alt={scene.label}
              fill
              sizes="160px"
              className="object-cover transition-opacity group-hover:opacity-80"
            />
            <span className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5 text-[11px] font-medium text-zinc-100">
              {scene.label}
            </span>
          </button>
        ))}
      </div>

      <label
        className={`mt-2 flex cursor-pointer items-center justify-center rounded-md border border-dashed border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-400 hover:border-brand hover:text-brand ${
          !ready || busy !== null ? "pointer-events-none opacity-40" : ""
        }`}
      >
        {busy
          ? `Uploading ${busy}…`
          : customImageSet
            ? "Replace your image"
            : "Upload your own image"}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          disabled={!ready || busy !== null}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) uploadCustomImage(file);
            e.target.value = "";
          }}
        />
      </label>

      {customImageSet && !busy && (
        <p className="mt-2 text-[11px] text-zinc-500">
          Image attached. Add a prompt above and click Start.
        </p>
      )}
    </div>
  );
}
