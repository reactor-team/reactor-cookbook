"use client";

import { useEffect, useState } from "react";
import {
  useLingbot,
  useLingbotState,
  type LingbotStateMessage,
} from "@reactor-models/lingbot";

// Setup-phase panel. Lets the user assemble a custom session: upload
// their own image, type their own prompt, then click Start.
//
// The chain is the same as in <ScenePicker> — `setImage` is the slow
// step (the model decodes the image inside its handler), and awaiting
// it waits for exactly that: `image_accepted` is the command's
// correlated reply. Without the wait the first chunk can render
// before the image conditioning is applied and visibly flicker.
//
// `start` only succeeds once both a prompt AND an image are set.
// We surface the readiness state with the disabled-state of the
// button (`has_prompt` AND `has_image` on the snapshot) so the user
// knows what's still missing without having to read error messages.
export function CustomStart() {
  const { status, uploadFile, setImage, setPrompt, start } = useLingbot();
  const [snapshot, setSnapshot] = useState<LingbotStateMessage | null>(null);
  const [text, setText] = useState("");
  const [imageBusy, setImageBusy] = useState<string | null>(null);

  useLingbotState((msg) => setSnapshot(msg));

  useEffect(() => {
    if (status !== "ready") {
      setSnapshot(null);
      setImageBusy(null);
    }
  }, [status]);

  if (status === "ready" && snapshot?.started) return null;

  const ready = status === "ready";
  const hasPrompt = snapshot?.has_prompt === true || text.trim().length > 0;
  const hasImage = snapshot?.has_image === true;

  async function uploadCustomImage(file: File) {
    setImageBusy(file.name);
    try {
      const ref = await uploadFile(file);
      // Resolves once the model has decoded the image; the snapshot that
      // follows flips `has_image`, which is what enables Start.
      await setImage({ image: ref });
    } finally {
      setImageBusy(null);
    }
  }

  async function startCustom() {
    if (!ready || !hasImage || !text.trim()) return;
    await setPrompt({ prompt: text.trim() });
    await start();
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        Or roll your own
      </label>

      <label
        className={`mt-2 flex cursor-pointer items-center justify-center rounded-md border border-dashed border-zinc-700 bg-zinc-950 px-3 py-2 text-xs hover:border-brand hover:text-brand ${
          !ready || imageBusy !== null
            ? "pointer-events-none opacity-40 text-zinc-400"
            : hasImage
              ? "text-active"
              : "text-zinc-400"
        }`}
      >
        {imageBusy
          ? `Uploading ${imageBusy}…`
          : hasImage
            ? "Image attached · click to replace"
            : "Upload a reference image"}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          disabled={!ready || imageBusy !== null}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) uploadCustomImage(file);
            e.target.value = "";
          }}
        />
      </label>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Describe the scene the image should become. Subject, environment, camera framing — full paragraph."
        disabled={!ready}
        rows={4}
        className="mt-2 w-full resize-none rounded-md border border-zinc-800 bg-zinc-950 p-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-brand focus:outline-none disabled:opacity-40"
      />

      <button
        disabled={!ready || !hasImage || !text.trim() || imageBusy !== null}
        onClick={startCustom}
        className="mt-2 w-full rounded-md bg-brand px-3 py-2 text-sm font-medium text-brand-fg hover:opacity-90 disabled:opacity-40"
      >
        {!hasImage
          ? "Upload an image first"
          : !hasPrompt
            ? "Add a prompt"
            : "Start generating"}
      </button>
    </div>
  );
}
