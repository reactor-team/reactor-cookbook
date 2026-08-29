"use client";

import { useRef, useState } from "react";
import { PRESET_CLIPS, type PresetClip } from "@/app/lib/clips";
import { cn, EYEBROW, FOCUS_RING } from "./ui";

// Setup-phase picker for the "video" input source. Picking a preset or a local
// file hands a playable URL up to the workspace; the stage's input pane then
// plays it and streams its frames into the `source` track (see VideoSource).
export function VideoPicker({
  onSelect,
}: {
  onSelect: (url: string, name: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState<string | null>(null);

  function pickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setName(file.name);
    onSelect(URL.createObjectURL(file), file.name);
    // Clear the native value so re-picking the same file fires onChange.
    if (inputRef.current) inputRef.current.value = "";
  }

  function pickPreset(clip: PresetClip) {
    setName(clip.label);
    onSelect(clip.src, clip.label);
  }

  return (
    <div className="flex flex-col gap-2">
      <span className={cn(EYEBROW)}>preset clips</span>
      <div data-testid="preset-clips" className="grid grid-cols-2 gap-2">
        {PRESET_CLIPS.map((clip) => (
          <button
            key={clip.id}
            type="button"
            data-testid={`preset-clip-${clip.id}`}
            onClick={() => pickPreset(clip)}
            aria-label={clip.label}
            title={clip.label}
            className={cn(
              "rounded-md border border-zinc-700 p-1.5 transition hover:border-zinc-500",
              FOCUS_RING,
            )}
          >
            <video
              preload="metadata"
              muted
              playsInline
              src={clip.src}
              className="aspect-video w-full rounded border border-zinc-800 object-cover"
            />
          </button>
        ))}
      </div>

      <label className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-zinc-700 px-4 py-6 text-center transition hover:border-zinc-500">
        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          accept="video/*"
          onChange={pickFile}
          className="hidden"
        />
        <span className={cn(EYEBROW)}>{name ?? "choose a video clip"}</span>
        <span className="text-xs text-zinc-500">click to browse</span>
      </label>
    </div>
  );
}
