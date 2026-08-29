"use client";

import { useRef, useState } from "react";
import { PRESET_IMAGES, type PresetImage } from "@/app/lib/clips";
import { cn, EYEBROW, FOCUS_RING } from "./ui";

// Setup-phase picker for the "image" input source. Picking a preset or a
// local file hands a displayable URL up to the workspace; the stage's input
// pane then streams it as a constant 24 fps feed into the `source` track
// (see ImageSource) so the drag pointer can animate it.
export function ImagePicker({
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

  function pickPreset(preset: PresetImage) {
    setName(preset.label);
    onSelect(preset.src, preset.label);
  }

  return (
    <div className="flex flex-col gap-2">
      <span className={cn(EYEBROW)}>demo images</span>
      <div data-testid="preset-images" className="grid grid-cols-3 gap-2">
        {PRESET_IMAGES.map((preset) => (
          <button
            key={preset.id}
            type="button"
            data-testid={`preset-image-${preset.id}`}
            onClick={() => pickPreset(preset)}
            aria-label={preset.label}
            title={preset.label}
            className={cn(
              "rounded-md border border-zinc-700 p-1.5 transition hover:border-zinc-500",
              FOCUS_RING,
            )}
          >
            {/* Plain <img>: tiny demo thumbnails, no next/image pipeline needed. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preset.src}
              alt={preset.label}
              className="aspect-square w-full rounded border border-zinc-800 object-cover"
            />
          </button>
        ))}
      </div>

      <label className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-zinc-700 px-4 py-6 text-center transition hover:border-zinc-500">
        <input
          ref={inputRef}
          data-testid="image-file-input"
          type="file"
          accept="image/*"
          onChange={pickFile}
          className="hidden"
        />
        <span className={cn(EYEBROW)}>{name ?? "choose an image"}</span>
        <span className="text-xs text-zinc-500">click to browse</span>
      </label>
    </div>
  );
}
