"use client";

// The sidebar's ways in while browsing:
//   • a custom prompt — compose your own world, choose Adventure or Directing;
//   • an existing world id — attach a world you built earlier (instant).
// Both resolve to a WorldIntent the session then runs.

import { useState } from "react";
import { MAX_FIRST_FRAME_IMAGE_BYTES } from "@reactor-models/happy-oyster";
import type { WorldIntent } from "@/lib/worlds";
import { Button } from "@/components/ui/button";
import { SectionLabel } from "./ui";

export function CustomCompose({
  onIntent,
}: {
  onIntent: (intent: WorldIntent) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<1 | 2>(1);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  // Mode-specific creation knobs. perspective/resolution carry the model's
  // documented defaults, so they always ride the payload harmlessly; layout
  // and narrative have no server default, so "auto" means omit and let the
  // model choose (matching a build that never set them).
  const [perspective, setPerspective] = useState<
    "third_person" | "first_person"
  >("third_person");
  const [resolution, setResolution] = useState<"720p" | "480p">("720p");
  const [layout, setLayout] = useState<"auto" | "Stable" | "Fast">("auto");
  const [narrative, setNarrative] = useState<
    "auto" | "Normal" | "Calm" | "Dramatic"
  >("auto");

  const build = () => {
    const text = prompt.trim();
    if (!text) return;
    const firstFrameImage = imageFile ?? undefined;
    onIntent({
      kind: "create",
      mode: mode === 2 ? "directing" : "adventure",
      title: "Your world",
      params:
        mode === 2
          ? {
              prompt: text,
              firstFrameImage,
              resolution,
              ...(layout !== "auto" ? { layout } : {}),
              ...(narrative !== "auto" ? { narrative } : {}),
            }
          : { prompt: text, firstFrameImage, perspective },
    });
  };

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <SectionLabel>Compose your own</SectionLabel>
      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        rows={4}
        maxLength={2000}
        placeholder="Describe a world… a paragraph with explicit setting, mood, and camera framing works best."
        className="w-full resize-none rounded-md border border-white/10 bg-black/30 px-3 py-2 font-mono text-sm text-white/85 outline-none transition placeholder:text-white/25 focus:border-white/30 focus:ring-2 focus:ring-primary/20"
      />
      {imageFile ? (
        <div className="flex items-center justify-between gap-2 rounded-md border border-white/10 bg-black/30 px-3 py-2">
          <span className="min-w-0 truncate font-mono text-xs text-white/70">
            {imageFile.name}
          </span>
          <button
            onClick={() => setImageFile(null)}
            className="shrink-0 text-xs text-white/40 transition hover:text-white/80"
          >
            Remove
          </button>
        </div>
      ) : (
        <label className="flex cursor-pointer items-center justify-center rounded-md border border-dashed border-white/15 bg-black/20 px-3 py-2 text-xs text-white/40 transition hover:border-white/30 hover:text-white/70">
          Optional first-frame image
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              event.target.value = "";
              if (!file) return;
              if (file.size > MAX_FIRST_FRAME_IMAGE_BYTES) {
                setImageError("That image is over the 2MB limit.");
                return;
              }
              setImageError(null);
              setImageFile(file);
            }}
          />
        </label>
      )}
      {imageError && (
        <p className="text-[11px] text-red-300/90">{imageError}</p>
      )}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1 rounded-md border border-white/10 bg-black/20 p-0.5">
          <ModeToggle active={mode === 1} onClick={() => setMode(1)}>
            Adventure
          </ModeToggle>
          <ModeToggle active={mode === 2} onClick={() => setMode(2)}>
            Directing
          </ModeToggle>
        </div>
        <Button onClick={build} disabled={prompt.trim().length === 0}>
          Build world
        </Button>
      </div>
      {mode === 1 ? (
        <OptionGroup
          label="Perspective"
          value={perspective}
          onChange={(value) => setPerspective(value)}
          options={[
            { value: "third_person", label: "Third-person" },
            { value: "first_person", label: "First-person" },
          ]}
        />
      ) : (
        <div className="flex flex-col gap-2.5">
          <OptionGroup
            label="Resolution"
            value={resolution}
            onChange={(value) => setResolution(value)}
            options={[
              { value: "720p", label: "720p" },
              { value: "480p", label: "480p" },
            ]}
          />
          <OptionGroup
            label="Camera motion"
            value={layout}
            onChange={(value) => setLayout(value)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "Stable", label: "Stable" },
              { value: "Fast", label: "Fast" },
            ]}
          />
          <OptionGroup
            label="Narrative"
            value={narrative}
            onChange={(value) => setNarrative(value)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "Normal", label: "Normal" },
              { value: "Calm", label: "Calm" },
              { value: "Dramatic", label: "Dramatic" },
            ]}
          />
        </div>
      )}
      <p className="text-[11px] leading-relaxed text-white/30">
        {mode === 1
          ? "Adventure worlds are playable, you drive them with WASD."
          : "Directing worlds are steered with text instructions and transport."}
      </p>
    </div>
  );
}

export function AttachById({
  onIntent,
}: {
  onIntent: (intent: WorldIntent) => void;
}) {
  const [id, setId] = useState("");
  // A world only attaches through its own experience's model, so the attach
  // has to connect to the matching mode — pick it here.
  const [mode, setMode] = useState<1 | 2>(1);
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <SectionLabel>Return to an existing world</SectionLabel>
      <p className="text-[11px] leading-relaxed text-white/30">
        Worlds are permanent. Paste an <code>encrypted_world_id</code> you saved
        from an earlier build to jump straight back in, no build wait — and pick
        the experience it belongs to.
      </p>
      <div className="flex items-center gap-1 self-start rounded-md border border-white/10 bg-black/20 p-0.5">
        <ModeToggle active={mode === 1} onClick={() => setMode(1)}>
          Adventure
        </ModeToggle>
        <ModeToggle active={mode === 2} onClick={() => setMode(2)}>
          Directing
        </ModeToggle>
      </div>
      <div className="flex gap-1.5">
        <input
          value={id}
          onChange={(event) => setId(event.target.value)}
          placeholder="encrypted_world_id"
          className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/30 px-3 py-2 font-mono text-xs text-white/85 outline-none transition placeholder:text-white/25 focus:border-white/30"
        />
        <Button
          onClick={() =>
            onIntent({
              kind: "attach",
              mode: mode === 2 ? "directing" : "adventure",
              encryptedWorldId: id.trim(),
              title: "Attached world",
            })
          }
          disabled={id.trim().length === 0}
        >
          Attach
        </Button>
      </div>
    </div>
  );
}

function ModeToggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-3 py-1.5 text-xs font-medium transition ${
        active
          ? "bg-primary text-primary-foreground"
          : "text-white/50 hover:text-white/80"
      }`}
    >
      {children}
    </button>
  );
}

// A labelled segmented control for one creation knob. Kept generic over the
// option value so each caller stays typed to its own enum.
function OptionGroup<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-wide text-white/35">
        {label}
      </span>
      <div className="flex gap-1 rounded-md border border-white/10 bg-black/20 p-0.5">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={`flex-1 whitespace-nowrap rounded px-2 py-1 text-center text-xs transition ${
              value === option.value
                ? "bg-white/15 text-white"
                : "text-white/50 hover:text-white/80"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
