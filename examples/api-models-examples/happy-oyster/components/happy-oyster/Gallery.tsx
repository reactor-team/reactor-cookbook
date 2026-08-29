"use client";

// Featured worlds as sidebar quick-prompts (the "Try a prompt" pattern the
// other Reactor examples use), grouped by experience mode. Each tile
// resolves to a WorldIntent: attach when the world has a pinned pre-built id
// (instant), otherwise create from its prompt (~30s build).

import { useState } from "react";
import {
  FEATURED_WORLDS,
  modeName,
  type FeaturedWorld,
  type WorldIntent,
} from "@/lib/worlds";
import { SectionLabel } from "./ui";

export function Gallery({
  onIntent,
}: {
  onIntent: (intent: WorldIntent) => void;
}) {
  const adventure = FEATURED_WORLDS.filter((world) => world.mode === 1);
  const directing = FEATURED_WORLDS.filter((world) => world.mode === 2);
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <SectionLabel>Example worlds</SectionLabel>
      <ModeGroup label="Adventure" worlds={adventure} onIntent={onIntent} />
      <ModeGroup label="Directing" worlds={directing} onIntent={onIntent} />
    </div>
  );
}

function ModeGroup({
  label,
  worlds,
  onIntent,
}: {
  label: string;
  worlds: FeaturedWorld[];
  onIntent: (intent: WorldIntent) => void;
}) {
  if (worlds.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-tight text-white/35">
        {label}
      </span>
      <div className="grid grid-cols-2 gap-1.5">
        {worlds.map((world) => (
          <WorldTile key={world.key} world={world} onIntent={onIntent} />
        ))}
      </div>
    </div>
  );
}

function WorldTile({
  world,
  onIntent,
}: {
  world: FeaturedWorld;
  onIntent: (intent: WorldIntent) => void;
}) {
  const [imageBroken, setImageBroken] = useState(false);
  const showImage = !!world.image && !imageBroken;
  return (
    <button
      onClick={() =>
        onIntent(
          world.encryptedWorldId
            ? {
                kind: "attach",
                mode: modeName(world.mode),
                encryptedWorldId: world.encryptedWorldId,
                title: world.title,
              }
            : {
                kind: "create",
                mode: modeName(world.mode),
                params: { prompt: world.prompt },
                title: world.title,
              },
        )
      }
      title={world.prompt}
      className="group relative flex h-20 flex-col justify-end overflow-hidden rounded-md border border-white/[0.08] p-2 text-left transition hover:border-white/30"
    >
      <div
        aria-hidden
        className="absolute inset-0 transition group-hover:scale-105"
        style={{ background: world.gradient }}
      />
      {showImage && (
        <img
          src={world.image}
          alt=""
          aria-hidden
          loading="lazy"
          onError={() => setImageBroken(true)}
          className="absolute inset-0 h-full w-full object-cover transition group-hover:scale-105"
        />
      )}
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent"
      />
      <span className="relative text-[11px] font-medium leading-tight text-white">
        {world.title}
      </span>
      {world.encryptedWorldId && (
        <span className="relative font-mono text-[9px] uppercase tracking-tight text-white/60">
          instant
        </span>
      )}
    </button>
  );
}
