"use client";

import { useState, type ReactNode } from "react";

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[10px] uppercase tracking-widest text-white/40">
      {children}
    </div>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[11px] uppercase tracking-widest text-primary/80">
      {children}
    </div>
  );
}

export function Spinner() {
  return (
    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
  );
}

// The world id is a capability: it exists from the moment the backend
// registers the build (visible in world_state snapshots while still
// building), and attachWorld() with it skips the build wait forever after.
export function WorldIdChip({ worldId }: { worldId: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        void navigator.clipboard.writeText(worldId).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
      title="Copy this world's attach id"
      className="max-w-full truncate rounded-md border border-white/10 bg-black/30 px-3 py-1.5 text-left font-mono text-[11px] text-white/50 transition hover:border-white/25 hover:text-white/80"
    >
      {copied ? "Copied attach id" : `id: ${worldId}`}
    </button>
  );
}

export function ModeBadge({ mode }: { mode: number | null }) {
  return (
    <span className="rounded-full border border-white/15 px-2 py-0.5 font-mono text-[10px] uppercase tracking-tight text-white/60">
      {mode === 2 ? "Directing" : "Adventure"}
    </span>
  );
}
