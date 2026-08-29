// Self-contained design tokens + helpers for the LongLive 2 example.
// Class-string constants (composed inline) keep every card, label, and
// focus ring identical across components — the same discipline as the
// webapp's design-system tokens, scaled down. No dependencies.

/** Tiny classname joiner (filters falsy). Avoids a clsx/tailwind-merge dep. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Mono uppercase section label — ONE definition, used by every card eyebrow. */
export const EYEBROW =
  "font-mono text-[10px] uppercase tracking-wider text-zinc-400";

/** Sidebar card chrome. */
export const PANEL = "rounded-lg border border-zinc-800 bg-zinc-900/40";

/** Gold focus ring for every interactive element. */
export const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950";

/** ~1.2s per chunk (29 frames @ 24fps) — mirrors storyboard-store. Used only
 *  for the secondary "~Ns" hint; chunks remain the primary unit. */
export const CHUNK_SECONDS = 1.2;

/** Whole-second string from a chunk count, e.g. secs(24) -> "29s". */
export const secs = (chunks: number) =>
  `${Math.round(chunks * CHUNK_SECONDS)}s`;

/** m:ss from whole seconds. */
export const timecode = (totalSec: number) => {
  const m = Math.floor(totalSec / 60);
  const s = Math.round(totalSec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};
