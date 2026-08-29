// Self-contained design tokens + helpers for the SANA-Streaming example.
// Class-string constants (composed inline) keep every card, label, and
// focus ring identical across components - the same discipline as the
// webapp's design-system tokens, scaled down. No dependencies.

/** Tiny classname joiner (filters falsy). Avoids a clsx/tailwind-merge dep. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Normalizes an unknown thrown value for display. */
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Mono uppercase section label - ONE definition, used by every card eyebrow. */
export const EYEBROW =
  "font-mono text-[10px] uppercase tracking-wider text-zinc-400";

/** Sidebar card chrome. */
export const PANEL = "rounded-lg border border-zinc-800 bg-zinc-900/40";

/** Gold focus ring for every interactive element. */
export const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950";
