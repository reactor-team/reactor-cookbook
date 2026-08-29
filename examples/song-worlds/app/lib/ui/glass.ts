/**
 * Shared glassmorphism class strings — visual-only, zero logic. Kept
 * here so the frosted-glass treatment stays consistent across
 * BundlePicker / ScorePanel / UploadPanel / SongWorldApp / WorldStage
 * without duplicating (and slowly drifting) the same Tailwind string in
 * five files.
 */

/** Base frosted panel: cards, banners, dropdowns. */
export const GLASS_PANEL =
  "rounded-2xl border border-white/10 bg-white/[0.04] backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.35)]";

/** Panel that's also a hoverable/clickable row (bundle list items). */
export const GLASS_PANEL_INTERACTIVE =
  GLASS_PANEL +
  " transition-colors hover:border-white/20 hover:bg-white/[0.08]";

/** Denser glass for HUD chips floating over the live world (countdown,
 *  "click to look around") — a bit darker so it reads over any footage. */
export const GLASS_CHIP =
  "rounded-full border border-white/10 bg-black/40 backdrop-blur-md shadow-[0_4px_16px_rgba(0,0,0,0.4)]";

/** Error/warning variant of GLASS_PANEL. */
export const GLASS_PANEL_ERROR =
  "rounded-2xl border border-red-400/25 bg-red-500/10 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.35)]";

export const GLASS_PANEL_WARNING =
  "rounded-2xl border border-amber-400/25 bg-amber-500/10 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.35)]";

/** Primary CTA — keeps the brand gold, wrapped in the same soft-glow
 *  glass language as everything else. */
export const GLASS_BUTTON_PRIMARY =
  "rounded-xl bg-brand/90 px-5 py-3 text-sm font-medium text-brand-fg shadow-[0_8px_24px_rgba(253,245,198,0.18)] backdrop-blur transition hover:bg-brand disabled:pointer-events-none disabled:opacity-50";

/** Secondary action (Replay, New song, different song). */
export const GLASS_BUTTON_SECONDARY =
  "rounded-xl border border-white/15 bg-white/5 px-5 py-2.5 text-sm text-zinc-200 backdrop-blur-xl transition hover:border-white/25 hover:bg-white/10 disabled:pointer-events-none disabled:opacity-50";
