import amara from "@/assets/amara.jpg";
import daniel from "@/assets/daniel.jpg";
import kenji from "@/assets/kenji.jpg";
import maya from "@/assets/maya.jpg";
import type { PresetId } from "@/lib/presets";

/** The preset portraits, bundled from `assets/`; the page's server reads the same files to start a call. */
export const PORTRAITS: Record<PresetId, { src: string }> = { maya, daniel, amara, kenji };
