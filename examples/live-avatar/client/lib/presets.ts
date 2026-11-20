/** The portraits that ship in `assets/`, each with the voice it speaks with by default. */
export const PRESETS = [
  { id: "maya", name: "Maya", voice: "marin" },
  { id: "daniel", name: "Daniel", voice: "cedar" },
  { id: "amara", name: "Amara", voice: "coral" },
  { id: "kenji", name: "Kenji", voice: "ash" },
] as const;

export type PresetId = (typeof PRESETS)[number]["id"];

/** OpenAI Realtime voices the agent can speak with. */
export const VOICES = ["marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"];

export function isPreset(id: string): id is PresetId {
  return PRESETS.some((p) => p.id === id);
}
