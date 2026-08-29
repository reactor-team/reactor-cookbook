export interface PresetClip {
  id: string;
  label: string;
  src: string;
}

export const PRESET_CLIPS: readonly PresetClip[] = [
  {
    id: "replace-background-softly",
    label: "Background swap",
    src: "/clips/replace-background-softly.mp4",
  },
] as const;
