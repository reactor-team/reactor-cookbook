export interface PresetClip {
  id: string;
  label: string;
  src: string;
}

// Demo source videos - generated originals, no third-party footage (see
// public/clips/CREDITS.md). Each pairs with a demo prompt chip: ball +
// "swap object for character", dog + "replace with reference", and figure +
// "follow the drag" (steer the figure with the pointer instead of a
// reference image).
export const PRESET_CLIPS: readonly PresetClip[] = [
  {
    id: "ball",
    label: "Red ball (swap object)",
    src: "/clips/ball.mp4",
  },
  {
    id: "dog",
    label: "Dog (replace with reference)",
    src: "/clips/dog.mp4",
  },
  {
    id: "figure",
    label: "Figure (drag to animate)",
    src: "/clips/figure.mp4",
  },
] as const;

export interface PresetImage {
  id: string;
  label: string;
  src: string;
}

// Demo images. They double as image-mode sources (streamed as a constant
// feed for drag-to-animate) and as reference-image presets. The model's own
// validation images are third-party IP, so instead we ship generated
// originals: one clean single subject each, on a plain backdrop, so they
// isolate cleanly for a character swap and read well as a steerable subject.
//
// All landscape 16:9 to match the clips (and the model's landscape output).
// A portrait reference conditioned into a landscape output distorts badly, so
// the whole preset set is kept to one aspect (see README, "Aspect ratios").
export const PRESET_IMAGES: readonly PresetImage[] = [
  { id: "kitten", label: "Kitten", src: "/refs/kitten.jpg" },
  { id: "knight", label: "Knight", src: "/refs/knight.jpg" },
  { id: "wizard", label: "Wizard", src: "/refs/wizard.jpg" },
] as const;
