export interface PromptExample {
  label: string;
  prompt: string;
}

export const PROMPT_EXAMPLES: readonly PromptExample[] = [
  {
    label: "Van Gogh",
    prompt: "Van Gogh oil painting, swirling brushstrokes, vivid colors",
  },
  {
    label: "Neon Cyber",
    prompt: "neon cyberpunk city glow, electric blue and magenta lights",
  },
  {
    label: "Claymation",
    prompt: "claymation stop-motion, soft clay textures, warm studio light",
  },
  {
    label: "Pencil Sketch",
    prompt: "detailed pencil sketch, fine crosshatching, paper texture",
  },
  {
    label: "Golden Hour",
    prompt: "golden hour film look, warm amber haze, cinematic grain",
  },
  {
    label: "Watercolor",
    prompt: "loose watercolor wash, bleeding edges, soft pastel palette",
  },
] as const;
