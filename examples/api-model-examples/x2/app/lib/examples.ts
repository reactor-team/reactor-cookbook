export interface PromptExample {
  label: string;
  prompt: string;
}

// Quick-prompt chips. Deliberately generic — no asset-specific nouns ("the
// subject", not "the dog") — so they work against whatever source the user
// brings; this is a remixable dev quick-start, not a scripted demo. Two of
// the three are pointer-driven, to land the point that you steer this model
// with your mouse.
export const PROMPT_EXAMPLES: readonly PromptExample[] = [
  {
    label: "Replace with reference",
    prompt:
      "replace the subject in the video with the character from the reference image",
  },
  {
    label: "Spawn at pointer",
    prompt: "place the character from the reference image at the pointer",
  },
  {
    label: "Follow the drag",
    prompt: "make the subject follow the pointer as you drag",
  },
] as const;
