export const CEREBRAS_STORY_MODEL = "gemma-4-31b";
export const CEREBRAS_MAX_IMAGES = 2;
export const CEREBRAS_MAX_IMAGE_PAYLOAD = 9 * 1024 * 1024;

export type StoryHistoryItem = {
  sceneSummary: string;
  dialogue: string;
  videoPrompt: string;
};

export type StoryPlanRequest = {
  direction: string;
  duration: number;
  history: StoryHistoryItem[];
  images: string[];
  props: string[];
};

export type StoryPlan = {
  videoPrompt: string;
  sceneSummary: string;
  dialogue: string;
  model: string;
  latencyMs: number | null;
};
