export const OPENAI_STORY_MODEL = "gpt-5.6-luna";
export const OPENAI_MAX_IMAGES = 2;
export const OPENAI_MAX_IMAGE_PAYLOAD = 9 * 1024 * 1024;

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
