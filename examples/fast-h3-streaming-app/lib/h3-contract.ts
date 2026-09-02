export const FAST_H3_MODEL = "reactor/fast-h3";
export const FAST_H3_DISPLAY_NAME = "FastH3";
export const FAST_H3_CANVAS = "1344×768";
export const FAST_H3_DEFAULT_CLIP_SECONDS = 10;
export const FAST_H3_MAX_PROMPT_LENGTH = 800;

export type FastH3ClipSeconds = 6 | 10 | 14;

export type FastH3ClipInfo = {
  clip_id: string;
  prompt: string;
  metadata?: string;
  frames: number;
  seconds: number;
  seed: number;
  continue_from_clip_id: string | null;
  has_starting_frame: boolean;
  ready: boolean;
};

export type FastH3Queue = {
  generation: FastH3ClipInfo[];
  playout: FastH3ClipInfo[];
  history: FastH3ClipInfo[];
};

export function clipFromMessageData(data: unknown): FastH3ClipInfo | null {
  if (!data || typeof data !== "object") return null;
  const clip = (data as { clip?: unknown }).clip;
  if (!clip || typeof clip !== "object") return null;
  const candidate = clip as Partial<FastH3ClipInfo>;
  return typeof candidate.clip_id === "string" && typeof candidate.prompt === "string"
    ? candidate as FastH3ClipInfo
    : null;
}

export function queueFromMessageData(data: unknown): FastH3Queue | null {
  if (!data || typeof data !== "object") return null;
  const candidate = data as Partial<FastH3Queue>;
  if (!Array.isArray(candidate.generation) || !Array.isArray(candidate.playout) || !Array.isArray(candidate.history)) {
    return null;
  }
  return candidate as FastH3Queue;
}
