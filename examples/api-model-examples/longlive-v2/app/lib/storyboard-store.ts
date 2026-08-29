import { create } from "zustand";

/**
 * The authored storyboard — pure client state shared by the composer
 * (<Storyboard>), the visual <Timeline>, and the "Start" action. It holds
 * the *plan* you build before pressing start; the model's live position is
 * read separately from `useLongliveV2State`.
 *
 * Grammar (see the model overview): a SCENE opens with the first beat
 * (a `set_shot`) or a `cut`, and runs up to SCENE_BUDGET (48) chunks before
 * it auto-completes. A `shot` is a soft beat inside a scene; a `cut` starts a
 * fresh scene and resets the 48-chunk budget (so cuts are how you extend
 * length). Beats after the opener are scheduled against the cumulative
 * `session_chunk` clock.
 */

export const SCENE_BUDGET = 48;
export const CHUNK_SECONDS = 1.2; // 29 frames at 24fps

export type BeatKind = "shot" | "cut";

export interface Beat {
  id: string;
  kind: BeatKind;
  prompt: string;
  /** Absolute session_chunk this beat fires at. The opener is always 0. */
  atChunk: number;
}

let seq = 0;
const nextId = () => `beat-${seq++}`;

interface StoryboardState {
  beats: Beat[];
  /** Set the opening shot (replaces any existing opener at chunk 0). */
  setOpening: (prompt: string) => void;
  /** Append a scheduled beat (shot or cut) at an absolute chunk. */
  addBeat: (kind: BeatKind, prompt: string, atChunk: number) => void;
  remove: (id: string) => void;
  /** Load a full preset storyboard at once. */
  load: (beats: Omit<Beat, "id">[]) => void;
  clear: () => void;
}

export const useStoryboard = create<StoryboardState>((set) => ({
  beats: [],
  setOpening: (prompt) =>
    set((s) => {
      const opener: Beat = { id: nextId(), kind: "shot", prompt, atChunk: 0 };
      const rest = s.beats.filter((b) => b.atChunk !== 0);
      return { beats: [opener, ...rest].sort((a, b) => a.atChunk - b.atChunk) };
    }),
  addBeat: (kind, prompt, atChunk) =>
    set((s) => ({
      beats: [...s.beats, { id: nextId(), kind, prompt, atChunk }].sort(
        (a, b) => a.atChunk - b.atChunk,
      ),
    })),
  remove: (id) => set((s) => ({ beats: s.beats.filter((b) => b.id !== id) })),
  load: (beats) =>
    set({
      beats: beats
        .map((b) => ({ ...b, id: nextId() }))
        .sort((a, b) => a.atChunk - b.atChunk),
    }),
  clear: () => set({ beats: [] }),
}));

/** The session chunk where each scene begins (every `cut`, plus chunk 0). */
export function sceneStarts(beats: Beat[]): number[] {
  const cuts = beats.filter((b) => b.kind === "cut").map((b) => b.atChunk);
  return [0, ...cuts].sort((a, b) => a - b);
}
