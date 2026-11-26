import { z } from "zod";

// The shot check, adapted from the Reactor Slack bot's ending check: a vision
// model compares the end of a built clip with its script and fails it when a
// cast member is missing, doubled, merged or changed body, or the beat's event
// clearly did not happen. Generation API only, where each clip's last frame
// arrives before the clip plays.

export const VerdictSchema = z.object({
  bodies: z.string().describe("For each cast member given an earlier picture: their legs and feet and clothes there, then in this picture. Written before deciding. Empty when there is no earlier picture."),
  passed: z.boolean(),
  score: z.number().int().describe("5 when everything holds clearly, 1 when broken beyond recognition. A fail gets 1 to 3."),
  missed: z.enum(["", "person", "body", "event"]).describe("What failed; empty for a pass."),
  reason: z.string().describe("One sentence naming the cast member who broke and what happened, or what holds for a pass."),
});
export type Verdict = z.infer<typeof VerdictSchema>;

export const CHECK_SYSTEM_PROMPT = `You check the end of one filmed clip from a live video stream. A video model filmed it from a written script, and it sometimes breaks the script by the end: it merges two characters into one, makes two copies of one character, turns one into someone else, gives a character a different body from the one they had before, or skips the one event the clip is about.

You get the clip's beat, its prompt, the cast members it shows (each with a label and a look), and the clip's last frame. When the clip before it showed the same cast, you also get that clip's last frame as an earlier picture. Lighting, small costume details, style and whether the clip is good are not your concern.

People. A cast member the beat shows whom no figure in the last frame matches by the colors, size and shape of their look is missing and fails, unless the beat or prompt has them leave or the camera move frames them out. Two figures who both match one cast member's look are copies and fail. A cast member who turned into someone else fails. People nobody gave a label (a crowd, extras) never fail a clip. When the beat shows nobody, any person in the frame fails it.

Body. Each cast member keeps one body: the same shape, the same way of standing (two legs or four, upright or not) and the same clothes as the earlier picture and their look. A close-up that crops the legs is no change. A change the beat asks for passes.

Event. Decide the one event the beat turns on (never a camera move). The last frame shows only the end, so the event may have happened earlier: fail only when the frame clearly shows it did not happen, such as a door still shut that the beat opens. When the frame agrees or says nothing, pass.

Answer with the verdict only.`;

export type CheckRequest = {
  beat: string;
  prompt: string;
  cast: { label: string; look: string }[];
  frame: string;
  earlier?: string;
};

export function checkMessage(r: CheckRequest, hasEarlier: boolean): string {
  const cast = r.cast.length ? r.cast.map((c) => `@${c.label}: ${c.look}`).join("\n") : "Nobody.";
  return `Beat: ${r.beat}\n\nCast this clip shows:\n${cast}\n\nPrompt:\n${r.prompt}\n\nPictures, in order: ${hasEarlier ? "the earlier picture (the clip before's last frame), then " : ""}this clip's last frame.`;
}
