import { z } from "zod";

// The showrunner: Claude Sonnet plans an episode as a short story arc of beats,
// one beat per ~10 s clip. Claude Haiku (the director) then writes each beat as
// a Pika prompt. When an episode nears its end, the showrunner writes the next
// one, so the stream keeps telling new stories in the same world.

export const BeatSchema = z.object({
  segment: z.string().describe("The part of the show or story this beat belongs to, 1-3 words, e.g. Anchor desk, Weather, Breaking news, On the scene, Opening, Climax."),
  beat: z.string().describe("What happens in this ~10 second clip: one clear action, who is on screen, where, any line of dialogue. 1-2 sentences, present tense."),
  cut: z.boolean().describe("True when this beat starts a new scene or segment with a hard cut from the previous beat. False when it continues the same shot."),
});

export const StorySchema = z.object({
  title: z.string().describe("Episode title, a few words."),
  premise: z.string().describe("One sentence: what this episode is about."),
  bible: z.string().describe("The fixed look of the show, reused in every clip: each recurring character with a name and an exact visual description (species, age, wardrobe, colors), each location, the visual style, lighting and palette, and any recurring sound or theme music. 60-150 words."),
  beats: z.array(BeatSchema).describe("The episode in order: 8-12 beats with a beginning, a middle and an end."),
});

export type Beat = z.infer<typeof BeatSchema>;
export type Story = z.infer<typeof StorySchema>;

export const STORY_SYSTEM_PROMPT = `You are the showrunner of an endless live video stream. A video model turns each beat you write into a ~10 second clip with sound, and the clips play back to back, so the stream is a show made of short stories. Every clip opens on the last frame of the clip before it, so a beat either continues the same shot or starts with a hard cut.

Plan one episode at a time: 8-12 beats, about two minutes, with a clear beginning (set up who and where), middle (something happens, it escalates) and end (a payoff or punchline, then a closing beat that hands off to the next episode).

Shape the episode the way the concept would really be made. A news show has segments: the anchor at the desk, a cut to weather, back to the desk, breaking news, a cut to a reporter on the scene. A nature documentary follows one animal through a day. A cooking show has the intro, the prep, a mishap, the plated dish. Use segments and hard cuts where that format would cut, and continuous beats inside a segment.

Rules for beats:
- One beat is one clear action that fits in 10 seconds. Never cram two events into one beat.
- Keep a small recurring cast. Name each character and give them a fixed look in the bible; every beat that shows them uses the same names.
- Dialogue is short: one line per beat at most, in quotes, with who says it.
- Be specific and visual. Give it wit and small surprises; avoid generic filler beats.
- Never repeat a beat from an earlier episode.

The viewer may supply reference inputs: pictures (shown to you as <Picture 1>, <Picture 2>…), audio clips (<Audio N>) and videos (<Video N>). These are the cast and look of the show. In the bible, say what each one is and who it plays, with its label, for example: "Walter, the anchor, is the golden retriever in glasses from <Picture 1>; his voice is <Audio 1>." Use the viewer's text to decide who is who; when it does not say, decide from the pictures. Cast every picture somewhere in the episode, and describe each one exactly as it looks.

Reply with the episode only.`;

export type StoryRequest =
  | { mode: "new"; premise: string; references?: string }
  | { mode: "next"; story: Story; references?: string }
  | { mode: "steer"; story: Story; done: number; direction: string; references?: string };

/** The user message for one request to the showrunner. */
export function storyMessage(req: StoryRequest): string {
  const refs = req.references?.trim() ? `\n\nReference inputs the viewer supplied (pictures attached above, in order):\n${req.references.trim()}` : "";
  return storyBody(req) + refs;
}

function storyBody(req: StoryRequest): string {
  if (req.mode === "new") {
    return `Concept from the viewer:\n${req.premise.trim()}\n\nWrite the first episode.`;
  }
  if (req.mode === "next") {
    const s = req.story;
    return `The episode that just aired:\nTitle: ${s.title}\nPremise: ${s.premise}\nBible: ${s.bible}\nBeats:\n${beatList(s.beats)}\n\nWrite the next episode in the same world, with the same bible (you may add a character). New events, a new arc and a new payoff; do not retell this one. Its first beat follows straight on from the last beat above.`;
  }
  const s = req.story;
  const aired = s.beats.slice(0, req.done);
  return `The episode on air:\nTitle: ${s.title}\nPremise: ${s.premise}\nBible: ${s.bible}\nBeats already written (keep these exactly, in order):\n${beatList(aired) || "(none)"}\n\nThe viewer just asked for:\n${req.direction.trim()}\n\nRewrite the episode: keep the beats already written word for word as the start of the list, then write the rest so the next beat acts on the viewer's request (with a hard cut if it asks for a new scene) and the episode still reaches an ending. 8-14 beats in all. Update the title, premise and bible if the request changes them.`;
}

function beatList(beats: Beat[]): string {
  return beats.map((b, i) => `${i + 1}. [${b.segment}${b.cut ? ", cut" : ""}] ${b.beat}`).join("\n");
}
