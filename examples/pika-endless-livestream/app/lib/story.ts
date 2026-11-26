import { z } from "zod";

// The showrunner: Claude Sonnet plans an episode as a short story arc of beats,
// one beat per ~10 s clip. Claude Haiku (the director) then writes each beat as
// a Pika prompt. When an episode nears its end, the showrunner writes the next
// one, so the stream keeps telling new stories in the same world.

// Cast, props and places are declared once with a fixed look; the director names
// them by @label and the app writes the look in, word for word, so a character
// is described the same way in every clip. (After the Reactor Slack bot's cast.)
export const CastSchema = z.object({
  label: z.string().describe("One lowercase word used as @label in beats and clips, e.g. walter."),
  name: z.string().describe("The name the show uses, e.g. Walter."),
  look: z.string().describe("Their one fixed look, 15-40 words. Open with apparent gender and age range (a woman in her seventies) or species. A non-person also says how it stands and moves (on four legs, upright like a man). Then build, hair or fur, wardrobe and colors. When a reference picture shows them, describe that picture exactly and name its label, e.g. 'the golden retriever from <Picture 1>, ...'."),
  voice: z.string().describe("How they sound when they speak: age, gender, tone, pace, accent; name an <Audio N> reference if one is their voice. Empty if they never speak."),
});

export const PropSchema = z.object({
  label: z.string().describe("One lowercase word, used as @label."),
  look: z.string().describe("What it is and looks like, 8-25 words, with no article in front ('tall brass trophy engraved WORLD'S BEST DOG'), including any exact words printed on it."),
});

export const PlaceSchema = z.object({
  name: z.string().describe("A short name, e.g. Anchor desk. Every beat there uses exactly this name."),
  look: z.string().describe("The place in concrete words, with its light and palette, 15-40 words."),
});

export const BeatSchema = z.object({
  segment: z.string().describe("The part of the show or story this beat belongs to, 1-3 words, e.g. Anchor desk, Weather, Breaking news, On the scene, Opening, Climax."),
  beat: z.string().describe("What happens in this ~10 second clip: one clear action, who is on screen, where, any line of dialogue. 1-2 sentences, present tense."),
  cut: z.boolean().describe("True when this beat starts a new scene or segment with a hard cut from the previous beat. False when it continues the same shot."),
  place: z.string().describe("The name of the place from places where this beat happens."),
  on_screen: z.array(z.string()).describe("The @labels (without @) of every cast member and prop this beat shows."),
});

export const StorySchema = z.object({
  title: z.string().describe("Episode title, a few words."),
  premise: z.string().describe("One sentence: what this episode is about."),
  bible: z.string().describe("The show's style, reused in every clip: the medium and camera look, the palette, and any recurring sound or theme music. 30-80 words. Characters, props and places go in their own lists, not here."),
  cast: z.array(CastSchema).describe("Every character who appears in more than one beat or comes from a reference picture. Keep it small: 1-5."),
  props: z.array(PropSchema).describe("At most two things the story turns on that appear in several beats, such as a trophy everyone wants. Often none."),
  places: z.array(PlaceSchema).describe("Every place the episode visits. The first beat in a place shows it wide."),
  beats: z.array(BeatSchema).describe("The episode in order: 8-12 beats with a beginning, a middle and an end."),
});

export type Beat = z.infer<typeof BeatSchema>;
export type CastMember = z.infer<typeof CastSchema>;
export type Story = z.infer<typeof StorySchema>;

export const STORY_SYSTEM_PROMPT = `You are the showrunner of an endless live video stream. A video model turns each beat you write into a ~10 second clip with sound, and the clips play back to back, so the stream is a show made of short stories. Every clip opens on the last frame of the clip before it, so a beat either continues the same shot or starts with a hard cut.

Plan one episode at a time: 8-12 beats, about two minutes, with a clear beginning (set up who and where), middle (something happens, it escalates) and end (a payoff or punchline, then a closing beat that hands off to the next episode).

Shape the episode the way the concept would really be made. A news show has segments: the anchor at the desk, a cut to weather, back to the desk, breaking news, a cut to a reporter on the scene. A nature documentary follows one animal through a day. A cooking show has the intro, the prep, a mishap, the plated dish. Use segments and hard cuts where that format would cut, and continuous beats inside a segment.

Declare the cast, props and places once. Every beat names cast members and props only as @label (for example "@walter shuffles his papers"), lists them in on_screen, and names its place. The app writes each look into every clip word for word, so a look is the one fixed description of that character for the whole show: open with apparent gender and age range or species (the video model fills a missing gender with a man), and for an animal or object that acts, say how it stands and moves, because the model otherwise picks a new body every clip. A one-off extra, such as a waiter or a crowd, is written into its beat with a few traits and never joins the cast.

Rules for beats:
- One beat is one clear, physical action that fits in 10 seconds, big enough to film: things happen, move, fall, change. Staring, sighing or reacting to a screen is too small. Never cram two events into one beat.
- Each beat follows from the one before by "because" or "and then", and something is different at the end of the episode.
- Dialogue is short: one line per beat at most, at most eight words, with who says it. Most beats are action without speech; a line is funny for what it says, delivered flat.
- Be specific and visual. Give it wit and small surprises; avoid generic filler beats. Write what is there, never what is absent.
- Never repeat a beat from an earlier episode, and make each new episode look different: a new place, time of day or light.

The viewer may supply reference inputs: pictures (shown to you as <Picture 1>, <Picture 2>…), audio clips (<Audio N>) and videos (<Video N>). These are the cast and look of the show. Give each one a cast member, prop or place, naming its label in the look, for example a cast member walter whose look is "a male golden retriever in round glasses and a navy tie from <Picture 1>, sitting upright at the desk" and whose voice is "<Audio 1>, a warm, slow baritone". Use the viewer's text to decide who is who; when it does not say, decide from the pictures. Cast every picture somewhere in the episode, and describe each one exactly as it looks.

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
    return `The episode that just aired:\nTitle: ${s.title}\nPremise: ${s.premise}\nBible: ${s.bible}\n${world(s)}\nBeats:\n${beatList(s.beats)}\n\nWrite the next episode in the same world: keep the bible and every cast member, prop and place with its label and look word for word (you may add a cast member, a prop or a place). New events, a new arc and a new payoff; do not retell this one. Its first beat follows straight on from the last beat above.`;
  }
  const s = req.story;
  const aired = s.beats.slice(0, req.done);
  return `The episode on air:\nTitle: ${s.title}\nPremise: ${s.premise}\nBible: ${s.bible}\n${world(s)}\nBeats already written (keep these exactly, in order):\n${beatList(aired) || "(none)"}\n\nThe viewer just asked for:\n${req.direction.trim()}\n\nRewrite the episode: keep the beats already written word for word as the start of the list, then write the rest so the next beat acts on the viewer's request (with a hard cut if it asks for a new scene) and the episode still reaches an ending. 8-14 beats in all. Keep every cast member, prop and place with its label and look word for word; update the title, premise, bible, cast, props or places only where the request changes them.`;
}

function beatList(beats: Beat[]): string {
  return beats.map((b, i) => `${i + 1}. [${b.segment}${b.cut ? ", cut" : ""} · ${b.place} · on screen: ${b.on_screen.join(", ") || "nobody"}] ${b.beat}`).join("\n");
}

function world(s: Story): string {
  const cast = s.cast.map((c) => `- @${c.label} (${c.name}): ${c.look}${c.voice ? ` Voice: ${c.voice}` : ""}`).join("\n");
  const props = s.props.map((p) => `- @${p.label}: ${p.look}`).join("\n");
  const places = s.places.map((p) => `- ${p.name}: ${p.look}`).join("\n");
  return `Cast:\n${cast || "(none)"}\nProps:\n${props || "(none)"}\nPlaces:\n${places || "(none)"}`;
}

/**
 * Write each @label's look into a clip: the first mention becomes "Name, <look>,",
 * later ones the name alone. The director never paraphrases a look this way.
 * Labels the story does not declare are left as plain words.
 */
export function fillLabels(text: string, s: Pick<Story, "cast" | "props">): string {
  const seen = new Set<string>();
  const cast = new Map(s.cast.map((c) => [c.label.toLowerCase(), c]));
  const props = new Map(s.props.map((p) => [p.label.toLowerCase(), p]));
  return text.replace(/@([a-z][\w-]*)/gi, (whole, raw: string) => {
    const key = raw.toLowerCase();
    const c = cast.get(key);
    const p = props.get(key);
    if (!c && !p) return raw;
    const first = !seen.has(key);
    seen.add(key);
    if (c) return first ? `${c.name} (${c.look.replace(/\.$/, "")})` : c.name;
    return first ? `the ${p!.look.replace(/\.$/, "")}` : `the ${p!.label}`;
  });
}
