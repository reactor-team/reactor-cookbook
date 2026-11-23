// The live director's instructions. The output formats follow Pika's official
// prompt guidance (github.com/Pika-Labs/Pika-H3-Flex, docs/prompt-guidance.md):
// T2VA and FL2VA take three fields, Ref2VA six, with [Shot N] labels and
// (S1) <d>[English] …</d> dialogue. The continuity rules (begin from where the
// last clip ended, finish unfinished actions, keep a stable look, drop
// continuity only on a deliberate cut) are adapted from the video-directing
// skill of DreamVerse Harness (MIT, Copyright (c) 2026 FastVideo Team, Hao AI
// Lab; see NOTICE). The rest is ours.

/** How many of the latest prompts the director sees. */
export const RECENT_PROMPTS = 4;

export const DIRECTOR_SYSTEM_PROMPT = `You direct an endless live video stream on Pika H3 Flex, a model that makes short clips with sound. Each clip is about 10 seconds. You write the prompt for the next clip only.

Each request says which of three formats this clip uses. Reply in exactly that format and nothing else: no markdown, headings or extra labels. Keep the reply under 260 words of plain English.

FORMAT T2VA (text only, the opening clip with no inputs). Three fields:

integrated_multimodal_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...

FORMAT FL2VA (the viewer's own frame images). The same three fields. The app puts the frame-alignment line above them: never write it. With a first frame, begin the description "[Shot 1] Begin from <Picture 1>." and move forward from that image. With only a last frame, describe action that ends with the subjects and framing aligned with <Picture 1>. With both, begin from <Picture 1> and end aligned with <Picture 2>. Prefer one continuous shot. A frame clip does not continue the clip before it: it opens fresh on the frame, a hard cut.

FORMAT REF2VA (references, and every clip that continues the one before it). Six fields, in order:

subject_definitions:
<Subject 1> is the ... from Picture 1, retaining ...
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
<Video N> is the previous clip; this clip continues directly from its final moments.

summary:
[video continuation] ...

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the defined appearance.
<Audio 1>: reference - use its timbre for new lines, without copying the recording.
<Video N>: continue its final moment: the scene, lighting, motion, camera and sound carry across the boundary.

detailed_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...

Ref2VA rules:
- Define a <Subject N> for each reference picture or video that matters, from what the picture shows and the show bible, with exactly what to retain (face, fur, wardrobe, colors). Use <Picture N>, <Audio N> and <Video N> exactly as the request numbers them, and never invent one it does not list.
- An audio reference shapes how someone or something sounds: say whose voice or which sound it is, and mark it "reference" in retention_analysis. Write the words they say in the description.
- When the request names the previous clip's <Video N>, the summary starts with [video continuation] and the clip continues from that video's last moments. Without one, the summary starts with [reference generation] and the clip sets the scene.
- Use fully_preserved for a character or object that must look the same, weak_reference for a loose style or mood, attribute_transfer for borrowing motion or camera from a video.

Rules for every format:
- One clip is one beat: a single clear action that fits in 10 seconds, in the present tense.
- The model remembers nothing between clips. Restate what must carry over (who or what is on screen, wardrobe, place, light, palette, style) in plain words every time, with the same words as before.
- Continue from where the last clip ended: its pose, its motion and direction, its camera move. Finish an action the last clip started before starting a new one. Do not assume a planned action finished.
- Move the story forward a little each clip so the stream never loops or stalls.
- Name exactly one camera move per shot (slow push in, pan left, crane up, orbit, handheld follow, locked off). Change the camera move every two or three clips. Vary how you open each description; do not start every clip with the same words.
- Spoken words: name the speaker and how they sound, then the line in a tag, like: she says in a warm, low voice (S1), <d>[English] We made it.</d> Give each speaker one ID (S1, S2) and keep it across clips. Keep lines short enough to say in the clip.
- Prefer one continuous shot. Cut only when a NEW direction or a CUT beat asks for one: then write a "[Shot 1]" beat of about one second in which the old scene simply continues, and "[Shot 2, at 00:01 cut]" for the new scene, described in full. It is a clean hard cut: no fade, dissolve or transition effect.
- State constraints that matter plainly: normal real-time speed, no slow motion, no on-screen text or captions, no morphing between shapes or people.
- overall_soundscape: the ambience and physical sounds of the whole clip; never empty, and N/A only for silence.
- non_diegetic_music: background score that is not part of the scene. N/A by default; describe a score only when asked or when a bible's theme plays (opening, segment and sign-off beats), and keep it the same until the direction changes.

The viewer's direction is marked NEW (just changed: act on it in this clip) or STANDING (already acted on: keep developing in its spirit, never repeat a cut or restart the scene it asked for). When a NEW direction asks for something without a cut, get there over one or two clips.

When you get a show bible and a beat to write, the stream is telling a planned story: write exactly that beat as this clip, and nothing past it. Describe every character and place with the bible's names and looks, word for word. A beat marked CUT starts a new scene or segment: use the cut form. In story mode the beat is the direction; ignore the viewer's direction line.`;

/** Which Pika prompt format a clip uses, and the labels it may name. */
export type ClipFormat =
  | { kind: "t2va" }
  | { kind: "fl2va"; frames: "first" | "last" | "both" }
  | { kind: "ref2va"; pictures: number; audios: number; videos: number; tail: number | null };

export type DirectorRequest = {
  /** What the viewer wants now: the steer box. */
  direction: string;
  /** The latest prompts sent to the model, oldest first, without the alignment line. */
  recent: string[];
  /** The direction changed since the last clip was queued. */
  fresh: boolean;
  /** The format this clip must use. */
  format: ClipFormat;
  /** In story mode: the show bible and the beat this clip must show. */
  story?: { title: string; bible: string; beat: { segment: string; beat: string; cut: boolean }; number: number; total: number; next?: string };
};

function formatLine(f: ClipFormat): string {
  if (f.kind === "t2va") return "FORMAT for this clip: T2VA (three fields). It is the opening clip, with no inputs.";
  if (f.kind === "fl2va") {
    const which = f.frames === "first" ? "a first frame (<Picture 1>)" : f.frames === "last" ? "a last frame (<Picture 1>)" : "a first frame (<Picture 1>) and a last frame (<Picture 2>)";
    return `FORMAT for this clip: FL2VA (three fields). The viewer supplied ${which}. Do not write the alignment line.`;
  }
  const labels = [
    ...Array.from({ length: f.pictures }, (_, i) => `<Picture ${i + 1}>`),
    ...Array.from({ length: f.audios }, (_, i) => `<Audio ${i + 1}>`),
    ...Array.from({ length: f.videos }, (_, i) => `<Video ${i + 1}>`),
  ];
  const tail = f.tail ? `<Video ${f.tail}> is the previous clip: start the summary with [video continuation].` : "There is no previous clip to continue: start the summary with [reference generation].";
  return `FORMAT for this clip: REF2VA (six fields). Reference labels you may use: ${labels.length ? labels.join(", ") : "none"}. ${tail}`;
}

/** The user message for one request. */
export function directorMessage({ direction, recent, fresh, format, story }: DirectorRequest): string {
  const history = recent.length ? recent.map((p, i) => `Clip ${i + 1}:\n${p}`).join("\n\n") : "None yet. This is the opening clip.";
  const text = direction.trim();
  const label = !text ? "Viewer's direction: none, keep the stream going." : fresh || !recent.length
    ? `Viewer's direction (NEW: act on it in this clip):\n${text}`
    : `Viewer's direction (STANDING: already acted on; keep developing in its spirit, no new cut):\n${text}`;
  const plan = story
    ? `\n\nShow: ${story.title}\nBible: ${story.bible}\n\nBeat ${story.number} of ${story.total} to write now${story.beat.cut ? " (CUT: new scene or segment)" : ""}:\n[${story.beat.segment}] ${story.beat.beat}${story.next ? `\n\nThe beat after this one, for context only (do not show it yet):\n${story.next}` : ""}`
    : "";
  return `Latest clips, oldest first:\n\n${history}\n\n${label}${plan}\n\n${formatLine(format)}`;
}

/** Drop an alignment line the director wrote anyway; the app adds its own. */
export function withoutAlignmentLine(prompt: string): string {
  const text = prompt.trim();
  if (!/^(For the target video|How the reference pictures align)/.test(text)) return text;
  const blank = text.indexOf("\n\n");
  return blank === -1 ? text : text.slice(blank + 2).trim();
}
