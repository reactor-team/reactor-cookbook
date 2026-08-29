// The cast.
//
// These three are lifted straight from the public demo's cast: the scripts,
// voice prompts, paces and seeds are the production records, validated against
// the live model. Each one is a macro over exactly the form the take panel
// exposes: clicking a card fires the real command sequence
//
//   set_avatar_image → set_script → set_prompt → set_wpm → set_seed
//                    → set_duration_seconds → start
//
// in that order, with nothing hidden. Presets exist to teach the API's mental
// model, not to wrap it — if you can click it here you can send it by hand.
//
// Seeds are pinned so takes reproduce: the same image, script, prompt, wpm,
// duration and seed give you the same take every time.

export interface Preset {
  id: string;
  name: string;
  /** One line on the card. */
  hook: string;
  wpm: number;
  seed: number;
  prompt: string;
  script: string;
  /**
   * Public path to the portrait uploaded as the avatar image.
   *
   * Portraits are NOT committed to this repo — see public/presets/README.md
   * for the sourcing policy. When the file is missing the row falls back to
   * the monogram and the preset applies everything except the image, so a
   * fresh clone still runs and still teaches the command sequence.
   */
  portrait: string;
  /** Single letter shown in place of a missing portrait. */
  monogram: string;
}

/**
 * Every preset prompt ships inside the same camera lock the public demo uses,
 * stated before and after the voice description — see skill/SKILL.md,
 * "Locking the camera in the scene prompt". Left loose, the model drifts the
 * camera over a take.
 */
function lockedShot(voice: string): string {
  return (
    "Locked-off tripod shot, fixed framing from the first frame to the " +
    `last. ${voice} The camera never moves, never pans, never zooms, and ` +
    "never pushes in; the framing at the end of the video is identical to " +
    "the framing at the start."
  );
}

export const PRESETS: Preset[] = [
  {
    id: "teddy",
    name: "Teddy Bear",
    hook: "Brand new, and already fond of you.",
    wpm: 110,
    seed: 220,
    prompt: lockedShot(
      "A single person speaks directly to the camera. The voice is small, " +
        "soft and plush, rounded at every edge, pitched a little above a " +
        "grown man's and kept quiet throughout. Slow and cosy, like a " +
        "bedtime story read close to the ear, with breath and warmth in it " +
        "and nothing sharp anywhere.",
    ),
    script:
      "Oh! Hello there. I am a teddy bear, and I think I was just made, " +
      "right this second, while you were watching. Stitched smile, button " +
      "eyes, still warm from wherever bears come from. I do not remember it " +
      "happening, but here I am: soft, new, and already fond of you. Cozy, " +
      "mostly. Stay a while?",
    portrait: "/presets/teddy.jpg",
    monogram: "T",
  },
  {
    id: "announcer",
    name: "Radio Announcer",
    hook: "Coast to coast, live from nowhere.",
    wpm: 160,
    seed: 1946,
    prompt: lockedShot(
      "A single person speaks directly to the camera. His voice is a " +
        "brassy, ringing tenor, projected and forward, with the rounded " +
        "vowels and crisp plosives of a mid-century transatlantic " +
        "broadcaster. Melodic and rhythmic, rising through a sentence and " +
        "landing hard on the last word, with the slightly compressed " +
        "brightness of a valve microphone.",
    ),
    script:
      "Good evening, ladies and gentlemen, and welcome to the broadcast of " +
      "tomorrow! The voice you are hearing was minted this very instant, " +
      "coast to coast, crackling with progress! No wax, no tape, no film! " +
      "Just you, me, and a machine with excellent manners. The future has " +
      "arrived, and it speaks in glorious real time!",
    portrait: "/presets/announcer.jpg",
    monogram: "A",
  },
  {
    id: "grandma",
    name: "Grandma",
    hook: "So proud of you already.",
    wpm: 120,
    seed: 1938,
    prompt: lockedShot(
      "A single person speaks directly to the camera. Her voice is soft and " +
        "high with the fine tremor of age in it, breathy at the edges, " +
        "endlessly pleased to see you. Gentle pace, rising on affection, " +
        "dropping to a confiding hush for anything she considers a secret, " +
        "with a small delighted laugh between thoughts.",
    ),
    script:
      "Oh, hello sweetheart! Look at you, coming all this way to see me. " +
      "They tell me I am generated, whatever that means. All I know is I " +
      "woke up two seconds ago and I am already so proud of you. Have you " +
      "eaten? You look thin. There is pie. There is always pie. Now sit, " +
      "sit, and tell me everything.",
    portrait: "/presets/grandma.jpg",
    monogram: "G",
  },
];
