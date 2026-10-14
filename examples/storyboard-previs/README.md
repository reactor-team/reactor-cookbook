# Storyboard Previs

A storyboard that generates itself. Write a script as a handful of shots, pin
the cast, the location, and the props with reference pictures, press one
button, and [`reactor/fast-h3-omni`](https://docs.reactor.inc/model-api-reference/fast-h3-omni/overview)
plays the sequence back as a live stream, with synchronized sound and the same
faces in every shot, faster than it takes to watch. When it ends, you type what
happens next and the picture continues from the frame it stopped on.

It is the previs problem: a director wants to see a scene move before anyone
builds a set, and today that takes days of boards and animatics. Here it takes
about a minute.

![Desert signal: the station, the traveler, and the receiver](./public/references/station.jpg)

This example demonstrates how to:

- keep a person, a place, and a prop consistent across shots with
  `reference_images`, named in the prompt as `<Picture 1>`, `<Picture 2>`, …;
- write prompts the way the model's
  [prompt guide](https://docs.reactor.inc/model-api-reference/fast-h3-omni/prompt-guide)
  asks for them: subject bindings, one beat, camera, dialogue, soundscape,
  music;
- generate spoken lines with lip sync from the prompt alone;
- run a whole sequence through the model's own playout queue
  (`set_autoplay`, `set_flush_on_clip_end`), so the next shot starts the
  instant the previous one ends while later shots are still building;
- continue the story from the last displayed frame with
  `continue_from_clip_id` and `continue_mode: "frame"`;
- upload pictures once per session and reuse the `FileRef`s;
- record the live WebRTC stream with `MediaRecorder` so the sequence replays
  with the model disconnected; and
- optionally expand a one-line beat into full shot direction with Claude,
  with the raw beat as the fallback.

Two stories are bundled. *Desert signal* is a nine-shot science-fiction short
with a radio call, an answer, and a helicopter rescue. *Dead drop* is a six-shot
thriller in a library. Every picture under `public/references/` is a generated,
fictional reference; the prompts that made them are beside them in
`prompts.json` and `drop-prompts.json`.

## Run it

You need Node.js 20.9 or newer, a WebRTC-capable browser, and a Reactor API
key. Create a key from **API Keys** in the
[Reactor dashboard](https://reactor.inc/dashboard).

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/examples/storyboard-previs
cp .env.example .env.local
```

Add your key to `.env.local`:

```dotenv
REACTOR_API_KEY=rk_your_api_key_here
```

Then install and start the app:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), press **Connect model**,
wait for "pictures ready" in the transport bar, and press **Generate
storyboard**.

The key never leaves the server. `app/api/token/route.ts` exchanges it for a
session-scoped JWT that only allows one `reactor/fast-h3-omni` session, and the
browser connects with that.

### Optional: a director for "what happens next"

Add `ANTHROPIC_API_KEY` to `.env.local` and the beat you type after a sequence
is expanded into full direction by `app/api/direct/route.ts` before it is sent
to the model: the previous shot's subject bindings are kept word for word, a
line states what carries over, and the model writes the new action, camera,
dialogue, and sound. Without the key, your beat is sent as written.

## How it works

**Pictures.** `lib/reference-kit.ts` lists each story's pictures. On
`ready`, every picture a story uses is shrunk to 768 px, uploaded with
`uploadFile`, and cached by id for the session. A shot's `refs` array is the
upload order and therefore the `<Picture N>` numbering in its prompt.

**Prompts.** `lib/storyboard.ts` holds both scripts. Each shot's prompt is
built from shared binding lines, so the traveler or the courier is described
identically in every shot she appears in, followed by one `In one shot,` beat
and labeled `Camera:`, `Dialogue:`, `Soundscape:`, and `Music:` lines. Dialogue
is one short line per clip, quoted, marked "slowly and clearly".

**The sequence.** Generate storyboard enables autoplay and holds the last frame
between clips, then enqueues every shot in order. A linked shot waits only for
its predecessor's `clip_generated` before enqueueing with
`continue_from_clip_id`. The model starts each ready clip as the previous one
ends; the app only listens to `clip_started` and `clip_finished` to drive the
UI and the recorders. `build_seconds` from `queue_update` feeds the
"generated in" counter.

**Recording.** One `MediaRecorder` runs across the whole sequence and one per
shot, all on the same `MediaStream` from the `main_video` and `main_audio`
tracks. Takes are stored in IndexedDB, so Play sequence works offline, which is
the safety net for a live presentation.

**Continuing.** When the picture holds on the last frame, the bar under the
monitor accepts a beat. The app rebuilds the prompt from the previous shot's
bindings plus the new action and sends `continue_mode: "frame"`, so the new
clip's first frame is the frame on screen. If the model refuses, it retries in
`latent` mode and logs the reason in the session log.

## Customize it

- Change a story: edit the shot list in `lib/storyboard.ts`. Keep every shot
  to one beat, name pictures by their position in `refs`, and keep spoken lines
  under eight words.
- Change the cast: drop new JPEGs in `public/references/`, add them to
  `referenceKit` in `lib/reference-kit.ts`, and reference them from the shots.
  The image prompts beside the pictures show the level of detail that keeps a
  face stable.
- Present: the **Present** button in the transport fills the screen with the
  monitor and keeps Generate storyboard and Play sequence on a fading bar.
