# Pika Endless Livestream (Pika H3 Flex on Reactor)

A bare-bones endless livestream on [Pika H3 Flex](https://docs.reactor.inc). Connect
to the model, turn autoplay on, and keep queuing clips so a built clip is always
waiting before the current one ends.

Describe a show or story, start the live stream, and steer it while it plays.

## Run

```bash
pnpm install
cp .env.example .env.local   # add REACTOR_API_KEY (and ANTHROPIC_API_KEY or OPENAI_API_KEY for the writers)
pnpm dev                     # http://127.0.0.1:3000
```

Describe a show or story, or pick one of the **Examples**, and press **Start live stream**. While live, type a direction and press **Steer**.

The **API keys** dropdown at the top of the panel lists the Reactor, Claude and OpenAI keys. A key set on
the server (`.env`, `.env.local`, or the host's environment) shows as set, with no input, and wins over
anything pasted. A key the server lacks gets a field. Pasted keys stay in that browser tab
(`sessionStorage`) and go only to this app's own route handlers. The Reactor key is still exchanged for a
session token on the server. With all three keys set on the server, the page only says so. On a public
deployment, anyone who opens the page spends on those keys, so put it behind protection. **Reset** ends the
stream and starts over, for trying another example.

## Streaming API or Generations API

Reactor serves Pika H3 Flex two ways, and the switch in the header picks one. The story, the director, the
inputs and the sidebar work the same with both; only the engine that films and plays the clips changes.

**Streaming API.** The page opens one realtime WebRTC session with the model (`@reactor-models/pika-h3-flex`)
and calls `enqueue` for each clip. The model keeps a playout queue: with autoplay on, it plays each clip the
moment it is built, straight into the video and audio tracks, and holds the last frame if the queue runs dry.
Each clip continues the one before it (latent continuation), so motion, camera and sound carry across the
cut. Use it when the clips should flow as one continuous take.

**Generations API.** Each clip is its own HTTP job: `POST /generations` with the model and the clip's input,
then `GET /generations/{id}` until it succeeds and returns `output.mp4` and `last_frame.png`. The app proxies
both through `app/api/generations` so the API key stays on the server, downloads each video, and plays them
in order in a plain `<video>`, keeping `CLIPS_AHEAD` jobs running. Reference files are reserved with
`POST /files` and uploaded to the presigned URL once. The clips are independent, so the joins are cuts and
continuity comes from the writers and the references. In exchange, each clip arrives as a file before it
plays, which is what lets the shot check look at its last frame and retake a broken clip.

| | Streaming API | Generations API |
| --- | --- | --- |
| Transport | One WebRTC session | One HTTP job per clip |
| Playback | The model's playout queue, live tracks | The page's `<video>`, downloaded MP4s |
| Between clips | Latent continuation: one continuous take | Independent clips: hard cuts |
| Shot check and retake | No (clips play as they are built) | Yes, with Enhance prompt on |
| Code | `send()`, `clipParams()` | `sendGeneration()`, `watchGeneration()`, `checkGeneration()` |

## How it works

The streaming loop is in `app/Livestream.tsx`, and the session token is minted in `app/api/token/route.ts`.

1. **Auth.** `/api/token` trades `REACTOR_API_KEY` for a short-lived JWT scoped to
   `reactor/pika-h3-flex`. The browser connects with the JWT.
2. **Session setup.** On connect: `setAutoplay({ enabled: true })` so each clip plays
   as soon as it's built, and `setFlushOnClipEnd({ enabled: false })` so the last
   frame holds instead of going black if the buffer ever runs dry.
3. **Beat the playhead.** The client keeps `CLIPS_AHEAD` clips queued that haven't
   started playing yet. Whenever that drops (a `clip_started` arrives), it
   `enqueue`s another.
4. **Continuity.** Every clip after the first continues from the last *queued* clip
   with `continue_mode: "frame"` (it opens on that clip's last frame). Frame mode can
   name a clip that is still building, so the queue fills immediately instead of
   waiting on each build.
5. **Tracks.** Video renders in `PikaH3FlexMainVideoView`; audio is a separate
   `main_audio` track played through an `<audio>` element.

The log shows each clip's queue → build → play timeline, including how long a build
took after enqueue. If builds take longer than a clip plays, raise `CLIPS_AHEAD` or
`CLIP_SECONDS`.

## Examples

The **Examples** row loads a starting show: its description and its reference pictures and voices. Dog news is
text only. Treehouse, Product reveal, Sitcom and Food truck are the Pika H3 Flex examples from reactor.inc,
recast as shows that keep going. Reactor made every picture and voice in `public/examples/`. The shows are in
`app/lib/examples.ts`.

## Model inputs

Under the text box, the **Inputs** panel exposes what Pika H3 Flex takes, with the
model's own labels:

- **References**: up to 9 images (`Picture 1–9`), 3 audio clips (`Audio 1–3`,
  2–15 s, 15 s total) and 2 videos (`Video 1–2`, experimental). They go into
  every clip. Name them in your text, e.g. "the corgi in Picture 2".
- **Frames**: a first frame and/or a last frame. They go on the next clip only
  (the first clip before you start; the next clip queued after Steer), which
  starts or ends on that image.

A clip uses references or frames, never both. Every clip after the first continues
the one before with a latent continuation (motion, camera, sound and references carry
across the cut), which needs that clip built first; the director writes the next
prompt while it builds, so the clip is sent the moment the build lands. Files upload
once per session.

## Enhance prompt (story writer + director)

With a writer key set and **Enhance prompt** on, two models write the stream. With Claude (the default) they
are Claude Sonnet 5.5 and Claude Haiku 5.5; with OpenAI (pick it under **Keys**, or set only
`OPENAI_API_KEY`) they are GPT-5.1 and GPT-5.1 mini. Both are in `app/lib/server/writer.ts`.

- **Showrunner** (`/api/story`, Claude Sonnet 5.5): turns your concept into an episode of
  8–12 beats with a beginning, middle and end, plus a bible of characters, places and
  look. It uses the segments the format would really have (a news show cuts between the
  anchor desk, weather, breaking news and a reporter on the scene). When four beats are
  left it writes the next episode in the same world, so the stream keeps telling new
  stories. **Steer** while live rewrites the rest of the current episode around your
  direction; beats already filmed stay.
- **Director** (`/api/director`, Claude Haiku 5.5): writes each beat as one ~10 s clip
  in Pika's official prompt format (three fields, `[Shot N]`, `(S1) <d>[English] …</d>`),
  continuing from the previous clip's last frame or with a hard cut when the beat
  starts a new segment.

With Enhance prompt off, your text goes to the model as is. If a writer call fails,
that clip uses your text and the stream keeps going. The prompts are in
`app/lib/story.ts` and `app/lib/director.ts`; the director's continuity rules are
adapted from [DreamVerse Harness](https://github.com/fastvideo-project/dreamverse-harness)
(MIT, see `NOTICE`).

## Knobs

| Constant | Default | Meaning |
| --- | --- | --- |
| `CLIPS_AHEAD` | 2 | Clips kept queued ahead of the playing one |
| `CLIP_SECONDS` | 10 | Clip length; whole seconds 5–15 |

## Cast, shot check and retake

The approach comes from Reactor's Slack video bot.

- **Fixed cast.** The showrunner declares each recurring character, prop and place once, with one look.
  The director names them only as `@label`, and `fillLabels` in `app/lib/story.ts` writes each look in
  word for word. The model never paraphrases a character between clips.
- **Craft rules.** The director's prompt carries the rules the bot learned on the same family of models.
  Silent clips say "their lips remain completely closed and nobody speaks". Lines are at most eight words.
  Feelings become body actions, and prompts say what is there, not what is absent. In generation mode,
  clips are filmed alone and never point back to the clip before.
- **Shot check (generation API).** Each generation returns its last frame before it plays. With Enhance
  prompt on, `/api/check` has the writer's fast model compare it with the beat and the cast. It fails a clip
  when a cast member is missing, doubled, merged or has a new body, or the beat's event plainly did not
  happen. A failed clip is written again once, aimed at what failed, and the better-scored take plays. The
  debug log shows each verdict. The streaming API plays clips as they are built, so it has no check.
