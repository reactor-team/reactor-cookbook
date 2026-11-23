# Pika Endless Livestream (Pika H3 Flex on Reactor)

A bare-bones endless livestream on [Pika H3 Flex](https://docs.reactor.inc). Connect
to the model, turn autoplay on, and keep queuing clips so a built clip is always
waiting before the current one ends.

Describe a show or story, start the live stream, and steer it while it plays.

## Run

```bash
pnpm install
cp .env.example .env.local   # add REACTOR_API_KEY (and ANTHROPIC_API_KEY for the director)
pnpm dev                     # http://127.0.0.1:3000
```

Describe a show or story and press **Start live stream**. While live, type a direction and press **Steer**.

## How it works

All of it is in `app/Livestream.tsx` (~150 lines) and `app/api/token/route.ts`.

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

With `ANTHROPIC_API_KEY` set and **Enhance prompt** on, two Claude models write the stream:

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
