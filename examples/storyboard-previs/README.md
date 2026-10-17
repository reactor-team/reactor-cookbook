# storyboard-previs

A previs tool on top of `reactor/fast-h3-omni`. You write a scene as a list of
shots, each with a prompt, a length, and the reference pictures it uses. The
app uploads the pictures once, queues all the shots, and the model plays them
back-to-back as a live stream while the later ones are still rendering. When
the sequence ends you can type what happens next and it continues from the
last frame.

![](./public/screenshot.jpg)

Two scripts ship with it: *Desert signal* (nine shots, sci-fi) and *Dead
drop* (six shots, a thriller in a library). The reference pictures are
generated and fictional. The prompts used to make them are in
`public/references/prompts.json` and `drop-prompts.json`.

## Running it

You need Node 20.9+, a browser with WebRTC, and a Reactor API key
(dashboard → API Keys).

```bash
cd examples/storyboard-previs
cp .env.example .env.local   # put your key in it
npm install
npm run dev
```

Open http://localhost:3000. Not `127.0.0.1`: the Next dev server won't serve
its client bundle to a different origin and you'll get an empty page.

Click **Connect model**, wait for "pictures ready" in the bar under the
monitor, then **Generate storyboard**. After the sequence ends the prompt bar
appears over the held frame. **Continue** starts the next clip from that
frame; **Cut** makes a new shot in the same scene. **Play sequence** replays
the recording, which works without the model connected. **Present** goes
fullscreen.

Beats you add live show up under "Added live" in the shot list and are not
part of the timeline. **Clear added** removes them, **Reset script** puts a
story back to its original shots.

### Environment

`.env.local` is read by the server route handlers in `app/api/` and nothing else.

```
REACTOR_API_KEY     required. Swapped for a session JWT in app/api/token/route.ts.
                    The JWT allows one fast-h3-omni session; the key never reaches the browser.
ANTHROPIC_API_KEY   optional. Turns the beat you type into full shot direction
                    (app/api/direct/route.ts). Without it the beat is sent as typed.
DIRECTOR_MODEL      optional. Defaults to claude-haiku-5-5.
OPENAI_API_KEY      enables live monitoring through OpenAI Decisions with gpt-6-luna.
```

## Opening Previs from a phone over Tailscale

Connect both devices to the same tailnet. Set `PREVIS_PUBLIC_ORIGIN` in `.env.local`
to the exact HTTPS address shown by Tailscale Serve. This allows API requests from
that configured address while retaining the origin checks. Build and start Previs
on IPv4 loopback, then enable the private proxy:

```bash
npm run build
npm run start -- --hostname 127.0.0.1 --port 3006
# In another terminal (use the app's full CLI path on macOS if needed):
tailscale serve --bg --https=443 http://127.0.0.1:3006
```

With Tailscale connected on the phone, open the printed `https://….ts.net` URL in
Safari or Chrome. HTTPS permits browser camera access. Serve is private to your
tailnet and obeys its access rules. The Mac and the Previs process must remain
running. Disable this proxy with `tailscale serve --https=443 off`.

## Live rules

Open the **Live rules** tab in the right inspector. Add a condition in plain
language, select an action, then start monitoring. Conditions can watch the
playing scene, an explicitly enabled webcam, or text from an external feed.
Rules are also editable as JSON and saved locally. No JavaScript is evaluated.

For the Desert signal demo, try:

```json
[
  {
    "id": "radio-reply",
    "source": "scene",
    "condition": "A person is holding the shortwave radio receiver in their hands.",
    "action": "direct",
    "direction": "A faint reply comes through the radio. She turns toward the distant antenna.",
    "enabled": true
  }
]
```

Actions are `mark` (timestamp and evidence), `chime` (audible cue), `stop`
(stop playback and the local generation sequence), and `direct` (queue a
user-written direction for the next shot). Each needs two distinct observations
with model probability at least 0.85, then fires once until **Rearm rule**.
Editing a rule rearms it. These are model judgments, not verified facts.
The event log shows the captured frame, timestamp, and returned probability.

**Reactive generation.** Start monitoring *before* clicking **Generate storyboard**.
The app then generates and plays one shot at a time, applying queued directions
before preparing the next shot. This intentionally adds a generation gap between
shots, and saves individual takes for **Play sequence**. Existing generated video
cannot be edited in place. Starting monitoring during an ordinary, already queued
sequence can still mark, cue, and stop it; director cues wait for a future
generation. At the end, **Generate next beat** applies remaining cues as a new
clip, or **Cancel** clears them. The app does not append shots indefinitely.

**Director and judging.** Set `ANTHROPIC_API_KEY` for director actions. The creative
brief persists locally; the last 12 detections are passed with the script to the
director as uncertain observations. **Judge the output** checks a concrete visual
requirement such as “The main subject remains clearly visible.” After two clear
violations, it records the issue, or queues a correction if that option is on.
A quality correction can fire once per played clip. A missing or failing director
leaves the cue pending and reports the error instead of claiming an adjustment.

**External input.** Enable the webcam explicitly, then add a `webcam` rule such as
“A person has a raised hand.” Webcam audio is never acquired. Frames are sent only
when monitoring an active webcam rule. Stopping monitoring releases the camera.
For audience comments or headlines, add a `feed` rule and paste text, or connect a
public HTTPS JSON URL with CORS enabled. Accepted shapes are `{ "text": "…" }` or
`{ "items": [{ "title": "…", "text": "…" }] }`. The browser polls every 10 seconds,
without credentials; a private/social API needs your own feed adapter. Failed
refreshes clear stale feed text. Feed contents are evidence, not executable code.

**Timing and cancellation.** Select a 0.5, 1, or 2 second target interval. A single
request loop shares time across active sources; provider latency and multiple
sources reduce each source's actual rate. Only actual playing frames are sampled,
not storyboard reference images or the held final frame. A paused/hidden tab does
not submit scene checks. Rule edits, shot changes, and stopping monitoring discard
in-flight results. Three consecutive provider errors turn monitoring off. **Stop
playback** also cancels local generation work, clears queued directions, and stops
autoplay. Work already submitted to the model may finish on the server.

Live monitoring sends 640 × 360 JPEG frames (current and, when available, previous)
or bounded feed text through `/api/observe` to OpenAI Decisions only, using
`gpt-6-luna` and typed predicate questions. There is no alternate monitoring
provider or fallback. Raw keys stay on the server. See the
[OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions).
Model accuracy and end-to-end latency need evaluation on your actual scenes. The routes follow this example's
existing local-demo auth setup; gate access before exposing provider-backed routes
on a public deployment.

Each trigger also dispatches a browser event for custom integrations:

```js
window.addEventListener('previs:trigger', ({ detail }) => {
  console.log(detail.ruleId, detail.shotId, detail.seconds, detail.probability);
  // Call your own cue, overlay, or scene-control code here.
});
```

## How it works

The studio is in `app/page.tsx`; the pieces worth knowing about:

**Pictures.** `lib/reference-kit.ts` lists each story's pictures. On `ready`
the app shrinks each one to 768px, calls `uploadFile`, and caches the
`FileRef` for the session. A shot's `refs` array is the upload order, which is
the `<Picture N>` numbering in its prompt.

**Prompts.** `lib/storyboard.ts`. Each shot is built from shared binding lines
(`<Subject 1> is the courier in <Picture 1>, with ...`) so a character is
described the same way in every shot, then one `In one shot,` beat, then
`Camera:`, `Dialogue:`, `Soundscape:`, `Music:`. This is the format from the
model's prompt guide. Dialogue is one short line per clip. Every shot says
where it takes place, otherwise the model will happily move the action
indoors between cuts.

**The sequence.** `runSequence` sends `set_autoplay` and
`set_flush_on_clip_end: false`, then enqueues every shot. A shot marked
`connected` waits for its predecessor's `clip_generated` so it can pass
`continue_from_clip_id`. The model starts each clip when the previous one
ends and holds the last frame if the next isn't ready. The app listens for
`clip_started` / `clip_finished` to move the UI and the recorders along, and
reads `build_seconds` off `queue_update` for the "generated in" counter.

**Recording.** One `MediaRecorder` across the whole sequence plus one per
shot, all on a stream built from the `main_video` and `main_audio` tracks.
Takes go into IndexedDB. Replay uses two `<video>` elements so the next file
is loaded before the current one ends.

**Continuing.** `continueStory` takes the binding lines from the previous
shot, adds a sentence about what carries over, appends the new action, and
enqueues with `continue_mode: "frame"`. If the model refuses (for instance the
source clip aged out of the eight it keeps) it retries with `latent`. A refusal
arrives on the call that sent the command: `lib/command.ts` reads it there and
throws the model's reason. The
session log at the bottom shows every upload, enqueue, clip event, and the
assembled prompt.

## Layout

```
app/page.tsx                the studio
app/api/token/route.ts      API key -> session JWT
app/api/direct/route.ts     optional beat + brief + observations -> direction via Claude
app/api/observe/route.ts    server-side OpenAI Decisions adapter
components/live-monitor.tsx  rule editor, sampling, webcam and feed inputs
lib/live-rules.ts           rule validation, trigger latch, and event memory
lib/decisions.ts            OpenAI predicate requests and response normalization
lib/director.ts             bounded director context and prompt
lib/reactive.ts             generation/playback feedback boundary
tests/*.test.mjs           trigger, cancellation, provider and route tests
components/generation-overlay.tsx
lib/storyboard.ts           the two scripts
lib/reference-kit.ts        pictures per story
lib/recordings.ts           IndexedDB for takes
lib/image.ts                shrink before upload
lib/command.ts              send a command, throw its refusal
public/references/          pictures + the prompts that made them
public/boards/PROMPTS.md    prompts for drawn panels, if you want them
```

## Changing things

Shots: edit `lib/storyboard.ts`. One beat per shot, pictures referenced by
position in `refs`, lengths from the model's allowed set (5.875, 8, 10.125,
12.25, 14.375), spoken lines short.

Cast: drop JPEGs in `public/references/`, add them to `referenceKit` with the
story id, reference them from shots. The existing image prompts show how much
wardrobe and face detail it takes to keep someone consistent.

Director: the system prompt in `lib/director.ts`.

## Problems

- Empty page: you're on `127.0.0.1`. Use `localhost`.
- No sound: the browser blocked autoplay. The app starts audio on your next
  click and says so in the log.
- Slow "uploading pictures": that's your uplink to the model's region. The
  log prints KB/s per picture.
- Continuation refused: the log has the model's reason. It falls back to
  latent on its own; if that fails too, use Cut.
- Title text garbled: video models are bad at text. The title shots are
  short; take another.

## Checking changes

```bash
npm run typecheck
npm test            # Node 22.18+ or 24; built-in TypeScript stripping, no extra test runner
npm run build
```

The automated tests use synthetic evidence and mocked provider responses. They
cover thresholds, duplicate suppression, invalid/refused decisions, API limits,
director context, and cancellation at every reactive generation boundary. They
do not measure live provider accuracy or GPU generation latency.

An optional browser check uses an isolated profile, a labelled synthetic recording,
a fake webcam, and intercepted provider responses. Start the production app on
port 3006, install Playwright separately if needed, and run:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
npm run start -- --hostname localhost --port 3006
# In another terminal:
npm run test:browser
```

`PREVIS_TEST_URL` changes the local test address. `PLAYWRIGHT_MODULE` and
`PLAYWRIGHT_EXECUTABLE_PATH` can point to an existing Playwright installation and
Chromium binary. The test checks replay pixels, one-shot triggers, cancellation,
text input, webcam release, output corrections, and the stop action. Screenshots
are saved under `/tmp/previs-live-rules*.png`.
