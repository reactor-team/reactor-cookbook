# Previs: a storyboard that generates itself

Write a scene as a handful of shots, pin the cast, the location, and the props
with reference pictures, and press one button. The sequence plays back as a
live stream with synchronized sound and the same faces in every shot, faster
than it takes to watch. When it ends, type what happens next and the picture
continues from the frame it stopped on.

Built on [`reactor/fast-h3-omni`](https://docs.reactor.inc/model-api-reference/fast-h3-omni/overview)
and the [Reactor JavaScript SDK](https://docs.reactor.inc/sdk-reference/reactor-class).

![Previs with the Desert signal storyboard loaded](./public/screenshot.jpg)

## What it shows

| Capability | Where it lives |
| - | - |
| Cast, location, and prop consistency across shots with `reference_images`, named `<Picture 1>`, `<Picture 2>`, … | `lib/storyboard.ts`, `lib/reference-kit.ts` |
| Prompts in the model's own format: subject bindings, one `In one shot,` beat, `Camera:`, `Dialogue:`, `Soundscape:`, `Music:` | `lib/storyboard.ts` |
| Spoken lines with lip sync from the prompt alone | the `Dialogue:` lines in each script |
| A whole sequence through the model's playout queue, so each shot starts as the previous one ends while later shots are still building | `runSequence` in `app/page.tsx` |
| A "generated in" counter from `build_seconds` | `handleClip` in `app/page.tsx` |
| Continuation from the last displayed frame with `continue_from_clip_id` and `continue_mode: "frame"` | `continueStory` in `app/page.tsx` |
| Pictures uploaded once per session and reused by `FileRef` | `uploadAsset` in `app/page.tsx` |
| Recording the live WebRTC stream for offline replay | `lib/recordings.ts` |
| Server-side key exchange for a session-scoped JWT | `app/api/token/route.ts` |
| Optional: a one-line beat expanded into full direction by Claude | `app/api/direct/route.ts` |

Two stories are bundled. **Desert signal** is a nine-shot science-fiction short:
a traveler, an abandoned radio station, a call, an answer, and a rescue. **Dead
drop** is a six-shot thriller in a library. Every picture under
`public/references/` is generated and fictional; the prompts that produced
them are beside them in `prompts.json` and `drop-prompts.json`.

## Requirements

- Node.js 20.9 or newer
- A WebRTC-capable browser (current Chrome, Edge, Firefox, or Safari)
- A Reactor API key with access to `reactor/fast-h3-omni`. Create one under
  **API Keys** in the [Reactor dashboard](https://reactor.inc/dashboard).

## Quick start

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/examples/storyboard-previs
cp .env.example .env.local      # then add your key, see below
npm install
npm run dev
```

Open <http://localhost:3000>. Use `localhost`, not `127.0.0.1`: the Next.js
dev server only serves its client bundle to its own origin.

Then:

1. **Connect model.** The app mints a session token, connects over WebRTC, and
   uploads the story's pictures. The transport bar reads "3 pictures ready"
   when it is done, a few seconds on a normal connection.
2. **Generate storyboard.** Every shot is queued at once. The model plays each
   clip as soon as it is built; the shot list shows queued, ready, and built
   times as it goes. The first clip is the title sequence.
3. **What happens next?** When the sequence ends the picture holds on its last
   frame and a prompt bar appears over the monitor. Type a beat, choose
   **Continue** (the next clip starts from this exact frame) or **Cut** (a new
   shot in the same scene), and press **Play it**.
4. **Play sequence** replays the saved recording with the exact live cuts,
   with the model connected or not. **Present** fills the screen with the
   monitor.

Added beats appear in the shot list under "Added live" and are not part of the
scripted timeline; **Clear added** removes them. **Reset script** restores a
story's original shots. Each story's edits are saved separately in the
browser.

## Environment variables

Copy `.env.example` to `.env.local`. The file is read only by the route
handlers under `app/api/` and never reaches the browser. Do not commit it.

| Variable | Required | Purpose |
| - | - | - |
| `REACTOR_API_KEY` | Yes | Exchanged by `app/api/token/route.ts` for a short-lived JWT limited to one `reactor/fast-h3-omni` session. The browser only ever sees the JWT. |
| `ANTHROPIC_API_KEY` | No | Enables the director in `app/api/direct/route.ts`, which turns the beat you type into full shot direction. Without it the beat is sent to the model as written. |
| `DIRECTOR_MODEL` | No | Claude model for the director. Defaults to `claude-haiku-5-5`. |

Restart `npm run dev` after changing the file.

## How it works

**Pictures.** `lib/reference-kit.ts` lists each story's pictures. On `ready`,
every picture a story uses is shrunk to 768 px on the long edge, uploaded with
`uploadFile`, and cached by id for the session. A shot's `refs` array is the
upload order, so it is also the `<Picture N>` numbering in that shot's prompt.
Keeping uploads small matters: a venue connection can be slow, and the model
only uses references for appearance.

**Prompts.** `lib/storyboard.ts` holds both scripts. A shot's prompt is built
from shared binding lines, so a character is described identically in every
shot she appears in, followed by one `In one shot,` beat and labeled `Camera:`,
`Dialogue:`, `Soundscape:`, and `Music:` lines. Dialogue is one short quoted
line per clip, marked "slowly and clearly". Every shot also states where it
takes place, so the model does not move the action indoors or change the
camera's side of the room between cuts.

**The sequence.** Generate storyboard sends `set_autoplay` and
`set_flush_on_clip_end: false`, then enqueues every shot in order. A shot with
`connected: true` waits only for its predecessor's `clip_generated` before
enqueueing with `continue_from_clip_id`. The model starts each ready clip as
the previous one ends and holds the last frame if the next is not ready yet.
The app reads `clip_started` and `clip_finished` to drive the UI and the
recorders, and `build_seconds` from `queue_update` for the speed counter.

**Recording.** One `MediaRecorder` runs across the whole sequence and one per
shot, all on a `MediaStream` made from the `main_video` and `main_audio`
tracks. Takes are stored in IndexedDB. Play sequence stitches the sequence
file with any added beats and plays them through two video elements so the
next file is loaded before the current one ends.

**Continuing.** The app rebuilds the prompt from the previous shot's binding
lines, adds a sentence stating what carries over, and appends the new action.
It sends `continue_mode: "frame"` so the new clip's first frame is the frame on
screen. If the model refuses, it retries in `latent` mode and logs the reason.
The session log at the bottom of the monitor shows every upload, enqueue,
clip event, and the full assembled direction.

## Project structure

```
storyboard-previs/
  app/
    page.tsx               # the whole studio: session, queue, recorders, UI
    layout.tsx
    globals.css
    api/token/route.ts     # API key -> session-scoped JWT
    api/direct/route.ts    # optional: beat -> full direction with Claude
  components/
    generation-overlay.tsx # the loading and "generating next" treatment
  lib/
    storyboard.ts          # the two stories and their prompts
    reference-kit.ts       # which pictures each story uses
    recordings.ts          # IndexedDB storage for takes
    image.ts               # client-side shrink before upload
  public/
    references/            # generated reference pictures and their prompts
    boards/PROMPTS.md      # prompts for hand-drawn storyboard panels
```

## Customize it

- **Change a story.** Edit the shot list in `lib/storyboard.ts`. Keep every
  shot to one beat that fits its length, name pictures by their position in
  `refs`, state where the shot takes place, and keep spoken lines under eight
  words. Lengths must be one of 5.875, 8, 10.125, 12.25, or 14.375 seconds.
- **Change the cast.** Drop JPEGs into `public/references/`, add them to
  `referenceKit` in `lib/reference-kit.ts` with the story they belong to, and
  reference them from the shots. The image prompts beside the bundled
  pictures show the level of detail that keeps a face and wardrobe stable.
- **Use real storyboard panels.** `public/boards/PROMPTS.md` has prompts for
  hand-drawn panels per shot; set a shot's `image` to the panel to show it
  before generation.
- **Tune the director.** The system prompt in `app/api/direct/route.ts`
  encodes the prompt-guide format and the continuation rules.

## Troubleshooting

- **Empty page at `127.0.0.1:3000`.** Use `http://localhost:3000`. Next.js
  blocks its dev client bundle for other origins.
- **No sound.** Browsers can block audio that starts without a click. The app
  starts audio on your next click and logs "Browser blocked audio" when that
  happens. Generate storyboard and Play it both count as clicks.
- **Uploads are slow.** Uploads go from your browser straight to object
  storage in the model's region. The app already shrinks pictures to about
  40 to 80 KB; a slow uplink shows in the session log as a low KB/s figure.
- **A continuation was refused.** The session log shows the model's reason,
  for example a source clip that is no longer among the eight it retains. The
  app retries in latent mode automatically; if that fails too, use Cut.
- **The title text is wrong.** Rendering text is the weakest thing video
  models do. The title shots are short, so New take is cheap.

## License

Apache-2.0, like the rest of this repository.
