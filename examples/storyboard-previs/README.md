# storyboard-previs

This example is a previs tool for `reactor/fast-h3-omni`.

You write a scene as a list of shots. Each shot has a prompt, a length, and
the reference pictures it uses. The app uploads the pictures one time. Then it
sends all the shots to the model. The model plays the shots one after the
other as a live stream. It renders the later shots while the first shots play.
When the sequence stops, you type what happens next. The model continues from
the last frame.

![](./public/screenshot.jpg)

The example has two scripts:

- *Desert signal*: nine shots, science fiction.
- *Dead drop*: six shots, a thriller in a library.

The reference pictures are generated. They do not show real people or places.
The prompts that made them are in `public/references/prompts.json` and
`public/references/drop-prompts.json`.

## Before you start

You need:

- Node.js 20.9 or later.
- A browser with WebRTC. Current Chrome, Edge, Firefox, and Safari are all
  correct.
- A Reactor API key. Make one in the Reactor dashboard, under **API Keys**.

## Start the app

1. Go to the example folder.

   ```bash
   cd examples/storyboard-previs
   ```

2. Make the environment file and put your key in it.

   ```bash
   cp .env.example .env.local
   ```

3. Install the packages and start the server.

   ```bash
   npm install
   npm run dev
   ```

4. Open <http://localhost:3000> in your browser.

Note: Use `localhost`. Do not use `127.0.0.1`. The Next.js development server
sends its browser code only to its own origin. With `127.0.0.1` the page stays
empty.

## Use the app

1. Click **Connect model**. The app gets a session token and connects. It
   uploads the pictures of the current story. The bar below the monitor shows
   "pictures ready" when this is complete.
2. Click **Generate storyboard**. The app sends all the shots. The model plays
   each shot when it is ready. The shot list shows the state of each shot and
   the time it took to build.
3. When the sequence stops, the picture holds on the last frame. A text box
   shows on the monitor. Type what happens next. Select **Continue** to start
   the next clip from this frame. Select **Cut** to make a new shot in the same
   scene. Click **Play it**.
4. Click **Play sequence** to play the recording. This works with no
   connection to the model.
5. Click **Present** to show only the monitor, full screen.

The shots you add after the sequence are listed under "Added live". They are
not part of the timeline. Click **Clear added** to remove them. Click **Reset
script** to put the original shots of a story back. The app keeps the edits of
each story in the browser.

## Environment variables

The file `.env.local` holds the keys. Only the two route handlers in
`app/api/` read this file. The browser does not get the keys. Do not commit
this file.

| Variable | Required | What it does |
| - | - | - |
| `REACTOR_API_KEY` | Yes | `app/api/token/route.ts` exchanges it for a session token. The token permits one `fast-h3-omni` session. |
| `ANTHROPIC_API_KEY` | No | Turns on the director in `app/api/direct/route.ts`. The director changes the text you type into full shot direction. Without the key, the app sends your text as you typed it. |
| `DIRECTOR_MODEL` | No | The Claude model for the director. The default is `claude-haiku-5-5`. |

Restart `npm run dev` after you change this file.

## How the app works

All the application logic is in `app/page.tsx`. These are the parts to know.

### Pictures

`lib/reference-kit.ts` lists the pictures of each story. When the connection
is ready, the app makes each picture smaller (768 px on the long edge), sends
it with `uploadFile`, and keeps the `FileRef` for the session. The `refs` list
of a shot is the upload order. This order is also the `<Picture N>` number in
the prompt of that shot.

### Prompts

`lib/storyboard.ts` holds the two scripts. The app makes the prompt of a shot
from these parts, in this order:

1. Binding lines, one for each picture. Example:
   `<Subject 1> is the courier in <Picture 1>, with ...`. The scripts share
   these lines, so a person has the same description in each shot.
2. One `In one shot,` line that gives the action.
3. `Camera:`, `Dialogue:`, `Soundscape:`, and `Music:` lines.

This is the format from the prompt guide of the model. A dialogue line is
short, and a clip has only one. Each shot also says where it takes place. If
you do not say this, the model can move the action to a different place
between two shots.

### The sequence

`runSequence` sends `set_autoplay` and `set_flush_on_clip_end: false`. Then it
sends all the shots with `enqueue`. A shot with `connected: true` waits for
the `clip_generated` message of the shot before it. Then it sends
`continue_from_clip_id`. The model starts each clip when the one before it
stops. If the next clip is not ready, the model holds the last frame. The app
reads `clip_started` and `clip_finished` to update the screen and the
recorders. It reads `build_seconds` from `queue_update` for the "generated
in" counter.

### Commands and refusals

`lib/command.ts` sends each command and waits for the answer on the same
call. When the model refuses a command, `sendCommand` returns `undefined` and
the SDK sends an `error` event with `operation: "sendCommand"`. Only the
client that sent the command gets this event. `sendModelCommand` listens for
the event during the call and throws `CommandRefused` with the reason from the
model. The app does not wait for a `command_error` message.

### Recording

One `MediaRecorder` records the full sequence. One more records each shot.
All of them record the stream made from the `main_video` and `main_audio`
tracks. The app keeps the recordings in IndexedDB. Playback uses two `<video>`
elements. The second element loads the next file while the first one plays.

### Continuation

`continueStory` takes the binding lines from the shot before. It adds one
sentence that says what stays the same. It adds the new action. Then it sends
the shot with `continue_mode: "frame"`. If the model refuses the request, the
app sends it again with `continue_mode: "latent"`. One reason for a refusal:
the model keeps only the last eight clips, and the source clip is older.

The session log at the bottom of the monitor shows each upload, each
`enqueue`, each clip message, and the full prompt that the app sent.

## Files

```
app/page.tsx                   the application
app/api/token/route.ts         API key -> session token
app/api/direct/route.ts        the director (optional)
components/generation-overlay.tsx
lib/storyboard.ts              the two scripts
lib/reference-kit.ts           the pictures of each story
lib/command.ts                 sends a command and throws its refusal
lib/recordings.ts              IndexedDB storage for recordings
lib/image.ts                   makes pictures smaller before upload
public/references/             pictures and the prompts that made them
public/boards/PROMPTS.md       prompts for drawn storyboard panels
```

## Change the app

### Change a script

Edit `lib/storyboard.ts`. Follow these rules:

- Give each shot one action.
- Refer to a picture by its position in `refs`.
- Use one of these lengths: 5.875, 8, 10.125, 12.25, or 14.375 seconds.
- Keep a dialogue line short.
- Say where the shot takes place.

### Change the pictures

1. Put JPEG files in `public/references/`.
2. Add them to `referenceKit` in `lib/reference-kit.ts` with the story id.
3. Refer to them from the shots.

The prompts in `public/references/` show how much detail of the face and the
clothes you need to keep a person the same in each shot.

### Change the director

Edit the system prompt in `app/api/direct/route.ts`.

## Problems

| Problem | Cause | What to do |
| - | - | - |
| The page is empty. | You opened `127.0.0.1`. | Open `http://localhost:3000`. |
| There is no sound. | The browser blocked audio that started without a click. | Click anywhere. The app starts the audio. The session log shows "Browser blocked audio". |
| "Uploading pictures" is slow. | Your upload speed to the region of the model is low. | Wait. The session log shows the speed of each upload in KB/s. |
| A continuation is refused. | The session log shows the reason from the model. | The app tries again in `latent` mode. If this also fails, use **Cut**. |
| The title text is wrong. | Video models make errors in text. | Click **New take** on the title shot. The shot is short. |
