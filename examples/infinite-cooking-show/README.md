# Infinite Cooking Show

A continuously directed cooking show built on
[`reactor/fast-h3`](https://docs.reactor.inc/model-api-reference/fast-h3/overview).
FastH3 generates synchronized video and audio into a live WebRTC track while
the app keeps its clip queue full, so Galton Ramshackle can keep cooking until
you stop the session.

![Galton Ramshackle, the bundled demo host](./public/characters/galton-ramshackle-moustache.png)

This example demonstrates how to:

- start the first clip from a selected still with `starting_frame`;
- build an infinite, seamless scene using Reactor's native
  `continue_from_clip_id` chain and autoplay queue;
- apply late prompts and ingredients only after clips that are already
  committed, without slowing or discarding the queue;
- turn image-based props into persistent textual scene instructions while
  FastH3 has no reference-image input;
- receive synchronized `main_video` and `main_audio` WebRTC tracks; and
- optionally use OpenAI GPT-5.6 Luna to plan the next beat and dialogue, with
  direct FastH3 prompting as the fallback.

The moustached Galton opening frame and all 16 cooking props are bundled demo
fixtures. User-added images remain in the local browser library.

## Run it

You need Node.js 20.9 or newer, a WebRTC-capable browser, and a Reactor API key.
Create a key from **API Keys** in the
[Reactor dashboard](https://reactor.inc/dashboard).

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/examples/infinite-cooking-show
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

Open [http://localhost:3000](http://localhost:3000). Live generation uses your
Reactor account and may incur usage charges.

## Direct the show

- Select a library image before starting to make it the opening frame. Galton
  Ramshackle is selected on a fresh load.
- Drag or click an ingredient to add its name and cooking instruction to every
  future prompt until you remove it. Already queued clips are never rewritten.
- Drop a library image on the live scene to turn its filename and saved prompt
  into a text-only scene cue.
- Type into the transparent stage prompt or use the microphone button to
  direct the next uncommitted clip.
- Press <kbd>M</kbd> to mute or unmute and <kbd>C</kbd> to collapse the settings.
- Choose a clip length and startup prebuffer before starting the session.

The selected image is only uploaded for the first clip. The app does not
extract final frames in the browser; continuations reference the preceding
Reactor clip ID directly.

## Optional story planning

Set `OPENAI_API_KEY` in `.env.local` to enable the optional low-latency story
planner. It uses GPT-5.6 Luna through the Responses API with reasoning disabled
and priority processing. Without the key, operator directions and prop
instructions go directly to FastH3, so the demo remains fully usable.

## Where to change things

- [`components/h3-studio.tsx`](./components/h3-studio.tsx) owns the session,
  queue scheduler, WebRTC tracks, live direction, and local image library.
- [`lib/cooking-props.ts`](./lib/cooking-props.ts) defines the bundled props and
  the textual action each adds to future prompts.
- [`lib/h3-contract.ts`](./lib/h3-contract.ts) contains the FastH3 model and
  queue message contract used by the client.
- [`app/api/reactor/token/route.ts`](./app/api/reactor/token/route.ts) exchanges
  the server-side API key for a short-lived token scoped to `reactor/fast-h3`.
- [`app/api/story/route.ts`](./app/api/story/route.ts) is the optional OpenAI
  planning proxy.

## Verify changes

```bash
npm run typecheck
npm run build
```

The build does not start a paid model session.
