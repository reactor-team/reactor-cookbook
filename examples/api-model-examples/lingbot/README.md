# Lingbot Interactive

A Next.js + TypeScript reference frontend for [**Lingbot**](https://reactor.inc) — Reactor's real-time interactive world model.

Pick a starting image, watch it come alive as a continuous video stream, and drive the scene with WASD. Lingbot is image-to-video: it anchors the generation to a reference image, then accepts realtime movement and camera commands that flow into the output a fraction of a second later. The whole thing is built on the typed [`@reactor-models/lingbot`](https://www.npmjs.com/package/@reactor-models/lingbot) SDK.

```
┌──────────────────────┬─────────────────────────────────────┐
│  Status   ▸ ready    │                                     │
│                      │                                     │
│  Now playing · run   │                                     │
│  chunk 12 · w+left   │                                     │
│  [Pause]   [Reset]   │         live video output           │
│                      │         (LingbotMainVideoView)      │
│  Drive the scene     │                                     │
│         W            │                                     │
│      A  ·  D         │                                     │
│         S            │                                     │
│      ←     →   (yaw) │                                     │
│      ↑  ·  ↓   (pitch│                                     │
│  Rotation: ▬▬▬▬▬─    │                                     │
│                      │                                     │
└──────────────────────┴─────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=lingbot` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

You'll need a Reactor API key — grab one at [reactor.inc/account/api-keys](https://www.reactor.inc/account/api-keys). It starts with `rk_`.

```bash
cp .env.example .env
# add your key: REACTOR_API_KEY=rk_...

pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), click **Connect**, pick a scene, and drive.

## What you can do with it

- **Start from a curated scene.** Five hand-tuned scenes ship in the sidebar — a dragon flying over a jungle castle, a golden retriever in a watercolor meadow, a horseback ride through a misted kingdom, a vintage 4x4 in a desert canyon, a work boat punching through dark seas. Each one bundles a reference image and a paragraph-length prompt that frames both the subject and the camera. One click, and the model begins generating.
- **Bring your own.** Upload any image, write any paragraph-length prompt, and start. The same fast path the curated scenes use.
- **Drive the scene with WASD.** Once it's live, the on-screen pad (and the global keyboard listener) sends `set_movement`, `set_look_horizontal`, and `set_look_vertical` commands. The model picks them up at chunk boundaries and the output video reflects them a fraction of a second later. Arrow keys turn the subject, a rotation-speed slider tweaks how fast.
- **Change the world on the fly.** A "World events" panel sits alongside the WASD pad — six curated atmospheric events (rain, snow, fog, sunset, night, storm). One click hot-swaps the prompt mid-stream via `set_prompt`; the model picks it up on the next chunk and the scene visibly shifts without restarting or losing the reference image. Re-click to revert to the base scene.
- **Snap a clip.** A "Capture" panel sits at the bottom of the sidebar. Click once and the SDK grabs the last 10 seconds of the live stream, opens a preview modal, and offers an MP4 download — no recording stack to wire up, no extra services.
- **Pause / Resume / Reset.** Real-time transport controls. Reset clears the session and brings the setup panel back.

## Architecture at a glance

The sidebar UI has two phases driven by the model's `state` snapshot:

| Phase     | When                          | What's visible                                                                                                                    |
| --------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Setup** | before generation has started | scene picker, custom-upload + custom-prompt                                                                                       |
| **Live**  | while generating or paused    | active prompt, chunk / action counter, Pause / Resume / Reset, the WASD + look pad, rotation-speed slider, the World events panel |

Each component subscribes to the snapshot itself and self-hides when it's not in its phase. No central orchestrator.

## Code tour

The interesting bits, in roughly the order you'd read them:

| File                                                                         | What's in it                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`app/page.tsx`](app/page.tsx)                                               | Server Component. Checks `REACTOR_API_KEY` is set, otherwise renders [`SetupRequired.tsx`](app/SetupRequired.tsx).                                                                                                                                                                       |
| [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts)           | GET route that mints a session-scoped Reactor JWT (pinned to `reactor/lingbot` via `authorization_details`) and returns it with its `expires_at`. The response is `no-store`; the client memoizes the token itself, because a session may only be operated by the token that created it. |
| [`app/LingbotApp.tsx`](app/LingbotApp.tsx)                                   | First `"use client"` boundary. Wires `<LingbotProvider jwtToken={fetchToken}>` with a resolver that memoizes the minted token until shortly before it expires, lays out the sidebar + video pane.                                                                                        |
| [`app/lib/scenes.ts`](app/lib/scenes.ts)                                     | The scene library. Every curated starting point — image + paragraph prompt — lives here. Adding a new scene is one entry plus an image in `public/images/`.                                                                                                                              |
| [`app/lib/dynamic-events.ts`](app/lib/dynamic-events.ts)                     | Curated atmospheric / weather events the live-phase panel can throw at the scene. Each entry is one sentence describing an environmental change. Adding a new world event is one entry — no component changes.                                                                           |
| [`app/components/ScenePicker.tsx`](app/components/ScenePicker.tsx)           | Setup phase. Curated scene cards → `uploadFile` → `setImage` → `setPrompt` → `start`. Awaiting `setImage` waits for its `image_accepted` reply, so the first chunk includes the image conditioning.                                                                                      |
| [`app/components/CustomStart.tsx`](app/components/CustomStart.tsx)           | Setup phase. Custom image upload + free-text prompt → `start`.                                                                                                                                                                                                                           |
| [`app/components/NowPlaying.tsx`](app/components/NowPlaying.tsx)             | Live phase. Current prompt, chunk / `current_action` counter, transport controls.                                                                                                                                                                                                        |
| [`app/components/MovementControls.tsx`](app/components/MovementControls.tsx) | Live phase, signature feature. On-screen WASD + look pad, global keyboard listener, rotation-speed slider. State-based axes that return to `"idle"` on release.                                                                                                                          |
| [`app/components/DynamicEvents.tsx`](app/components/DynamicEvents.tsx)       | Live phase. Captures the session's "base prompt" on the first `started` snapshot, then appends a curated event sentence on click and re-sends via `setPrompt`. Single-active toggle — re-click reverts to the base.                                                                      |
| [`app/components/SnapClip.tsx`](app/components/SnapClip.tsx)                 | Model-agnostic. Captures the last N seconds of the live stream via `reactor.requestClip(...)` and opens a preview modal with the SDK's `<ClipPlayer>` + `<ClipDownloadButton>`. Drop-in for any Reactor example.                                                                         |
| [`app/components/StatusBadge.tsx`](app/components/StatusBadge.tsx)           | The four-state connection lifecycle (`disconnected → connecting → waiting → ready`) plus Connect / Disconnect.                                                                                                                                                                           |
| [`app/components/CommandError.tsx`](app/components/CommandError.tsx)         | Surfaces `command_error` messages from the model so failed preconditions are never silent.                                                                                                                                                                                               |
| [`app/components/Video.tsx`](app/components/Video.tsx)                       | One line: `<LingbotMainVideoView />`. The SDK component handles the `<video>` element, the `srcObject` wiring, and browser autoplay policies.                                                                                                                                            |

## Going further

For the full design rationale, prompt-engineering rules, and every gotcha the app exists to teach, read **[`skill/SKILL.md`](skill/SKILL.md)** — the SDK guide you can hand to an AI agent (or a human) to scaffold their own Lingbot frontend on the same patterns.

A few things this demo deliberately leaves out so the patterns stay clean:

- **Seed control.** `useLingbot().setSeed({ seed })` is one extra Setup-phase component.
- **Free-form mid-stream prompt textarea.** `DynamicEvents` ships a curated picker; if you want to let the user type any prompt mid-flight, drop a textarea next to it that calls `setPrompt` with the user's text composed onto the captured base prompt.
- **Scheduled prompt changes.** No chunk-level schedule is built in; emulate it by reacting to `useLingbotChunkComplete` and sending the next `setPrompt` when the target chunk fires.
- **Gamepad / stick input.** Same shape as the keyboard handler — fire-and-forget typed methods on press, `"idle"` on release.

Extending is mostly additive — drop new components into the sidebar phases, or add new scenes to `app/lib/scenes.ts` (or new world events to `app/lib/dynamic-events.ts`) and they show up automatically.

## Tech stack

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · [`@reactor-models/lingbot`](https://www.npmjs.com/package/@reactor-models/lingbot) · [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk) (recording primitives) · [`@reactor-team/ui`](https://www.npmjs.com/package/@reactor-team/ui) (design tokens only) · [`hls.js`](https://www.npmjs.com/package/hls.js) (Chromium/Firefox clip preview)
