# Helios Interactive

A Next.js + TypeScript reference frontend for [**Helios**](https://reactor.inc) — Reactor's real-time, prompt-driven video generation model.

Connect, send a prompt, and watch the model produce a continuous video stream you can steer mid-flight. Start from a curated text prompt, an example image, or your own image. Hot-swap prompts to evolve the scene without ever restarting. The whole thing is built on the typed [`@reactor-models/helios`](https://www.npmjs.com/package/@reactor-models/helios) SDK.

```
┌──────────────────────┬─────────────────────────────────────┐
│  Status   ▸ ready    │                                     │
│                      │                                     │
│  Try a prompt        │                                     │
│  ┌────────┬────────┐ │         live video output           │
│  │ Leo    │ Rain   │ │         (HeliosMainVideoView)       │
│  └────────┴────────┘ │                                     │
│  ┌────────┬────────┐ │                                     │
│  │ Flower │ Max    │ │                                     │
│  └────────┴────────┘ │                                     │
│                      │                                     │
│  Or start from image │                                     │
│  ┌────────┬────────┐ │                                     │
│  │ 🐶     │ 🐱     │ │                                     │
│  └────────┴────────┘ │                                     │
│  [Upload your own]   │                                     │
└──────────────────────┴─────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=helios` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

You'll need a Reactor API key — grab one at [reactor.inc/account/api-keys](https://www.reactor.inc/account/api-keys). It starts with `rk_`.

```bash
cp .env.example .env
# add your key: REACTOR_API_KEY=rk_...

pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), click **Connect**, and pick a starting point.

## What you can do with it

- **Start a scene from a text prompt.** Four curated prompt presets in the sidebar, plus a free-text input. One click and the model begins generating.
- **Start a scene from an image.** Two example images come with the app (a puppy in a village; a cat dancing in a bar with a boombox); each pairs with a hand-tuned prompt. Or upload your own image — the model picks a continuous scene built around it.
- **Evolve the scene mid-stream.** Once a curated scene is playing, the live panel offers one-click prompt continuations that keep the subject and camera intact while changing what happens. The model picks them up on the next chunk — no restart, no flash.
- **Snap a clip.** A "Capture" panel sits at the bottom of the sidebar. Click once and the SDK grabs the last 10 seconds of the live stream, opens a preview modal, and offers an MP4 download — no recording stack to wire up, no extra services.
- **Pause / Resume / Reset.** Real-time transport controls. Reset clears the session and brings the setup panel back.

## Architecture at a glance

The sidebar UI has two phases driven by the model's `state` snapshot:

| Phase     | When                          | What's visible                                                                        |
| --------- | ----------------------------- | ------------------------------------------------------------------------------------- |
| **Setup** | before generation has started | preset prompts, example images, custom upload, free-text textarea                     |
| **Live**  | while generating or paused    | active prompt, frame/chunk counter, Pause / Resume / Reset, curated evolution prompts |

Each component subscribes to the snapshot itself and self-hides when it's not in its phase. No central orchestrator.

## Code tour

The interesting bits, in roughly the order you'd read them:

| File                                                                     | What's in it                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`app/page.tsx`](app/page.tsx)                                           | Server Component. Checks `REACTOR_API_KEY` is set, otherwise renders [`SetupRequired.tsx`](app/SetupRequired.tsx).                                                                                                                                                                                                                    |
| [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts)       | GET route that mints a session-scoped Reactor JWT (pinned to `reactor/helios` via `authorization_details`) and returns it with its `expires_at`. The response is `no-store`; the client memoizes the token itself, because a session may only be operated by the token that created it.                                               |
| [`app/HeliosApp.tsx`](app/HeliosApp.tsx)                                 | First `"use client"` boundary. Wires `<HeliosProvider jwtToken={fetchToken}>` with a resolver that memoizes the minted token until shortly before it expires, lays out the sidebar + video pane.                                                                                                                                      |
| [`app/lib/prompts.ts`](app/lib/prompts.ts)                               | The scene library. Every prompt the app suggests — both starting prompts and mid-stream evolutions — lives here. Same source feeds the setup-phase presets, the example image cards, and the live-phase evolution picker.                                                                                                             |
| [`app/components/PromptComposer.tsx`](app/components/PromptComposer.tsx) | Setup phase. Preset prompts + free-text input → `setPrompt` + `start`.                                                                                                                                                                                                                                                                |
| [`app/components/ImageStarter.tsx`](app/components/ImageStarter.tsx)     | Setup phase. Curated image scenes call `uploadFile` → `setConditioning({ prompt, image })` → `start` — `setConditioning` (Helios 0.9.0+) commits both pieces of conditioning atomically, so the first chunk is guaranteed to include both. Custom uploads just call `setImage`; the user's prompt arrives later via `PromptComposer`. |
| [`app/components/NowPlaying.tsx`](app/components/NowPlaying.tsx)         | Live phase. Current prompt, chunk/frame counter, transport controls.                                                                                                                                                                                                                                                                  |
| [`app/components/EvolveScene.tsx`](app/components/EvolveScene.tsx)       | Live phase. Matches the active prompt against the scene library and renders that scene's evolutions as one-click hot-swaps. `setPrompt` only — no restart.                                                                                                                                                                            |
| [`app/components/SnapClip.tsx`](app/components/SnapClip.tsx)             | Model-agnostic. Captures the last N seconds of the live stream via `reactor.requestClip(...)` and opens a preview modal with the SDK's `<ClipPlayer>` + `<ClipDownloadButton>`. Drop-in for any Reactor example.                                                                                                                      |
| [`app/components/StatusBadge.tsx`](app/components/StatusBadge.tsx)       | The four-state connection lifecycle (`disconnected → connecting → waiting → ready`) plus Connect / Disconnect.                                                                                                                                                                                                                        |
| [`app/components/CommandError.tsx`](app/components/CommandError.tsx)     | Surfaces `command_error` messages from the model so failed preconditions are never silent.                                                                                                                                                                                                                                            |
| [`app/components/Video.tsx`](app/components/Video.tsx)                   | One line: `<HeliosMainVideoView />`. The SDK component handles the `<video>` element, the `srcObject` wiring, and browser autoplay policies.                                                                                                                                                                                          |

## Going further

For the full design rationale, prompt-engineering rules, and every gotcha the app exists to teach, read **[`skill/SKILL.md`](skill/SKILL.md)** — the SDK guide you can hand to an AI agent (or a human) to scaffold their own Helios frontend on the same patterns.

A few things this demo deliberately leaves out so the patterns stay clean:

- **Mid-stream image swap.** The model supports it; the demo only exposes mid-stream prompt switching.
- **Custom-prompt evolutions.** If the user types a free-text prompt, the live panel skips evolution suggestions (those only appear for curated scenes from the library).
- **Tuning knobs** — `set_seed`, `set_sr_scale`, `set_image_strength`, `schedule_prompt`. Each is one extra event method on `useHelios()` and a small control component.

Extending is mostly additive — drop new components into the sidebar phases, or add new scenes to `app/lib/prompts.ts` and they show up everywhere.

## Tech stack

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · [`@reactor-models/helios`](https://www.npmjs.com/package/@reactor-models/helios) · [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk) (recording primitives) · [`@reactor-team/ui`](https://www.npmjs.com/package/@reactor-team/ui) (design tokens only) · [`hls.js`](https://www.npmjs.com/package/hls.js) (Chromium/Firefox clip preview)
