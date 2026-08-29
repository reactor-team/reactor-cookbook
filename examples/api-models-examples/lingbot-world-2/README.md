# LingBot World 2

A Next.js + TypeScript reference frontend for **LingBot World 2** — a real-time interactive image-to-video world model on Reactor. Click an example scene, then drive the generated world like a game: WASD to move, arrows or mouse-look to turn, number keys to trigger world events, Space and C for jump and crouch — all while the model keeps streaming video.

Everything runs through the strongly-typed
[`@reactor-models/lingbot-world-2`](https://www.npmjs.com/package/@reactor-models/lingbot-world-2)
package, which wraps the base
[`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk)
with typed commands, message hooks, and the `main_video` track view.

```
┌──────────────────────┬─────────────────────────────────────┐
│  Status  ▸ ready     │                                     │
│                      │                                     │
│  Quick Start         │                                     │
│  ┌────────────┬───┐  │        live video output            │
│  │ Noir alley │ ✎ │  │    (LingbotWorld2MainVideoView)     │
│  ├────────────┼───┤  │                                     │
│  │ Horseman   │ ✎ │  │                                     │
│  ├────────────┼───┤  ├─────────────────────────────────────┤
│  │ Jet ski    │ ✎ │  │  [1 rain] [2 fog]   ← hold events   │
│  └────────────┴───┘  │   WASD  ◯ joystick  ←↑↓→  ◉ mouse   │
│  Custom scene        │   jump / crouch / orbit switches    │
└──────────────────────┴─────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=lingbot-world-2` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

You'll need a Reactor API key — grab one at [reactor.inc/account/api-keys](https://www.reactor.inc/account/api-keys). It starts with `rk_`.

```bash
cp .env.example .env.local
# add your key: REACTOR_API_KEY=rk_...

pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), click **Connect**, then click one of the Quick Start examples. It uploads the scene's starting image, sends its composed prompt, and starts generating; from there the controls are live.

The API key never reaches the browser: the server route [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts) exchanges it for a short-lived JWT scoped to `reactor/lingbot-world-2` sessions (via `authorization_details` — the token can only create sessions for that model and act on the sessions it created), and the SDK re-fetches it (via the browser's HTTP cache) on every API call through the `getJwt` resolver.

## What you can do with it

- **Quick Start examples.** Three curated scenes (a noir alley patrol, a battlefield horseman, a jet-ski cruise). One click uploads the image, sends the prompt, and starts.
- **Drive the world.** WASD (or the joystick) moves via `set_move_longitudinal` / `set_move_lateral`; arrows and click-to-engage mouse-look rotate via per-latent `set_camera_pose` deltas; Q/E roll; O toggles orbit (circle a point ahead instead of turning in place).
- **Trigger world events.** Each scene binds detail clauses to hold-keys 1–9 — hold to weave the event into the prompt, release to revert.
- **Jump and crouch** with selectable modes, from a simple prompt swap up to hand-editable per-step motion arcs (charge levels, dip patterns). The motion system is documented in [`skill/SKILL.md`](skill/SKILL.md).
- **Edit prompts live.** Click ✎ on any example to open the layered scene editor (base / camera / movement / events / vertical) — editing the running scene re-sends the prompt on the fly, and edits persist in `localStorage` until you press ↺. The **Show prompt** inspector (under Advanced) shows exactly what composed prompt the model is seeing and why.
- **Bring your own scene.** The Custom scene card takes your image plus a from-scratch layered prompt.
- **Backend knobs** (under Advanced): seed, rotation speed, DiT attention window, and KV-cache reset mode.
- **Snap a clip.** A "Capture" panel sits at the bottom of the sidebar. Click once and the SDK grabs the last 10 seconds of the live stream, opens a preview modal, and offers an MP4 download — no recording stack to wire up, no extra services.

## How the prompt is built

The model only ever sees a single prose string (`set_prompt`), but the app authors it in layers (`lib/lingbot-world-prompts.ts`):

- **base** — world identity: subject, environment, style
- **camera / movement** — each with `static` and `dynamic` variants, selected by whether you're currently moving
- **events** — hold-key detail clauses that stack while held
- **vertical** — the jump / crouch / stand sentence while those controls are engaged

`composePrompt()` flattens the active selection to prose and the controller re-sends it whenever the input state changes — so the text always matches the motion. The inspector panel visualizes this composition live.

## Configuration

| Env var                       | Required | What it does                                                                    |
| ----------------------------- | -------- | ------------------------------------------------------------------------------- |
| `REACTOR_API_KEY`             | yes      | Server-side key exchanged for session JWTs by `app/api/reactor/token/route.ts`. |
| `NEXT_PUBLIC_REACTOR_API_URL` | no       | Reactor API base URL. Defaults to `https://api.reactor.inc`.                    |

If `REACTOR_API_KEY` is missing, the app renders a friendly setup landing instead of erroring — see [`app/SetupRequired.tsx`](app/SetupRequired.tsx).

## Code tour

The interesting bits, in roughly the order you'd read them:

| File                                                                                                             | What's in it                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`app/page.tsx`](app/page.tsx)                                                                                   | Server Component gate: renders the app when `REACTOR_API_KEY` is set, the setup landing when it isn't.                                                                                                   |
| [`app/LingbotWorld2App.tsx`](app/LingbotWorld2App.tsx)                                                           | Client tree: the `getJwt` resolver, `<LingbotWorld2Provider>`, connection status bar, page layout.                                                                                                       |
| [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts)                                               | Cacheable GET route that exchanges `REACTOR_API_KEY` for a short-lived, session-scoped JWT (pinned to the model via `authorization_details`).                                                            |
| [`components/lingbot-world-2/LingbotWorldController.tsx`](components/lingbot-world-2/LingbotWorldController.tsx) | The heart of the app: message handling, WASD/look/jump/crouch input → typed SDK commands, prompt recomposition, the sidebar and control surfaces.                                                        |
| [`lib/lingbot-world-prompts.ts`](lib/lingbot-world-prompts.ts)                                                   | The layered `StructuredScene` model and the pure `composePrompt()`.                                                                                                                                      |
| [`lib/lingbot-cases-examples.ts`](lib/lingbot-cases-examples.ts) + [`lib/lingbot-cases/`](lib/lingbot-cases)     | The example scenes, one JSON per scene.                                                                                                                                                                  |
| [`components/lingbot-world-2/LayeredSceneEditor.tsx`](components/lingbot-world-2/LayeredSceneEditor.tsx)         | The full-screen layered prompt editor behind every ✎ button.                                                                                                                                             |
| [`components/lingbot-world-2/LivePromptInspector.tsx`](components/lingbot-world-2/LivePromptInspector.tsx)       | Read-only live view of the composed prompt with per-layer breakdown.                                                                                                                                     |
| [`components/lingbot-world-2/prompt-segments.ts`](components/lingbot-world-2/prompt-segments.ts)                 | Segment-level composition mirror that powers the inspector and editor preview.                                                                                                                           |
| [`components/SnapClip.tsx`](components/SnapClip.tsx)                                                             | Model-agnostic. Captures the last N seconds of the live stream via the SDK's `requestClip(...)` and opens a preview modal with `<ClipPlayer>` + `<ClipDownloadButton>`. Drop-in for any Reactor example. |
| [`skill/SKILL.md`](skill/SKILL.md)                                                                               | The extension guide, including the motion-system deep dive: the `camera_pose` contract, per-step arcs, symmetry, trigger semantics.                                                                      |

## Going further

[`skill/SKILL.md`](skill/SKILL.md) is the deep dive for extending this app — the SDK's connection / events / messages model, the camera-pose channel rules, the layered prompt system and its override store, auth, and clip capture. Read it before adding controls or scenes.

Deferred features you could add next: a gamepad binding, free-text mid-stream prompt entry, multi-clip capture galleries, scheduled world events.

## Learn more

- [Reactor Docs](https://docs.reactor.inc/overview)
- [`skill/SKILL.md`](skill/SKILL.md) — how input becomes motion, and every extension pattern, in depth

## Tech stack

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · [`@reactor-models/lingbot-world-2`](https://www.npmjs.com/package/@reactor-models/lingbot-world-2) · [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk) (recording primitives) · [`hls.js`](https://www.npmjs.com/package/hls.js) (clip preview) · [`@reactor-team/ui`](https://www.npmjs.com/package/@reactor-team/ui) (design tokens)
