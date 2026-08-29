# HappyOyster

A Next.js + TypeScript reference frontend for **HappyOyster**, a real-time interactive world model on Reactor. Build a world from a prompt (or attach one you built before), then travel it live: **Adventure** worlds you drive like a game with WASD, **Directing** worlds you steer with text instructions and pause / rewind transport.

```
┌─────────────────────────┬────────────────────────────────────┐
│  Featured worlds        │                                    │
│  ┌────────┬────────┐    │                                    │
│  │ Meadow │ City    │   │          live world video          │
│  ├────────┼────────┤    │                                    │
│  │ Forest │ Ruins   │   │                                    │
│  └────────┴────────┘    │                                    │
│  Compose your own       │                                    │
│  Attach by world id     │                                    │
│  ── while traveling ──  │                                    │
│  0:42 countdown         │                                    │
│  WASD · look · verbs    │                                    │
│  (Directing: instruct,  │                                    │
│   pause, rewind)        │                                    │
└─────────────────────────┴────────────────────────────────────┘
```

Everything model-specific runs through the typed
[`@reactor-models/happy-oyster`](https://www.npmjs.com/package/@reactor-models/happy-oyster)
package, which wraps the base
[`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk)
with a typed `connect → createWorld / attachWorld → startTravel` flow, live controls, and the `<HappyOysterVideo>` element the live world renders into.

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=happy-oyster` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

```bash
cp .env.example .env.local
# add your key: REACTOR_API_KEY=rk_...   (grab one at reactor.inc/account/api-keys)

pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), pick a featured world or compose your own, and the app connects a Reactor session, builds (or attaches) the world, and drops you into the live travel.

The API key never reaches the browser: the server route [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts) exchanges it for a short-lived JWT (see [docs.reactor.inc/authentication](https://docs.reactor.inc/authentication)), and the SDK re-fetches it (through the browser's HTTP cache) on every Reactor Platform call via the `getJwt` resolver.

## What you can do with it

- **Featured worlds.** Six curated worlds: four Adventure, two Directing. A world with a pinned id **attaches** instantly; the rest **create** from their prompt (a ~30s build).
- **Compose your own.** Free-text prompt, an Adventure/Directing mode toggle, an optional first-frame image upload (≤2MB), and the knobs that apply to the chosen mode: perspective for Adventure; resolution, camera motion, and narrative for Directing.
- **Attach by id.** Worlds are permanent; paste an `encrypted_world_id` you saved earlier (and pick its experience) to jump straight back in, no build.
- **Drive Adventure worlds.** WASD moves, arrows (or the on-screen pad) look, chords compose (W+A strafes, Shift+W sprints), and the world's advertised action verbs appear as buttons.
- **Steer Directing worlds.** Type instructions to steer the next scene, pause / resume, and rewind (multiples of 4s, while paused). The instruction timeline and auto-detected chapters render live.

## How it works

Each experience is its own Reactor model — `happy-oyster-adventure` and `happy-oyster-director` — so the **mode is chosen before connecting** and fixed for the life of the session. The composer (and the featured-world tiles) pick the mode; [`HappyOysterApp`](app/HappyOysterApp.tsx) mounts the provider on it, and switching experiences remounts a fresh session.

From there the flow is the typed SDK's linear lifecycle:

1. **`connect()`** opens the Reactor session and syncs the first `world_state` snapshot.
2. **`createWorld(params)` / `attachWorld(id)`** makes a world the session's current one — create builds a fresh world (~30s), attach reopens a permanent one (instant).
3. **`startTravel()`** begins streaming the live world into `<HappyOysterVideo>` and unlocks the controls.

The model owns all world state and broadcasts one authoritative `world_state` snapshot on every change (and a `travel_state` snapshot during travel). The app mirrors those snapshots and never derives world state locally, so the UI can't drift from the model. Adventure `hold()` / `interact()` and Directing `instruct()` / `pause()` / `rewind()` steer the live world.

## The typed SDK

Everything model-specific runs through the typed **`@reactor-models/happy-oyster`** package, imported at `@reactor-models/happy-oyster` (the plain-JS client and types) and `@reactor-models/happy-oyster/react` (the provider, hooks, and `<HappyOysterVideo>`). It wraps the base `@reactor-team/js-sdk` with the `connect → createWorld / attachWorld → startTravel` flow, the live controls, and the video element — so this app never touches the base SDK directly.

## Configuration

| Env var                        | Required   | What it does                                                                                                                                                     |
| ------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REACTOR_API_KEY`              | yes (live) | Server-side key exchanged for session JWTs by `app/api/reactor/token/route.ts`.                                                                                  |
| `NEXT_PUBLIC_REACTOR_API_URL`  | no         | Reactor API base URL. Defaults to `https://api.reactor.inc`.                                                                                                     |
| `NEXT_PUBLIC_HO_LOCAL_RUNTIME` | no         | Set to `1` to talk straight to a runtime-served model (adventure on `:8080`, directing on `:8081`), skipping the Reactor Platform: no `REACTOR_API_KEY`, no JWT. |

If `REACTOR_API_KEY` is missing, the app renders a friendly setup landing instead of erroring (see [`app/SetupRequired.tsx`](app/SetupRequired.tsx)).

## Code tour

| File                                                                                                                                  | What's in it                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`app/page.tsx`](app/page.tsx)                                                                                                        | Server Component gate: live app or the setup landing.                                                                                                                                               |
| [`app/HappyOysterApp.tsx`](app/HappyOysterApp.tsx)                                                                                    | The fixed shell: header on top, control sidebar beside the content screen. Owns the pending intent and its mode; mounts the client provider keyed on the mode.                                      |
| [`components/happy-oyster/ho-client.tsx`](components/happy-oyster/ho-client.tsx)                                                      | The `useHappyOysterClient()` surface adapting the live SDK. Start here.                                                                                                                             |
| [`components/happy-oyster/use-world-session.ts`](components/happy-oyster/use-world-session.ts)                                        | The session driver: walks a `WorldIntent` through connect → create/attach → auto-travel, phase-driven and StrictMode-safe.                                                                          |
| [`lib/view.ts`](lib/view.ts)                                                                                                          | The app's one reducer: SDK snapshot in, `AppView` out — plus the four-step loading journey the screen traces live.                                                                                  |
| [`components/happy-oyster/Sidebar.tsx`](components/happy-oyster/Sidebar.tsx)                                                          | The control rail, topped by the `StatusBadge` connection panel: browse surfaces, then the build card, ready card, travel deck (countdown + mode-matched controls), or error card as the view moves. |
| [`components/happy-oyster/Screen.tsx`](components/happy-oyster/Screen.tsx)                                                            | The content screen the travel video plays in: the journey pane while loading, then the live stream, then the end scene with the world id.                                                           |
| [`components/happy-oyster/Gallery.tsx`](components/happy-oyster/Gallery.tsx) + [`Composer.tsx`](components/happy-oyster/Composer.tsx) | The browse surfaces: featured worlds, custom compose (prompt, mode toggle, ≤2MB first-frame upload), and attach-by-id.                                                                              |
| [`components/happy-oyster/AdventureControls.tsx`](components/happy-oyster/AdventureControls.tsx)                                      | WASD + arrows + chords → `hold` / `interact` / `release`; world-advertised verbs.                                                                                                                   |
| [`components/happy-oyster/DirectingControls.tsx`](components/happy-oyster/DirectingControls.tsx)                                      | Text `instruct`, pause / resume / rewind transport, the instruction + chapter timeline.                                                                                                             |
| [`app/api/reactor/token/route.ts`](app/api/reactor/token/route.ts)                                                                    | Cacheable GET route that exchanges `REACTOR_API_KEY` for a short-lived JWT.                                                                                                                         |
| [`lib/worlds.ts`](lib/worlds.ts) + [`lib/featured-worlds.json`](lib/featured-worlds.json)                                             | Featured world data, the countdown lengths, and the `WorldIntent` type.                                                                                                                             |
| [`skill/SKILL.md`](skill/SKILL.md)                                                                                                    | The extension guide: the client surface, the lifecycle, the input models, auth, and every gotcha.                                                                                                   |

## Going further

Read [`skill/SKILL.md`](skill/SKILL.md) before extending. Deferred features you could add: touch/pointer control pads for mobile, snap-clip recording of the live stream, a shareable `?world=` deep link, and prompt upsampling in the composer.

## Tech stack

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · [`@reactor-models/happy-oyster`](https://www.npmjs.com/package/@reactor-models/happy-oyster) (typed HappyOyster SDK) · [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk) · [`@reactor-team/ui`](https://www.npmjs.com/package/@reactor-team/ui) (design tokens)
