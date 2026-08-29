# LongLive 2 Director

A Next.js + TypeScript reference frontend for [**LongLive 2**](https://reactor.inc) — Reactor's multi-shot video generation model.

Direct a video like a storyboard: set an opening shot, then compose **shots** (soft transitions that keep the scene) and **cuts** (hard transitions to a new scene) on a chunk timeline, press start, and watch the playhead fire your beats. Then keep directing live. Built on the typed [`@reactor-models/longlive-v2`](https://www.npmjs.com/package/@reactor-models/longlive-v2) SDK.

```
┌──────────────────────┬─────────────────────────────────────┐
│  Status / errors     │                                     │
│                      │            live video               │
│  Setup phase:        │                                     │
│   • Storyboard       │                                     │
│     (compose beats)  ├─────────────────────────────────────┤
│                      │  Timeline (scenes · beats · playhead)│
│  Live phase:         │                                     │
│   • Now playing      │                                     │
│   • Direct live      │                                     │
│  Snap clip           │                                     │
└──────────────────────┴─────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=longlive-v2` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

```bash
cp .env.example .env        # then add your REACTOR_API_KEY
pnpm install
pnpm dev                    # http://localhost:3000
```

Get an API key from the [Reactor dashboard](https://reactor.inc). It stays on the server — the browser only ever sees a short-lived JWT minted by `app/api/reactor/token/route.ts`, scoped to `reactor/longlive-v2` sessions via `authorization_details`.

## What you can do with it

- **Compose a storyboard** — set an opening shot, then add shots and cuts at chunk positions. Load a preset storyboard to see a full multi-shot sequence.
- **Understand chunks & length** — a scene runs up to 48 chunks (~58s) before it auto-completes; a **cut** resets that budget and extends the video, a **shot** spends it. The timeline and the "scene chunk N/48" readout make this visible.
- **Direct live** — once running, fire a shot or cut at the next chunk boundary, or schedule one ahead.
- **Snap a clip** — capture the last N seconds of the stream (model-agnostic recording).

## Architecture at a glance

The sidebar is **phase-driven** by the model's `state` snapshot (`snapshot.started`):

| Phase                 | Visible                       | What you do                                                                                                        |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Setup** (idle)      | `<Storyboard>`                | Compose the plan: opening shot + scheduled shots/cuts. "Start" compiles it to `set_shot` → `schedule_*` → `start`. |
| **Live** (generating) | `<NowPlaying>` + `<Director>` | Watch the playhead; fire or schedule shots/cuts in real time.                                                      |

`<Timeline>` (under the video) visualizes the plan and the playhead in both phases. `<SnapClip>` is model-agnostic and shows whenever the session is `ready`.

## Code tour

| Path                             | What it is                                             |
| -------------------------------- | ------------------------------------------------------ |
| `app/LongLiveApp.tsx`            | Provider + phase-driven layout                         |
| `app/components/Storyboard.tsx`  | Setup: compose beats, load presets, "Start storyboard" |
| `app/components/Timeline.tsx`    | Visual chunk timeline — scenes, beats, playhead        |
| `app/components/Director.tsx`    | Live: fire/schedule shots & cuts                       |
| `app/components/NowPlaying.tsx`  | Live: active prompt, scene budget, pause/resume/reset  |
| `app/lib/storyboard-store.ts`    | Zustand store for the authored storyboard              |
| `app/lib/prompts.ts`             | Preset storyboards (full multi-shot sequences)         |
| `app/api/reactor/token/route.ts` | Mints the short-lived, session-scoped JWT server-side  |
| `app/components/SnapClip.tsx`    | Model-agnostic clip recording (drop-in)                |

## Going further

`skill/SKILL.md` documents the patterns this app uses — the connection model, the shot-vs-cut grammar, the chunk/scene-length budget, scheduling, and how to extend the storyboard. Point your coding agent at it when you build on top.

Intentionally left out (good first extensions): a **draggable** timeline (this one is read-only — see the Reactor webapp playground for the full editor), **removing** a scheduled beat (needs an `unschedule` command — not in this release; `reset` clears all), and reference-image input (LongLive 2 is text-to-video).

## Tech stack

Next.js 15 · React 19 · TypeScript · Tailwind v4 · zustand · `@reactor-models/longlive-v2` · `@reactor-team/js-sdk` · `@reactor-team/ui` · hls.js
