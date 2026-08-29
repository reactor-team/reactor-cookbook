# SANA-Streaming

A Next.js + TypeScript reference frontend for [**SANA-Streaming**](https://docs.reactor.inc/model-api-reference/sana-streaming/overview) - Reactor's real-time video **editing** model.

Where Reactor's other models generate video from a prompt, SANA-Streaming edits the video you bring: describe a change in plain text and the model makes that change while everything you don't mention carries through from the source untouched. Point your webcam at it and watch yourself transformed in real time, or pick a clip and stream it into the model the same way — frame for frame, edited and streamed back next to the source. Re-prompt mid-stream and the new edit lands at the next chunk boundary, about a second later. Built on the typed [`@reactor-models/sana-streaming`](https://www.npmjs.com/package/@reactor-models/sana-streaming) SDK, so every command, message, and track is checked at compile time.

```
┌──────────────────────┬───────────────────────────────────────┐
│  Status / errors     │                                       │
│  Input               │             edited output             │
│   • webcam self-view │   webcam mode shows just this; video    │
│   • or pick a video  │   mode puts your source clip on the    │
│   • start → pause/   │   left and the edit on the right       │
│     resume / reset   ├───────────────────────────────────────┤
│  Prompt + presets    │   status row (idle/streaming/paused    │
│  Snap clip           │            · chunk · prompt)           │
└──────────────────────┴───────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=sana-streaming` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

```bash
cp .env.example .env.local  # then add your REACTOR_API_KEY
pnpm install
pnpm dev                    # http://localhost:3000
```

Get a **production** API key (`rk_...`) from the [Reactor dashboard](https://reactor.inc) - the app targets `https://api.reactor.inc`. The key stays on the server (`REACTOR_API_KEY` is the only env var); the browser only ever sees a short-lived JWT minted by `app/api/reactor/token/route.ts`, scoped to `reactor/sana-streaming` sessions via `authorization_details`.

## What you can do with it

- **Webcam** - your webcam is published to the model's `camera` input track and transformed in real time. Edited frames come back on `main_video` at 1280 × 704 in 24-frame chunks, one every ~1-1.5s.
- **Video** - pick a preset clip or a local file. Instead of uploading it, the app **plays it and streams its frames into the same `camera` track**, so the model edits it on its live path — the source pane and the edited pane share one feed and can't drift apart.
- **Steer the prompt** - re-prompt mid-stream at any time; the new edit lands at the next chunk boundary, with no re-render and no break in the stream. Prompts are editing instructions, not scene descriptions - the [prompt guide](https://docs.reactor.inc/model-api-reference/sana-streaming/prompt-guide) covers how to write edits that land where you aim them.
- **Playback** - once the edit is running, the Input panel turns into pause / resume / reset controls. Set a seed before you start for reproducible results.
- **Snap a clip** - capture the last N seconds of the stream (model-agnostic recording).

## Architecture at a glance

The model is the **source of truth**: only `state` messages mutate local state, and the UI gates entirely off the reduced `SanaState` (`app/lib/state.ts`). Both input sources stream into the `camera` track, so generation is always the model's live path. Commands out: `set_prompt`, `set_seed`, `start`, `pause`, `resume`, `reset`. Tracks: input `camera`, output `main_video`. The full wire surface - every command, message, and the `state` payload - is documented in the [schema reference](https://docs.reactor.inc/model-api-reference/sana-streaming/schema).

> **Streaming a clip, not uploading it.** A selected video is played in a `<video>` element and captured with `captureStream()`; that track is published as `camera`. From v2.0.0 the model is live-only - the file path and its `set_mode` / `set_video` commands are gone from the schema - so starting is just `start`.

The app talks to the model through the typed SDK: `<SanaStreamingProvider jwtToken={fetchToken}>` (model name + tracks baked in), the `useSanaStreaming()` hook for status and one typed method per command (`setPrompt`, `start`, …), per-message hooks like `useSanaStreamingState` / `useSanaStreamingCommandError`, and `<SanaStreamingMainVideoView />` for output. The one exception is `app/components/SnapClip.tsx`, which uses `@reactor-team/js-sdk` directly for model-agnostic clip recording.

## Code tour

| Path                                   | What it is                                                                        |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `app/SanaStreamingApp.tsx`             | `SanaStreamingProvider` shell + layout + the typed message subscriptions          |
| `app/components/ModeInput.tsx`         | The Input panel — source toggle, webcam self-view / video picker, Start, Playback |
| `app/components/WebcamSource.tsx`      | Webcam capture + self-view (in the Input panel); produces the `camera` track      |
| `app/components/VideoSource.tsx`       | Plays a chosen clip and streams it into `camera` via `captureStream()` (stage)    |
| `app/components/VideoPicker.tsx`       | Preset + local-file picker for the video source                                   |
| `app/components/useCameraPublisher.ts` | Publishes whichever source track is current; unpublishes before publishing        |
| `app/components/Playback.tsx`          | Live-phase pause / resume / reset, shown in the Input panel once started          |
| `app/components/SeedField.tsx`         | Pre-start seed setting, shown in the Input panel's setup view                     |
| `app/components/Prompt.tsx`            | Prompt textarea + Apply, preset chips, active prompt                              |
| `app/components/Stage.tsx`             | Edited output — single in webcam mode; split with `your video` in video mode      |
| `app/lib/state.ts`                     | Reduces the typed `state` message into `SanaState`; the start flow                |
| `app/api/reactor/token/route.ts`       | Mints the short-lived, session-scoped JWT server-side                             |
| `app/components/SnapClip.tsx`          | Model-agnostic clip recording on `@reactor-team/js-sdk` (drop-in)                 |

## Going further

`skill/SKILL.md` documents the patterns this app uses - the typed-SDK surface, the connection and state model, how webcam and video both stream into the `camera` track, the manual camera publish with `contentHint = "detail"`, and how to extend the example. Point your coding agent at it when you build on top.

The published docs cover the model itself: the [overview](https://docs.reactor.inc/model-api-reference/sana-streaming/overview) for the conceptual model and quick start, the [schema](https://docs.reactor.inc/model-api-reference/sana-streaming/schema) for every command and message, the [prompt guide](https://docs.reactor.inc/model-api-reference/sana-streaming/prompt-guide) for writing edit instructions, and a [tutorial](https://docs.reactor.inc/model-api-reference/sana-streaming/tutorial) built around this app.

## Tech stack

Next.js 15 · React 19 · TypeScript · Tailwind v4 · `@reactor-models/sana-streaming` (typed SDK) · `@reactor-team/js-sdk` (recording primitives) · `@reactor-team/ui`
