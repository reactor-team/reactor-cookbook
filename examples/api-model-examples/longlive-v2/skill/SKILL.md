---
name: building-longlive-v2-frontends
description: Extend this cloned LongLive 2 example app — add controls, presets, timeline features, or directing affordances on top of `@reactor-models/longlive-v2` without breaking the patterns the existing code uses. Covers the SDK connection / events / messages model, the phase-based UI (setup composer vs live director), the shot-vs-cut grammar, the per-scene 48-chunk budget and how cuts extend length, scheduling against session_chunk, the storyboard store, and the read-only timeline.
---

# Building on this LongLive 2 app

This is a directorial reference frontend for LongLive 2, Reactor's multi-shot video model. Read this before extending it so you keep the patterns the code already uses.

## What LongLive 2 is

A continuous, autoregressive video model you direct like a storyboard. You open a scene with a **shot**, then transition with soft **shots** (same world, continuity preserved) and hard **cuts** (new scene, memory purged). Output is a single `main_video` track. Text-to-video — no reference-image input.

## The shot-vs-cut grammar (internalize this)

|              | **shot** (`set_shot`)                 | **cut** (`scene_cut`)           |
| ------------ | ------------------------------------- | ------------------------------- |
| Feel         | new beat, same world                  | clean break to a new scene      |
| Memory       | preserved                             | purged                          |
| Chunk budget | **spends** the current scene's budget | **resets** it (fresh 48 chunks) |
| Length       | does not extend                       | **extends** the video           |

## Chunks, scenes, and length

A chunk is 29 frames (~1.2s at 24fps). Two counters arrive on every `chunk_complete` / `state`:
`current_chunk` (per scene, resets on cut) and `session_chunk` (cumulative, never resets). **A scene auto-completes at 48 chunks (~58s).** To go longer, `scene_cut` to a new scene — that resets the per-scene budget. Scheduling fires against `session_chunk`; a beat scheduled past where its scene auto-completes never fires.

## The four concepts

| Concept                    | API                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Connection**             | `useLongliveV2()` → `status`, `connect()`, `disconnect()`. Four states: disconnected → connecting → waiting → ready.           |
| **Commands (you send)**    | `useLongliveV2()` → `setShot`, `sceneCut`, `scheduleShot`, `scheduleSceneCut`, `start`, `pause`, `resume`, `reset`, `setSeed`. |
| **Messages (you receive)** | `useLongliveV2State((msg) => …)`, `useLongliveV2CommandError((msg) => …)`.                                                     |
| **Tracks**                 | `<LongliveV2MainVideoView />` — pre-bound `<ReactorView track="main_video">`.                                                  |

## The UI phase model

Driven by the `state` snapshot's `started` flag:

- **Setup** (not started): `<Storyboard>` — compose the plan, then `Start` compiles it to `set_shot(opener)` → `schedule_shot` / `schedule_scene_cut(...)` → `start`.
- **Live** (started): `<NowPlaying>` (active prompt, `current_chunk`/48 budget, pause/resume/reset) + `<Director>` (fire/schedule shots & cuts).

Each component subscribes via `useLongliveV2State` and returns `null` when it's not its phase. `<Timeline>` shows in both. Clear the local snapshot on disconnect so a reconnect doesn't show stale data.

## The storyboard store

The authored plan is **client state**, not model state — it lives in `app/lib/storyboard-store.ts` (zustand), shared by the composer, the timeline, and the start action. The model's live position comes separately from `useLongliveV2State`. Keep that split: don't try to mirror model state into the store.

## Sending commands — rules

- **Status-gate every control.** Only send commands when `status === "ready"`.
- **A command resolves when the model's handler has finished**, and carries
  whatever that handler answered with. `setShot`, `scheduleShot`,
  `scheduleSceneCut`, `pause`, `resume` and `reset` answer with a message;
  `start`, `sceneCut` and `setSeed` answer with nothing, so awaiting those is a
  completion barrier and no more.
- **An answer reaches only the connection that asked.** It is correlated to its
  command and is **not** raised on the message event, so a `use*Accepted`-style
  subscription for one compiles, subscribes, and never fires — read the result off
  the awaited call. The `state` snapshot and `command_error` do broadcast.
- **None of them reject.** A refusal arrives as a broadcast `command_error` and
  resolves the call with `undefined`; so does a send that never completed, with the
  reason on `lastError`. `try/catch` is not how you detect a failed command.
- **The opener is `set_shot` + `start`.** Everything authored after the opener is scheduled (`schedule_*`) at an absolute `session_chunk`, then `start` runs them.
- **Live "now" beats** are `set_shot` / `scene_cut` (next boundary). **Scheduled** beats are `schedule_shot` / `schedule_scene_cut`.
- **No `unschedule` yet.** You can't move or cancel a scheduled beat once it's on the model — only `reset` clears everything. Compose before `start`, or fire live. (A future `unschedule` command would unlock live drag-editing.)

## Receiving messages

The `state` snapshot is the source of truth: `running`, `started`, `paused`, `current_chunk`, `session_chunk`, `current_prompt`, `seed`, `scheduled_shots`, `scheduled_scene_cuts`. Surface `command_error` (`<CommandError>`) so a rejected beat (empty prompt, wrong state, past chunk) is never silent.

## The timeline

`<Timeline>` is **read-only** here — scene dividers at cuts, beats as ticks, a playhead at `session_chunk`. The full **draggable** editor (resize scenes, drag beats) lives in the Reactor webapp playground; this example keeps it simple for readability. Adding drag is a good extension — gate edits to before `start` (the model's schedule can't be mutated live without `unschedule`).

## Capturing clips

`<SnapClip>` uses the base `@reactor-team/js-sdk` (`useReactor`, `ClipPlayer`, `ClipDownloadButton`) — recording is model-agnostic and not re-exported by the typed package. Drop it in unchanged for any model.

## Common mistakes

- Reaching for the base SDK for LongLive-specific calls — use the typed `useLongliveV2()` methods.
- Sending a command before `status === "ready"`.
- Scheduling a beat past a scene's 48-chunk ceiling with no earlier cut — it never fires.
- Using a `set_shot` for a true scene change (bleeds the old world) or a `scene_cut` for a small framing change (throws away continuity).
- Mirroring model state into the storyboard store — keep authored plan (store) and live position (`useLongliveV2State`) separate.
- Forgetting to clear the snapshot on disconnect.
- Single-line prompts — write full paragraphs (subject, action, setting, camera, light).

## Checklist for a new control

1. Decide its phase (setup vs live) and early-return on `snapshot.started`.
2. Status-gate every command on `status === "ready"`.
3. Use the typed `useLongliveV2()` methods; subscribe via `useLongliveV2State`.
4. Keep authored plan in the store, live position from the snapshot.
5. Surface failures via `command_error`.
6. Full-paragraph prompts; respect the 48-chunk-per-scene budget.
7. Brand colors via the `bg-brand` / `text-active` Tailwind tokens (from `@reactor-team/ui`).
