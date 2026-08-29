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
  whatever that handler answered with:

| The model | Reaches you | LongLive 2 commands |
| --- | --- | --- |
| **answers** the command that asked | the awaited call's return value — and the **sending** connection's `message` event | `setShot`, `scheduleShot`, `scheduleSceneCut`, `pause`, `resume`, `reset` |
| **broadcasts** to every connection | the per-message hooks | `state`, `chunk_complete`, `command_error`, `scene_cut`, `generation_started` / `complete` |
| answers with **nothing** | the await resolves `undefined`; nothing reaches the message event | `start`, `sceneCut`, `setSeed` |

- **An answer is addressed to the one connection that asked.** There it resolves
  the awaited call *and* raises the `message` event, so a `use*Scheduled`-style
  hook does fire — but only on this connection, and with no way to tell which
  in-flight call it answers. Read the result off the awaited call. Anything a
  second client must also see has to broadcast, which is what the `state`
  snapshot is for.
- **None of them reject.** A refusal arrives as a broadcast `command_error` and
  resolves the call with `undefined`; so does a send that never completed, with the
  reason on `lastError`. `try/catch` is not how you detect a failed command.
- **The opener is `set_shot` + `start`.** Everything authored after the opener is scheduled (`schedule_*`) at an absolute `session_chunk`, then `start` runs them.
- **Live "now" beats** are `set_shot` / `scene_cut` (next boundary). **Scheduled** beats are `schedule_shot` / `schedule_scene_cut`.
- **No `unschedule` yet.** You can't move or cancel a scheduled beat once it's on the model — only `reset` clears everything. Compose before `start`, or fire live. (A future `unschedule` command would unlock live drag-editing.)

## Receiving messages

The `state` snapshot is the source of truth: `running`, `started`, `paused`, `current_chunk`, `session_chunk`, `current_prompt`, `seed`, `scheduled_shots`, `scheduled_scene_cuts`. Surface `command_error` (`<CommandError>`) so a rejected beat (empty prompt, wrong state, past chunk) is never silent.

`useLongliveV2ShotSet`, `ShotScheduled`, `SceneCutScheduled`, `GenerationPaused`, `GenerationResumed` and `GenerationReset` are answers, so they only ever fire on the connection that sent the command. Don't build the timeline or the phase gate on them — read `state`, which broadcasts.

## Auth — a memoizing `jwtToken` resolver + a scoped mint route

Two pieces work together: a Next.js GET route that mints a session-scoped JWT
server-side (`app/api/reactor/token/route.ts`), and the `jwtToken` resolver on
`<LongliveV2Provider>` that the SDK calls on every Reactor API hop.

`jwtToken` takes a `JwtSource` — `string | (() => string | Promise<string>)`. The
example passes `jwtToken={fetchToken}`, and the SDK re-invokes it on every Reactor
API call (uploads, `GET /clips`, ICE refresh, SDP renegotiation), so a token aging
out mid-session can't 401 those hops. A bare string works but fixes one value at
construction time. (Porting from js-sdk 2.x: the prop used to be a separate
`getJwt`; it is gone, and TypeScript catches the rename.) The provider stabilizes
the resolver via `useRef + useMemo`, so an inline arrow is safe and `useCallback`
is unnecessary.

**`fetchToken` memoizes the token in module scope until shortly before it expires,
and fetches with `cache: "no-store"` — that is load-bearing, not an optimization.**
The token is session-scoped: a session may only be operated by the exact token
that created it, so every hop of one session must present the same JWT. The
browser's HTTP cache cannot promise that — DevTools "Disable cache" and ordinary
eviction both make it miss — and on a miss the resolver mints a fresh token with
no bound sessions, so the next upload or clip call answers:

```
403 … this token is session-scoped and is not authorized for this resource;
mint it again with authorization_details.resources.sessions.bind …
```

The edge it does not cover: a session created just before the memo expires is
orphaned at the re-mint, because the fresh token is not bound to it. Covering that
needs a re-mint naming the live session in
`authorization_details.resources.sessions.bind`.

The route returns `expires_at` alongside the JWT so the resolver memoizes for
exactly the lifetime the server granted; sets `Cache-Control: private, no-store`
so no CDN stores a per-user credential; is a GET because nothing about the request
varies (it still POSTs to Reactor internally); and scopes the mint through
`authorization_details` to `reactor/longlive-v2` with a bounded `max_sessions`, so
the browser's token can only create sessions for that one model and act on the
ones it created. Your `rk_` API key never leaves the server.

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
