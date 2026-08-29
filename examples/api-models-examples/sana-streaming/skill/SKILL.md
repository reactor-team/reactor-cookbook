---
name: building-sana-streaming-frontends
description: Extend this cloned SANA-Streaming example app — add controls, presets, or stage features on top of `@reactor-models/sana-streaming` without breaking the patterns the existing code uses. Covers the SDK connection / events / messages model, the two input sources (webcam and a streamed video clip) that both feed the `camera` track, the single model-driven reducer fed by the typed `state` hook, mid-stream re-prompting (~1 chunk latency), and the behaviors to preserve — the camera publish hint and streaming a clip in rather than uploading it.
---

# Building on this SANA-Streaming app

This is a reference frontend for sana-streaming, Reactor's real-time video-to-video editor. Read this before extending it so you keep the patterns the code already uses.

## What sana-streaming is

A continuous video editor you steer with text. You stream a source into the model on the `camera` track — either your **webcam** or a **pre-recorded clip** — and a prompt describes a change; the model applies that change while everything you don't mention carries through from the source. Edited frames stream back on the `main_video` track in 24-frame chunks (~1–1.5s each).

The prompt is **optional**: with no prompt the model streams the source back nearly untouched; set or change one — at any time, including mid-stream — to steer the edit. A mid-stream prompt change lands at the next chunk boundary, about one chunk later.

## The two input sources

Both sources publish to the **same `camera` track**; the model only ever runs its live path. The difference is purely client-side — what media you put on the track.

|        | **webcam**                                  | **video**                                                    |
| ------ | ------------------------------------------- | ------------------------------------------------------------ |
| Source | `getUserMedia` → the live webcam track      | a chosen clip, played in a `<video>` and `captureStream()`'d |
| Flow   | produce track → `start` | pick clip → produce track → `start`      |
| Where  | `WebcamSource` self-view in the Input panel | `VideoSource` is the left pane in the stage                  |
| Stage  | single edited-output pane                   | split: your source clip (left) + edited output (right)       |

A selected video is **streamed, not uploaded**: `VideoSource` plays it and exposes its `captureStream()` track, so the video pane is literally the frames the model edits — the two stage panes share one feed and can't drift apart. The webcam self-view sits in the Input panel (it has no separate "before" to compare), so webcam mode is a single edited pane.

The **Input panel** (`ModeInput`) is phase-aware on the model's `started` flag: before `start` it shows the source toggle, the webcam self-view or the video picker, the seed, and Start; once `started` is true it swaps the setup controls for `Playback` (pause / resume / reset). In webcam mode the self-view stays mounted across the transition so the camera keeps streaming `camera`. Gate new controls on `started` / `paused` the same way, rather than disabling them in place.

> **The model is live-only from v2.0.0 on.** The file path and its `set_mode` / `set_video` commands are gone from the schema, so starting is just `start`. Earlier versions needed a `set_mode("live")` first; if you are reading code that still sends it, it is targeting a pre-2.0.0 release.

## The four concepts

| Concept                    | API                                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Connection**             | `useSanaStreaming()` → `status`, `connect()`, `disconnect()`, `lastError`. Four states: disconnected → connecting → waiting → ready.                          |
| **Commands (you send)**    | `useSanaStreaming()` → `setPrompt`, `setSeed`, `setAnchorInterval`, `start`, `pause`, `resume`, `reset` (+ `publish` / `unpublish`).                          |
| **Messages (you receive)** | `useSanaStreamingState((msg) => …)`, `useSanaStreamingCommandError((msg) => …)`, `useSanaStreamingGenerationReset((msg) => …)`, and a hook per other message. |
| **Tracks**                 | `<SanaStreamingMainVideoView />` — pre-bound `<ReactorView track="main_video">` for output. Input is the `camera` track you `publish` into.                   |

The provider is `<SanaStreamingProvider jwtToken={fetchToken}>` (`app/SanaStreamingApp.tsx`): the model name and tracks are baked in, and every base-provider prop (`jwtToken`, `connectOptions`, `apiUrl`, …) passes straight through.

## Auth — a memoizing `jwtToken` resolver + a scoped mint route

Two pieces work together: a Next.js GET route that mints a session-scoped JWT
server-side (`app/api/reactor/token/route.ts`), and the `jwtToken` resolver on
`<SanaStreamingProvider>` that the SDK calls on every Reactor API hop.

### `jwtToken` takes a string **or** a resolver

```tsx
type JwtSource = string | (() => string | Promise<string>);
```

The example passes `jwtToken={fetchToken}`. The SDK re-invokes that function on
every Reactor API call — `POST /sessions/:id/uploads`, `GET /clips`, ICE refresh,
SDP renegotiation — so a token aging out mid-session can't 401 those hops. A bare
string works too, but it fixes one value at construction time and breaks the
moment that value expires.

(Porting from js-sdk 2.x: the prop used to be a separate `getJwt`. It is gone;
`jwtToken` absorbed both shapes, and TypeScript catches the rename.)

The provider stabilizes the resolver via `useRef + useMemo`, so an inline arrow is
safe — a parent re-render does **not** tear the session down. Do not wrap it in
`useCallback`.

### The resolver memoizes, and that is load-bearing

`fetchToken` holds the minted token in module scope until shortly before it
expires, and fetches with `cache: "no-store"`. **Do not hand this job to the
browser's HTTP cache.** The token is session-scoped: a session may only be
operated by the exact token that created it, so every hop of one session must
present the same JWT. A browser cache cannot promise that — DevTools "Disable
cache" and ordinary eviction both make it miss — and on a miss the resolver mints
a fresh token with no bound sessions, so the next upload or clip call answers:

```
403 … this token is session-scoped and is not authorized for this resource;
mint it again with authorization_details.resources.sessions.bind …
```

The edge it does not cover: a session created just before the memo expires is
orphaned at the re-mint, because the fresh token is not bound to it. Covering that
needs a re-mint naming the live session in
`authorization_details.resources.sessions.bind`.

### The route — `app/api/reactor/token/route.ts`

Already implemented. You usually don't need to touch it, but here's why it works
the way it does so you don't accidentally break it:

1. **It returns `expires_at` alongside the JWT.** Reactor's `/tokens` endpoint
   takes an `expires_after` body and answers `{ jwt, expires_at }`. Handing
   `expires_at` to the client is what lets the resolver memoize for exactly the
   lifetime the server granted rather than a number guessed in the client.
2. **`Cache-Control: private, no-store`.** The client owns the cache (above), and
   `private` keeps any CDN or proxy from storing a per-user credential.
3. **GET, not POST.** Nothing about the request varies, so a GET reads as the
   lookup it is. The handler still POSTs to Reactor internally.
4. **`authorization_details` scopes the token.** The mint pins the JWT to
   `reactor/sana-streaming` with a bounded session budget (`max_sessions`): the
   browser's token can only create sessions for that one model and act on the
   sessions it created — everything else on the account answers 403. Never hand a
   browser an unscoped token; that is the API key's full user-level access in
   cookie-jar form.

Your `rk_` API key never leaves the server. The JWT is the only credential the
browser ever holds.

## The model is the source of truth

The browser sends commands and renders model-reported state; it never tracks generation state optimistically.

- The typed `state` snapshot (`useSanaStreamingState`) is the **only** thing that mutates the reducer. The model sends it on connect, after every accepted command, and at each chunk boundary, so the UI renders from one message instead of accumulating individual events. `app/lib/state.ts:reduce` projects it into `SanaState` (`app/lib/types.ts`): `running`, `started`, `paused`, `currentChunk`, `currentPrompt`, `seed`. Every gate in the UI — the Input panel's setup-vs-playback phase (`started`), pause-vs-resume (`paused`) — keys off this state, not local guesses.
- `command_error` is handled imperatively in the `Workspace` shell via its own typed hook — `useSanaStreamingCommandError` → a transient 6s banner (`<CommandError>`). The reset cleanup does **not** go through a hook: `generation_reset` answers `reset()`, so `Playback` calls `onReset` on the resolved await, which bumps `resetNonce` (children clear their local UI in step) and blacks out the stage until generation runs again.
- Reset local state to `DEFAULT_STATE` on full disconnect so a reconnect starts clean.

No `autoConnect` — `<StatusBadge>` surfaces the four-state machine with Connect/Disconnect buttons so the lifecycle is visible. Flip on `connectOptions={{ autoConnect: true }}` for a production app.

## Sending events — rules

- **Status-gate every control.** Only call command methods when `status === "ready"`.
- **The start flow is just `start`** — `lib/state.ts:startGeneration` encapsulates it; both sources call it. Awaiting it is a completion barrier: the runtime acknowledges a correlated command once its handler has run, so a resolved `start()` means the model started, not merely that bytes left the browser.
- **`setPrompt` is valid any time, including mid-stream.** It applies at the next chunk boundary; the prompt is an editing instruction, not a scene description.
- **A new control is a new typed method off `useSanaStreaming()`**, gated on `status === "ready"`, enabled/disabled off the reduced `SanaState` (see `Playback` for the smallest example). `setAnchorInterval` is the most obvious not-yet-surfaced knob — it periodically re-grounds the edit on the source to limit drift over long runs (every N chunks, `0` to disable); each re-ground may show a brief visible refresh.

## Where a result arrives: the call, or a subscription

A command resolves when the model's handler has finished, and carries whatever
that handler answered with. **None of them reject:** a refusal arrives as a
broadcast `command_error` and resolves the call with `undefined`, and so does a
send that never completed, with the reason on `lastError`.

| The model | Reaches you | Sana commands |
| --- | --- | --- |
| **answers** the command that asked | the awaited call's return value — and the **sending** connection's `message` event | `setPrompt`, `pause`, `resume`, `reset` |
| **broadcasts** to every connection | the per-message hooks | `state`, `command_error`, `chunk_complete`, `anchored`, `generation_started` / `_complete` |
| answers with **nothing** | the await resolves `undefined`; nothing reaches the message event | `start`, `setSeed`, `setAnchorInterval` |

An answer is **addressed**: it goes to the one connection whose command earned it,
correlated by request id. There it resolves the awaited call *and* raises the
`message` event, so `useSanaStreamingPromptAccepted` and
`useSanaStreamingGenerationReset` do fire — but only on this connection, and with
no way to tell which in-flight call they answer. Read the result off the call
instead: `Playback` runs its reset cleanup from the resolved `reset()`, which is
both unambiguous and the only shape that still works with a second client attached.

## Receiving messages

Each message has its own typed hook; subscribe to only the ones you care about, and the handler gets the fully-typed message (flat fields, no `.data` envelope).

| Message (hook)                                         | Role                                                             |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `state` (`useSanaStreamingState`)                      | Broadcast. **The only reducer input.** Full snapshot.             |
| `command_error` (`useSanaStreamingCommandError`)       | Broadcast. `{ command, reason }`. Always surface it (the shell banners it). |
| `chunk_complete` (`useSanaStreamingChunkComplete`)     | Broadcast. Per-chunk progress. Informational.                    |
| `generation_started` / `_complete` (matching hooks)    | Broadcast. Informational lifecycle markers.                      |
| `anchored` (`useSanaStreamingAnchored`)                | Broadcast when the edit re-grounds on the source (see `setAnchorInterval`). |
| `prompt_accepted` (`useSanaStreamingPromptAccepted`)   | **Sender-only** — the answer to `setPrompt`. Read it off the awaited call. |
| `generation_paused` / `resumed` / `reset`              | **Sender-only** — answers to `pause` / `resume` / `reset`. The shell runs its reset cleanup off the resolved `reset()` call, threaded to `Playback` as `onReset`. |

Anything that should change what the UI shows belongs in the reducer, fed only by `state`. `useSanaStreamingMessage` is a catch-all over the whole `SanaStreamingMessage` union (handy for devtools).

## Two behaviors to preserve

### 1. One owner publishes the `camera` track, with a content hint.

`WebcamSource` and `VideoSource` only _produce_ a track and hand it up via `onTrack`; a single owner — `useCameraPublisher` in `Stage` — publishes whichever track is current. Each source sets `track.contentHint = "detail"` first: the model expects a stable resolution, and `"detail"` tells the browser to hold resolution steady and trade framerate instead of ramping it up and down (the declarative `<SanaStreamingCameraView>` gives no hook to set the hint, which is why we publish manually). The publisher **always unpublishes before publishing**, so switching sources can't race into `publisher slot already taken` — the prior source's slot is freed first. Keep this single-owner shape if you add another input source.

### 2. A video source is streamed, not uploaded.

`VideoSource` plays the chosen clip in a muted, looping `<video>` and publishes its `captureStream()` track as `camera`. Playback is driven off the model's run state — paused at frame 0 while set up (a still poster that also seeds the published track), playing from the top once `started`, pausing/resuming in step. Because the published track _is_ the element you see, the source and edited panes stay in lockstep (offset only by the model's processing latency).

## Capturing clips

`<SnapClip>` uses the base `@reactor-team/js-sdk` (`useReactor`, `ClipPlayer`, `ClipDownloadButton`) — recording is model-agnostic and not re-exported by the typed package. `useReactor` works because `<SanaStreamingProvider>` wraps the base `ReactorProvider`. Drop the file into any model example unchanged.

## Common mistakes

- Reaching for the base SDK for sana-streaming-specific calls — use the typed `useSanaStreaming()` methods. The recording surface in `<SnapClip>` is the one intentional exception.
- Sending a command before `status === "ready"`.
- Uploading a video instead of streaming it — there is no `setVideo`; a clip goes in on the `camera` track like the webcam (see behavior 2).
- Swapping a source for `<SanaStreamingCameraView>` and losing the `contentHint` (see behavior 1).
- Forgetting to reset local state on disconnect, so a reconnect shows stale data.
- Single-line prompts — write a clear edit instruction (what changes, and what stays).

## Checklist for a new control

1. Decide its phase (setup vs live) and gate visibility off the reduced `SanaState`.
2. Status-gate every command on `status === "ready"`.
3. Use the typed `useSanaStreaming()` methods; subscribe via the per-message hooks.
4. Keep model state in the reducer (fed only by `state`); keep local UI drafts resettable on `resetNonce`.
5. Surface failures via `command_error` (`<CommandError>`).
6. Brand colors via the `bg-brand` / `text-active` Tailwind tokens (from `@reactor-team/ui`).
