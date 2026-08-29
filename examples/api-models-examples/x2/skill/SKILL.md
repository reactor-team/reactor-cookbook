---
name: building-x2-frontends
description: Extend this cloned XMAX X2 example app — add new controls, sources, or knobs on top of the Reactor JS SDK without breaking the patterns the existing code uses. Covers the typed @reactor-models/x2 client, the connection / commands / messages / tracks model, the single-owner source publish, the state_update reducer, the reference-image upload path, the pointer protocol, the auth route, and clip capture.
---

# Building on this XMAX X2 app

You've cloned this folder and now you want to extend it — a new control, a new input source, a different UX. This guide explains the patterns the existing code uses and the rules to follow so your additions feel native instead of bolted on.

All the code referenced below already exists in this folder. Read this guide alongside the source — especially [The source track](#the-source-track--three-producers-one-publisher) before touching anything in the input path.

## What X2 actually is, in three sentences

XMAX X2 is a **real-time streaming video-to-video editing model**. The client publishes a live video track to the model, describes an edit in plain text, and the model streams the edited video back — continuously, while you re-prompt mid-stream, swap the reference image, or drag a pointer across the output to steer the subject. The frontend's job reduces to (a) producing and publishing exactly one source track, (b) sending commands (`set_prompt`, `set_reference_image`, `set_pointer`, `set_keep_backlog`, `reset`), and (c) mirroring the model's `state_update` snapshot into the UI.

## The typed client is the published `@reactor-models/x2` package

The typed client is installed from npm as [`@reactor-models/x2`](https://www.npmjs.com/package/@reactor-models/x2) — the `X2Model` class and command/message types, plus the React surface (`X2Provider`, `useX2`, per-message hooks, `<X2MainVideoView>`). It is generated from the model's schema and only depends on `@reactor-team/js-sdk`.

## The four concepts you'll touch

| Concept        | What it is                                                                                                        | Hook / API                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Connection** | The lifecycle of the model session (`disconnected → connecting → waiting → ready`)                                | `useX2()` → `status`, `connect`, `disconnect`                     |
| **Commands**   | Things you send TO the model. Always async.                                                                       | `useX2()` → `setPrompt({...})`, `setPointer({...})`, `reset()`, … |
| **Messages**   | What the model broadcasts — the `state_update` snapshot and lifecycle markers. Acks answer their command instead. | `useX2StateUpdate((msg) => …)`, one hook per broadcast type       |
| **Tracks**     | Video in (`source`, published by the client) and out (`main_video`).                                              | `useX2()` → `publish` / `unpublish`, `<X2MainVideoView />`        |

The full wire surface — every command, every message, the `state_update` payload — is the model's schema reference on docs.reactor.inc. When this guide says "check the schema", that's the page it means.

## No start button — prompts arm generation

X2 has no `start` / `pause` / `resume`. Generation begins on its own once two things are true: source frames are arriving on the wire, and a non-empty prompt is set. That shapes the UX:

- The Prompt panel is the ignition. `Prompt.tsx`'s placeholder says so explicitly, and preset chips apply immediately on click.
- `state_update.generating` is the flag everything gates on — the stage's status row, the source-mode lock, the Reset button's visibility.
- `reset()` stops generation and clears prompt, reference image, and pointer server-side. The model answers with `generation_stopped { reason: "reset" }` and a fresh `state_update`.
- Swapping the reference image mid-run also stops generation — but with `reason: "reference_image_changed"`, and the model restarts by itself. `Workspace` distinguishes the two reasons: only a user reset bumps the nonce that remounts draft-holding children.

While generating, the source-mode toggle is locked (`SourcePanel`): switching the feed under a live edit produces garbage; reset first.

## Auth — a memoizing `jwtToken` resolver + a scoped mint route

Two pieces work together: a Next.js GET route that mints a session-scoped JWT
server-side, and a `jwtToken` resolver on `<X2Provider>` that the SDK calls on
every Reactor API hop.

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

Clip surfaces (`<ClipPlayer>`, `<ClipDownloadButton>`, `useClipDownload`) inherit
the resolver through React context, so you do not pass it through `SnapClip` — see
[Capturing clips](#capturing-clips).

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

Owning the memo in the app makes the token's lifetime something the app controls:

```tsx
const TOKEN_REFRESH_SKEW_MS = 60_000;
let cachedToken: { jwt: string; expiresAtMs: number } | null = null;
let inflightToken: Promise<string> | null = null;

async function fetchToken(): Promise<string> {
  if (
    cachedToken &&
    Date.now() < cachedToken.expiresAtMs - TOKEN_REFRESH_SKEW_MS
  ) {
    return cachedToken.jwt;
  }
  if (inflightToken) return inflightToken; // coalesce connect-time parallel hops
  // … fetch("/api/reactor/token", { cache: "no-store" }), store { jwt, expires_at } …
}
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
4. **`authorization_details` scopes the token.** The mint pins the JWT to this
   app's model with a bounded session budget (`max_sessions`): the browser's token
   can only create sessions for that one model and act on the sessions it created
   — everything else on the account answers 403. Never hand a browser an unscoped
   token; that is the API key's full user-level access in cookie-jar form.

Your `rk_` API key never leaves the server. The JWT is the only credential the
browser ever holds.

### Configuring autoConnect

The example passes `connectOptions={{ autoConnect: false }}` so the user clicks Connect and sees the `disconnected → connecting → waiting → ready` transitions. In your own product you can flip it on — just make sure your status indicator still surfaces the intermediate states, because sessions don't reach `ready` instantly.

## The state snapshot — one reducer, cleared on disconnect

The model broadcasts a `state_update` message on connect and after every observable change: the full picture (`prompt`, `has_reference_image`, `pointer_x/y/active`, `keep_backlog`, `generating`, `width`, `height`). It is the **single source of truth**. The app never infers session state from its own button clicks.

`Workspace` (in `app/X2App.tsx`) feeds every snapshot through `reduce()` (`app/lib/state.ts`) into the app-level `X2UiState` (`app/lib/types.ts`). Two wire quirks the reducer already handles — keep them handled:

- `prompt`, `width`, and `height` are typed `unknown` (nullable on the wire); the model only ever sends values or null.
- The snapshot only says _whether_ a reference image is set. The decoded dimensions come back as the answer to `setReferenceImage`, so the reducer drops the stale dimensions ack whenever the snapshot reports no reference.

Two rules to keep:

1. **Only `state_update` mutates the reducer.** Transition notifications (`generation_stopped`, acks) trigger one-shot reactions, never state reconstruction.
2. **Clear everything on disconnect.** `Workspace` has a `useEffect` on `status === "disconnected"` that resets the reducer, the error banner, and the selected media URLs. The SDK does not emit a final `state_update` on disconnect; without this, a reconnect shows stale data from the previous session. If you add new session state, add its reset there too.

## Sending commands

Commands are typed methods off `useX2()`:

```tsx
const { setPrompt, setPointer, reset } = useX2();
await setPrompt({ prompt: "make it watercolor" });
```

A command resolves when the model's handler has finished. **None of them reject:**
a refusal resolves the call with `undefined` and the reason lands on `lastError`.
So `try/catch` is not how you detect a failed command.

**There is no `command_error` message on this model.** From release 1.0.0 the
schema declares none, so no `useX2CommandError` hook is generated. A refused
command reports itself as that command's own error reply, which the SDK records on
`lastError` (and raises on the `error` event). `X2App` watches `lastError` and
banners a value that is new to this render — `lastError` is a persistent record
that success never clears, so comparing against the last-seen value is what
distinguishes a new failure from an old one still recorded.

### Where a result arrives

| The model                          | Reaches you                                           | X2 commands                                                                 |
| ---------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------- |
| **answers** the command that asked | the awaited call's return value — and the **sending** connection's `message` event | `setPrompt`, `setPointer`, `setReferenceImage`                              |
| **broadcasts** to every connection | the per-message hooks                                 | `state_update`, `generation_started`, `generation_stopped`                  |
| answers with **nothing**           | the await resolves `undefined`; nothing reaches the message event | `reset`, `setPointerX`, `setPointerY`, `setPointerActive`, `setKeepBacklog` |

An answer is **addressed**: the runtime sends it to the one connection whose
command earned it, correlated by request id. There it resolves the awaited call
*and* raises the `message` event, so `useX2ReferenceImageAccepted`,
`useX2PromptAccepted` and `useX2PointerChanged` do fire — but only on this
connection, and with no way to tell which in-flight call they answer. Read the
value off the call:

```tsx
// ⚠️ fires, but for any setReferenceImage on this connection — and never on a
// second client watching the same session.
useX2ReferenceImageAccepted((msg) => setSize(msg));

// ✅ tied to this call
const accepted = await setReferenceImage({ reference_image: ref });
if (accepted) setSize({ width: accepted.width, height: accepted.height });
```

That is what `ReferenceImage.tsx` does, lifting the decoded size to the shell
through an `onAccepted` prop.

- **Gate every interactive control on `status === "ready"`** — a command sent
  earlier resolves `undefined` and records the reason on `lastError`.
- **Prompts are hot-swappable.** `setPrompt` mid-stream applies from the next generated block; no restart, no re-render. This is the model's signature capability — lead with it in anything you build. Prompts cap at 1000 characters (`maxLength` on the schema).
- **`reset()` stops the run.** The stage blacks itself out until frames from the next run land (the WebRTC `<video>` would otherwise freeze on the last edited frame — see `stageCleared` in `X2App.tsx`).
- The split pointer commands (`setPointerX`, `setPointerY`, `setPointerActive`) exist for integrations that can only send one scalar at a time; the app uses the combined `setPointer({ x, y, active })`.

## Receiving messages

The typed client delivers each message type through its own hook; `Workspace` subscribes once per type it cares about:

| Message                        | Hook                     | What the app does                                                                                          |
| ------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `state_update`                 | `useX2StateUpdate`       | Broadcast. Feeds the reducer. Everything the UI gates on comes from here.                                   |
| `generation_stopped`           | `useX2GenerationStopped` | Broadcast. Blacks out the stage; bumps the reset nonce only when `reason === "reset"`.                      |
| `reference_image_accepted`     | `useX2ReferenceImageAccepted` | **Sender-only** — the answer to `setReferenceImage`. Read it off the awaited call instead.             |
| ~~`command_error`~~            | —                        | **Gone from the schema** as of 1.0.0. A refusal is the command's own error reply, recorded on `lastError`. |

`generation_started` also broadcasts; the app relies on the `state_update` echo instead, which arrives for the same transitions and carries the full picture. `prompt_accepted` and `pointer_changed` are answers, so their hooks fire only on the connection that sent the command — the app reads both off the awaited call.

Keep message handling centralized in `Workspace` — scattering per-message hooks across leaf components makes ordering unobvious.

## The source track — three producers, one publisher

The model edits whatever arrives on the `source` track. Three components _produce_ a track; exactly one hook _publishes_ it:

- **`WebcamSource`** — `getUserMedia` at 640×360, renders the self-view in the panel, hands the track up.
- **`VideoSource`** — plays the chosen clip in a `<video>` element and grabs its frames with `captureStream()`. The same element is the "original" pane in the stage, so what you see is literally what the model receives.
- **`ImageSource`** — paints a still image cover-cropped onto a 1280×720 canvas, re-paints on an interval so `canvas.captureStream(24)` keeps emitting, and hands that constant feed up. This is the drag-to-animate mode: a static source plus pointer drags.
- **`useSourcePublisher`** — the single owner of the `source` slot. A serialized reconcile loop converges the wire to the latest desired track. It deliberately does **not** unpublish before switching — the SDK's `publishTrack` replaceTrack()s the new media onto the existing sender without renegotiating — and it clears its belief when status leaves `ready` so the next session re-publishes.

Rules when extending:

1. **Never publish from a component.** New sources (screen share, a canvas, a second camera) produce a `MediaStreamTrack` and hand it to the workspace via `onTrack`; the publisher does the rest.
2. **Set `track.contentHint = "detail"`** on any new source. The model wants a stable input resolution; Chrome's encoder ramps resolution at stream start and on bandwidth dips, and `"detail"` holds it steady and trades framerate instead.
3. **The model picks its output resolution per session from the source stream.** Feed it a portrait webcam and you get a portrait edit; don't letterbox the source yourself.

## The reference image — uploads, not tracks

`ReferenceImage.tsx` conditions the edit on a picked image. The path is: `uploadFile(file)` (off `useX2()`) returns a `FileRef`, then `setReferenceImage({ reference_image: ref })` hands it to the model. Two details worth keeping:

- Preset reference images are `fetch`ed back into a `File` so presets and local picks share the exact same upload path — one code path, one set of bugs.
- A mid-run swap stops and auto-restarts generation (`generation_stopped { reason: "reference_image_changed" }`). Keep the drafts; see the reset-nonce logic in `Workspace`.

The awaited call answers with `reference_image_accepted { width, height }` — the decoded size, useful for telling the user what the model actually sees.

## The pointer — normalized, throttled, released

`PointerOverlay.tsx` turns drags on the edited output into `setPointer({ x, y, active })` calls. If you rebuild or extend it, preserve these three properties:

1. **Coordinates are normalized (0..1) in the output frame**, not the DOM box. The overlay accounts for `object-fit: contain` letterboxing using the model-reported output aspect (`width`/`height` from `state_update`), so a drag on the visible video maps to the frame the model is editing.
2. **Sends are throttled (~33 ms) with a trailing send**, so the last position of a fast gesture always lands.
3. **Release always sends `active: false`.** A pointer left active keeps steering the model after the user let go.

The sidebar's `PointerPanel.tsx` renders the raw `pointer_x` / `pointer_y` / `pointer_active` values from the `state_update` echo — the model's view of the pointer, not the local gesture. It shows exactly what a `set_pointer` call carries, in every source mode.

## Keep backlog — the latency / completeness dial

The `set_keep_backlog` toggle (checkbox in `SourcePanel`) picks what happens when the model falls behind the source: keep every frame queued and edit all of them (latency grows, nothing is skipped — right for editing a finite clip end-to-end) or drop stale frames and always edit newest (right for live webcam). Default is off. The current value echoes back on `state_update.keep_backlog`, like every other knob.

## Curated prompt presets

Preset chips live in `app/lib/examples.ts` and are English translations of the model's validation prompts — they demonstrate the density that works. X2 prompts are **editing instructions, not scene descriptions** — say what should change; everything unmentioned carries through from the source. The model's prompt guide on docs.reactor.inc has good-vs-bad pairs to copy density from.

## Capturing clips

The Reactor base SDK exposes a recording surface that works for every model: ask for the last N seconds of the live stream, get back a `Clip`, and either preview it with `<ClipPlayer>` or download it with `<ClipDownloadButton>`.

The app ships a drop-in `app/components/SnapClip.tsx` panel wiring this together — a Capture button calling `requestClip(durationSeconds)` off the store, a preview modal, an MP4 download. It is model-agnostic: the same file ships (modulo theme classes) in every Reactor example, and it is the one place that imports `@reactor-team/js-sdk` directly (recording is base-SDK surface that typed clients don't re-export).

The pattern, if you rebuild it:

1. `useReactor((s) => s.requestClip)` — `requestClip`, `requestRecording`, and `downloadClipAsFile` are first-class store actions.
2. Return `null` when `status !== "ready"` — same hide-on-disconnect contract as every live-only control.
3. Catch `RecordingError` (typed reasons: `DISCONNECTED`, `RECORDER_DISABLED`, `INVALID_DURATION`, `REQUEST_TIMEOUT`) and surface it inline.
4. `<ClipPlayer>` + `<ClipDownloadButton>` auto-inherit the JWT resolver from the provider via context; no `getJwt` prop needed.
5. `hls.js` is a direct dep: `<ClipPlayer>` plays HLS natively on Safari and dynamically imports `hls.js` on Chrome/Firefox/Edge.

**Clips are short-lived.** The URL on a `Clip` expires after a few minutes. Download the MP4 if you need an artifact; don't hand `clip.playlistUrl` to users.

## Brand alignment — tokens through the theme, not components

`app/layout.tsx` imports `@reactor-team/ui/styles.css` (fonts + brand CSS vars) and `app/globals.css` re-exposes them as Tailwind utilities (`bg-brand`, `text-brand-fg`, `bg-active`, `font-mono`).

- **Use the theme utilities and the primitives in `app/components/ui/`** (Button, Panel, SegmentedToggle, IconButton). Don't invent parallel color systems with raw hex values.
- **Don't import `@reactor-team/ui` React components.** They use hooks internally — importing one into a Server Component (like `SetupRequired`) dies at runtime, not build time. The stylesheet import gives you everything you need.

## Common mistakes when extending

1. **A second publisher for `source`.** All sources hand their track to `useSourcePublisher` via `onTrack`. Two publishers race against the same transceiver.
2. **Inferring session state from clicks.** Gate off the reduced `X2UiState`; only `state_update` mutates it.
3. **Forgetting the disconnect reset.** New session state must be cleared in the `status === "disconnected"` effect, or the next session starts haunted by the last one.
4. **Swallowing a command failure.** There is no `command_error` message to subscribe to; failures land on `lastError`, and the shell's effect banners each new one. Keep that effect wired when you add commands, and treat a falsy resolved value as the signal that one happened.
5. **Adding a Start button.** Generation is armed by the prompt; a Start button would have nothing to send and teaches the wrong mental model.
6. **Treating every `generation_stopped` as a user reset.** `reference_image_changed` stops auto-restart; only `reason === "reset"` should clear drafts.
7. **Uploading the clip instead of streaming it.** `VideoSource` exists so the model edits the clip on its live path; the only upload in this app is the reference image.
8. **Missing `contentHint = "detail"`** on a new source track — the model gets a resolution ramp instead of a stable stream.
9. **Leaving the pointer active.** Every drag must end with `active: false`, including cancel paths (pointer leave, unmount).
10. **Importing `@reactor-team/ui` React components into Server Components.** Runtime error. Use the theme tokens.
11. **Storing `Clip` objects or sharing clip URLs.** They expire in minutes. Download the MP4.
12. **Hand-rolling a parallel typed client.** `@reactor-models/x2` is generated from the model's schema; import from it instead of re-declaring command or message types locally.

## Checklist for new components

- [ ] Gated on `status === "ready"` plus the right `X2UiState` flags (`generating` for run-sensitive controls)
- [ ] New sources produce a track and hand it up via `onTrack` — no new publish call sites
- [ ] New message handling lives in `Workspace` via the typed per-message hooks
- [ ] New session state resets in the disconnect effect
- [ ] Command failures still surface through the `lastError` banner (don't swallow them)
- [ ] Reads an acceptance off the awaited call, not off a `use*Accepted` subscription (those never fire)
- [ ] Command payloads use the typed methods off `useX2()` — no raw `sendCommand` strings
- [ ] Colors via theme utilities / `app/components/ui` primitives, not raw hex
- [ ] Typed surface imported from `@reactor-models/x2`; base `@reactor-team/js-sdk` imported only where the typed client doesn't cover (SnapClip / recording)
