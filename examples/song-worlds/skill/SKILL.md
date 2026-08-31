---
name: building-helios-frontends
description: Extend this cloned Helios example app — add new controls, scenes, knobs, image flows, or features on top of `@reactor-models/helios` (pinned to ^0.9.0) without breaking the patterns the existing code already uses. Covers the SDK's connection / events / messages model, the phase-based UI architecture, the state snapshot pattern, the atomic `setConditioning({ prompt, image })` command for image-to-video flows, mid-stream prompt switching, the curated scene library, and prompt design rules for smooth continuous video generation.
---

# Building on this Helios app

You've cloned this folder and now you want to extend it — a new control, a new scene, a new model knob, a different UX. This guide explains the patterns the existing code uses and the rules to follow so your additions feel native instead of bolted on.

All the code referenced below already exists in this folder. Read this guide alongside the source.

## What Helios actually is, in three sentences

Helios is a **continuous, prompt-driven** video model. Once it starts generating, it produces an unending stream of video on a single track (`main_video`) — there is no "request, get clip, end". You steer the scene mid-stream by changing the prompt (which the model picks up on the next chunk), or by changing the reference image (which retints the conditioning without restarting).

The frontend's job is to (a) start the generation, (b) keep the user steering it, and (c) gracefully reflect the model's state.

## The four concepts you'll touch

| Concept        | What it is                                                                         | Hook / API                                                                                                                     |
| -------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Connection** | The lifecycle of the model session (`disconnected → connecting → waiting → ready`) | `useHelios().status`, `.connect()`, `.disconnect()`                                                                            |
| **Events**     | Things you send TO the model. Always async.                                        | `useHelios().setPrompt({...})`, `.setImage({...})`, `.setConditioning({...})`, `.start()`, `.pause()`, `.resume()`, `.reset()` |
| **Messages**   | Things the model sends BACK to you — including the all-important `state` snapshot. | `useHeliosState((m) => …)`, `useHeliosCommandError`, `useHeliosImageAccepted`, etc.                                            |
| **Tracks**     | The model's video output, rendered as a live `MediaStreamTrack`.                   | `<HeliosMainVideoView />`                                                                                                      |

You almost never have to drop below this surface. If you find yourself reaching for `@reactor-team/js-sdk` directly, stop and re-read the typed hooks list — there's likely a typed hook you're missing. The one documented exception is the recording surface (see [Capturing clips](#capturing-clips) below), which is a base-SDK feature that the typed packages deliberately do not re-export.

## The UI phase model

A real-time video session is not one screen — it's a state machine. This app maps that state machine to **two visible UI phases**, and each component decides for itself which phase it lives in:

```
       ┌──────────────┐    setPrompt + start    ┌────────────────┐
       │  WAITING     │ ──────────────────────▶ │   GENERATING   │
       │  (Setup UI)  │ ◀───────────────────────│   (Live UI)    │
       └──────────────┘          reset          └─────┬──────────┘
                                                     │ ▲
                                                pause│ │resume
                                                     ▼ │
                                                ┌────────────────┐
                                                │     PAUSED     │
                                                │   (Live UI)    │
                                                └────────────────┘
```

| UI phase  | When                                                                           | What's visible                                                                         | What's hidden                    |
| --------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------- |
| **Setup** | `snapshot.started === false` (or no snapshot — fresh page / just disconnected) | StatusBadge · CommandError · setup controls (prompt presets + image picker)            | NowPlaying · live-scene controls |
| **Live**  | `snapshot.started === true` (running OR paused)                                | StatusBadge · CommandError · NowPlaying (Pause/Resume/Reset) · evolution prompt picker | setup controls                   |

Components self-hide via early returns on the snapshot. No orchestration logic in the parent — adding a new component means dropping it into the sidebar and putting the right early-return at its top.

### When you add a new control, decide its phase first

Before writing a new component, decide which phase it belongs to:

- **Knob that primes a session** (e.g. seed picker, super-resolution mode) → Setup phase. Early-return when generating.
- **Knob that adjusts the live scene** (e.g. image-strength slider, schedule-prompt picker) → Live phase. Early-return when not generating.
- **Always-on** (e.g. a stats panel) → no early return; just gate interactivity on `status === "ready"`.

```tsx
// Setup-phase component
if (status === "ready" && snapshot?.started) return null;

// Live-phase component
if (status !== "ready" || !snapshot?.started) return null;
```

The `status === "ready"` half of these checks matters — without it, your component will render stale data from a previous session after a disconnect/reconnect.

## What's intentionally not exposed (and where to add it)

The model offers more knobs than this app surfaces. Each one is straightforward to add — drop a new component into the matching phase and call the relevant typed method. The model itself supports:

| Knob                  | Hook                                                        | Lives in                     | Notes                                                            |
| --------------------- | ----------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------- |
| `set_seed`            | `useHelios().setSeed({ seed })`                             | Setup (read once at `start`) | `-1` = random.                                                   |
| `set_sr_scale`        | `useHelios().setSrScale({ sr_scale })`                      | Live OR Setup                | `"off" \| "2x" \| "4x"`. Takes effect on the next chunk.         |
| `set_image_strength`  | `useHelios().setImageStrength({ image_strength })`          | Live                         | `0..1`. Ignored when no image is set.                            |
| `schedule_prompt`     | `useHelios().schedulePrompt({ prompt, chunk })`             | Live                         | Queues a prompt at a specific future chunk.                      |
| Mid-stream image swap | Re-run the upload→`setImage` flow from a live-phase control | Live                         | `set_image` is a conditioning tweak; the prompt stays untouched. |

For each: add a small component, drop it into the sidebar at the right phase, and the rest of the architecture (status gating, snapshot lifecycle, error surfacing) Just Works.

## Auth — `getJwt` resolver + cacheable GET route

Two pieces work together: a Next.js GET route that mints (and caches) the JWT server-side, and a `getJwt` resolver prop on `<HeliosProvider>` that calls it on every Coordinator HTTP hop.

### `getJwt`, not `jwtToken`

`@reactor-team/js-sdk` ≥ 2.10.1 accepts a **resolver** anywhere it used to take a static string:

```tsx
type JwtSource = string | (() => string | Promise<string>);
```

The example passes `getJwt={fetchToken}` to `<HeliosProvider>`. The SDK then re-invokes that function on every Coordinator HTTP call — `POST /sessions/:id/uploads`, `GET /clips`, ICE refresh, SDP renegotiation — so a token aging out mid-session can't 401 those hops. The legacy `jwtToken="..."` string prop still works but caches one value at construction time and breaks the moment that value expires.

The provider auto-stabilizes the resolver via `useRef + useMemo`, so the inline arrow form is safe — a parent re-render does **not** tear the session down. Do not wrap it in `useCallback`.

Clip surfaces (`<ClipPlayer>`, `<ClipDownloadButton>`, `useClipDownload`) auto-inherit `getJwt` via React context, so you do not pass it through `SnapClip` anymore — see [Capturing clips](#capturing-clips).

### The route — `app/api/reactor/token/route.ts`

Already implemented. You usually don't need to touch it, but here's why it works the way it does so you don't accidentally break it:

1. **GET, not POST.** Browsers don't cache POST responses. The route handler still POSTs to the Reactor API internally; the public route exposes itself as GET so the browser's HTTP cache can transparently serve repeat calls.
2. **`Cache-Control: private`.** Never `public` — JWTs are per-user and must not be shared across users by any CDN or proxy.
3. **`max-age` derived from the server's `expires_at`**, not a hardcoded number. The Reactor `/tokens` endpoint accepts an `expires_after` body and returns `{ jwt, expires_at }`. The route uses `expires_at` to set the cache window so it always tracks what the server actually granted.

Because the route is GET + cacheable, the `getJwt` resolver is also dumb on the wire — every Coordinator hop calls `fetch("/api/reactor/token")`, which 99% of the time comes back from the browser's HTTP cache without ever touching your server.

### Wiring an identity-provider JWT instead (Clerk, Auth0, …)

If your app uses Clerk session tokens or any other short-TTL identity JWT (Clerk's `getToken({ template: "reactor" })` ships with a default ~60s TTL), `getJwt` is _the_ hook for that:

```tsx
import { useAuth } from "@clerk/nextjs";

function App() {
  const { getToken } = useAuth();
  return (
    <HeliosProvider
      getJwt={async () => (await getToken({ template: "reactor" })) ?? ""}
    >
      {/* ... */}
    </HeliosProvider>
  );
}
```

Returning `""` suppresses the `Authorization` header entirely (use this for local-dev / unauthenticated paths). `getJwt` wins over `jwtToken` when both are passed.

### Configuring autoConnect

`<HeliosProvider>` is initialized **without** `autoConnect`. The user clicks "Connect" so they see the `disconnected → connecting → waiting → ready` transitions. If you're shipping a polished product where you'd rather the connection happen on page load:

```tsx
<HeliosProvider getJwt={fetchToken} connectOptions={{ autoConnect: true }}>
```

Just make sure your status indicator still surfaces the intermediate states (`connecting`, `waiting for GPU`) — sessions don't reach `ready` instantly, and you don't want users staring at an unexplained loading state.

## The state snapshot — your UI's single source of truth

Helios emits a `state` message after every command and every completed chunk. Subscribe via `useHeliosState`, hold it in `useState`, and read fields off it. **Don't aggregate `chunk_complete`, `generation_started`, `generation_paused` and try to reconstruct state yourself** — the snapshot already contains everything.

```tsx
const [snapshot, setSnapshot] = useState<HeliosStateMessage | null>(null);
useHeliosState((msg) => setSnapshot(msg));
```

Fields you'll actually read:

| Field                             | Meaning                                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `started`                         | True once `start()` has succeeded. Stays true through pause. Reset to false by `reset()`. **This is the phase switch.**                                 |
| `running`                         | True while the model is actively producing frames. Equal to `started && !paused`.                                                                       |
| `paused`                          | True after `pause()`, false again after `resume()`.                                                                                                     |
| `current_prompt`                  | The prompt currently driving generation. `null` (typed as `unknown`) before `start()`. **Match this against the scene library** to drive evolution UIs. |
| `current_chunk` / `current_frame` | Progress counters since the last `reset` / connect.                                                                                                     |
| `image_set`                       | True if a reference image has been provided this session.                                                                                               |

### Clear the snapshot on disconnect

The SDK does not emit a final `state` message when the session ends. Without an explicit reset, the last snapshot from the previous session lingers in your component's state — so after a reconnect, your UI shows stale "we're still generating!" data until the new session's first `state` arrives.

Every component that holds a snapshot does this:

```tsx
useEffect(() => {
  if (status !== "ready") setSnapshot(null);
}, [status]);
```

When you add a new component that subscribes to `useHeliosState`, include this. Three lines, no abstraction needed.

## Sending events — the typed methods

Every event Helios accepts has a typed wrapper on `useHelios()`. Always await them; they return a Promise that can reject.

```tsx
const { setPrompt, setImage, setConditioning, start, pause, resume, reset } =
  useHelios();

// Text-to-video
await setPrompt({ prompt: "A serene mountain lake at sunrise…" });
await start();

// Image-to-video (curated scene with both pieces known up front)
await setConditioning({ prompt: "A serene mountain lake…", image: ref });
await start();

// Transport
await pause();
await resume();
await reset();
```

**`setConditioning` (`@reactor-models/helios@^0.9.0`)** bundles a prompt and an image into a single data-channel message. Prefer it over separate `setImage` + `setPrompt` calls whenever both pieces of conditioning are known at the same time — see the next section for why.

**Never reach for `sendCommand("set_prompt", ...)` when a typed method exists.** You lose autocomplete and the param-name typo check.

### Status-gate every interactive control

Sending an event when `status !== "ready"` is a no-op with a console warning. Surface this as `disabled` on the button so the user sees what's clickable:

```tsx
const { status, setPrompt, start } = useHelios();
const ready = status === "ready";

<button disabled={!ready || !text.trim()} onClick={...}>Start generating</button>
```

On disconnect, the gate trips and your new control greys out automatically — exactly the same visual state as a freshly loaded, never-connected page.

## Chaining commands — prefer atomic commands over acknowledgment waits

Events are fire-and-forget over a data channel. Each event the SDK sends is a separate message, and commands that carry uploads (like `setImage`) take longer to resolve on the model than commands that don't (like `start`). Naive chaining races the model:

```tsx
// ❌ flickers on the first frame
await setImage({ image: ref });
await setPrompt({ prompt: "..." });
await start();
```

`setImage` carries a `FileRef` that the runtime resolves into a real upload before the model dispatches it. While that resolution is in flight, `start` (which carries no upload) sails past on the same data channel and the model dispatches it first. The first chunk is then generated with no image conditioning, the image lands a tick later, and the user sees a visible scene-shift on the second chunk.

### `setConditioning` is the right answer when you know both up front

For "kick off an image-to-video session from a known prompt + image" — which is the entire `startFromExample` path in `app/components/ImageStarter.tsx` — use the atomic command:

```tsx
async function startFromExample(scene) {
  const blob = await fetch(scene.imageUrl).then((r) => r.blob());
  const ref = await uploadFile(blob, { name: `${scene.id}.jpg` });

  await setConditioning({ prompt: scene.initial.text, image: ref });
  await start();
}
```

That's the whole flow. No `useRef`, no one-shot resolver, no `useHeliosImageAccepted` wiring. Why this works:

- `setConditioning` is **one message on the wire**. A single message can't be split or reordered, so the runtime can't insert `start` between the prompt and the image.
- The model handles it as a single transaction: validate → decode → VAE-encode → commit. The prompt schedule and image latents are only mutated together, and only after both pieces succeed. If anything fails (no prompt, no image, non-image MIME, undecodable bytes), the model emits `command_error` and **mutates nothing** — there's no partial state to recover from.
- After it lands, the model emits `prompt_accepted`, `image_accepted`, `conditions_ready(True, True)`, and a fresh `state` snapshot in one go. By the time the next message on the queue (`start`) is dispatched, conditioning is fully ready.

Use `setConditioning` whenever:

- You're priming a fresh session with both pieces (curated scene launches, "load this preset" buttons).
- The user has already typed a prompt **and** picked an image, and you want to commit them as one click.

### When to fall back to `setImage` (image only) or `setPrompt` (prompt only)

The example only sends one piece at a time when there's a genuine human gap between them:

- **`uploadCustomImage` in `ImageStarter.tsx`** calls `setImage` alone. The user uploads an image, then types a prompt in the textarea above, then clicks Start. By the time `setPrompt + start` fires from `PromptComposer`, the upload has long since been processed — no race.
- **`EvolveScene.tsx`** calls `setPrompt` alone for mid-stream hot-swaps. The model is already running; there's no `start` to race against.

Rule of thumb: **if you're about to call `start()` and you need conditioning, use `setConditioning` for both pieces or `setPrompt` for just a prompt.** Never chain `setImage + setPrompt + start` in the same function — that's the race case.

### The general "wait for an ack before `start()`" pattern

Helios 0.9.0 covers the prompt + image case atomically, so the example doesn't need an acknowledgment wait anywhere today. The pattern below is preserved as a reference for any **future slow command without an atomic wrapper** (a hypothetical `setControlNet`, `setLora`, `setReferenceAudio`, …). Subscribe to that command's `*_accepted` message and gate `start()` on it:

```tsx
// Reference pattern. Not used by the current example —
// `setConditioning` covers the only chained case we have.
const readyRef = useRef<(() => void) | null>(null);

useHeliosSomethingAccepted(() => {
  if (readyRef.current) {
    readyRef.current();
    readyRef.current = null;
  }
});

async function startWithSlowStep(params) {
  // Park the resolver BEFORE sending the command — registering it
  // afterwards would race the model's response.
  const ready = new Promise<void>((resolve) => {
    readyRef.current = resolve;
  });

  await setSomethingSlow(params);
  await ready; // ← gate start() on the ack
  await start();
}
```

Prefer specific `*_accepted` messages over the broader `conditions_ready` — the latter fires once per conditioning command with partial flags, which makes predicate matching finicky.

The text-only path (`setPrompt → start`) never needs an ack wait — prompt updates are instant.

## Receiving messages — the typed hooks

`@reactor-models/helios` ships one typed subscription hook per message:

| Hook                                                | Purpose                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `useHeliosState(handler)`                           | The state snapshot. **Use this; almost everything you need is here.**                                                                                                                                                                                                                                                                                     |
| `useHeliosCommandError(handler)`                    | A command was rejected (bad preconditions, bad input). Render this somewhere visible.                                                                                                                                                                                                                                                                     |
| `useHeliosImageAccepted(handler)`                   | An uploaded image was successfully processed. **Not required for the curated image-to-video flow** — `setConditioning` already commits both pieces atomically before `start` can be dispatched. Useful for "image uploaded ✓" toasts on `setImage`-only flows, or as the gate for a hypothetical future slow command that doesn't have an atomic wrapper. |
| `useHeliosPromptAccepted(handler)`                  | A prompt was queued. Useful for toast notifications.                                                                                                                                                                                                                                                                                                      |
| `useHeliosChunkComplete(handler)`                   | One chunk finished generating. Useful for progress sounds, telemetry.                                                                                                                                                                                                                                                                                     |
| `useHeliosGenerationStarted` / `Paused` / `Resumed` | Lifecycle transitions. Useful for one-shot reactions (toasts, sounds), but **don't aggregate these into your own state** — read the snapshot instead.                                                                                                                                                                                                     |
| `useHeliosMessage(handler)`                         | Catch-all over the typed discriminated union. Useful for devtools / logging.                                                                                                                                                                                                                                                                              |
| `useHeliosConditionsReady(handler)`                 | Fires after each conditioning command with `has_prompt` / `has_image` flags. Lower-level than `image_accepted`; prefer the specific hooks.                                                                                                                                                                                                                |

### Always surface `command_error`

`app/components/CommandError.tsx` already does this. The pattern:

```tsx
"use client";
import { useState } from "react";
import { useHeliosCommandError, useHeliosState } from "@reactor-models/helios";

export function CommandError() {
  const [err, setErr] = useState<{ command: string; reason: string } | null>(
    null,
  );
  useHeliosCommandError((m) =>
    setErr({ command: m.command, reason: m.reason }),
  );
  useHeliosState(() => setErr(null)); // any state update means the user moved on
  if (!err) return null;
  return (
    <div className="error">
      {err.command} failed: {err.reason}
    </div>
  );
}
```

When you add a new event method, this will surface its failures automatically — no changes needed.

## Image-to-video flow

Two patterns live in `app/components/ImageStarter.tsx`, one per launch path:

### Curated scene — prompt + image known at the same time

```tsx
const blob = await fetch(scene.imageUrl).then((r) => r.blob());
const ref = await uploadFile(blob, { name: `${scene.id}.jpg` });
await setConditioning({ prompt: scene.initial.text, image: ref });
await start();
```

1. Get bytes (`fetch(url).then(r => r.blob())` for a curated image, or `e.target.files[0]` from `<input type="file">`).
2. `await uploadFile(blob)` → returns a `FileRef`.
3. `await setConditioning({ prompt, image: ref })` — atomic. The SDK lifts the `FileRef` out of the params into an `uploads` envelope automatically; you treat `image: ref` as a regular field. Both pieces of conditioning are validated, decoded, VAE-encoded, and committed as one transaction. On failure the model emits `command_error` and leaves state untouched.
4. `await start()`. No acknowledgment wait needed — `setConditioning` lands before `start` reaches the model.

### Custom upload — user picks an image, types a prompt later

```tsx
const ref = await uploadFile(file);
await setImage({ image: ref });
// …user types in <PromptComposer>, which fires setPrompt + start
```

1. Same upload path as above.
2. `await setImage({ image: ref })` — image only. Snapshot now reports `image_set: true`.
3. The user types in the textarea above and clicks Start. `PromptComposer` fires `setPrompt + start`. By that point the human delay has long covered the model's image processing, so there's no race — and no acknowledgment wait — in this path either.

### Mid-stream image swap

Once `started === true`, calling `setImage` is a conditioning tweak, not a restart. Leave the prompt alone — overwriting the user's active prompt is destructive. (Mid-stream image swap UIs aren't shipped in this example; add one as a live-phase component if you want it.)

## The scene library — one source, three surfaces

All suggested prompts live in `app/lib/prompts.ts`. Each scene is self-contained:

```ts
export interface Prompt {
  title: string; // short headline for the button label
  text: string; // full paragraph sent to the model
}

export interface Scene {
  id: string;
  label: string;
  initial: Prompt;
  evolutions: ReadonlyArray<Prompt>;
  imageUrl?: string; // present on image-backed scenes
}

export const SCENES: ReadonlyArray<Scene> = [
  /* ... */
];

export const TEXT_SCENES = SCENES.filter((s) => !s.imageUrl);
export const IMAGE_SCENES = SCENES.filter(
  (s): s is Scene & { imageUrl: string } => !!s.imageUrl,
);

export function findSceneForPrompt(
  prompt: string | null | undefined,
): Scene | null {
  if (!prompt) return null;
  return (
    SCENES.find(
      (s) =>
        s.initial.text === prompt ||
        s.evolutions.some((e) => e.text === prompt),
    ) ?? null
  );
}
```

The library feeds three surfaces:

- `PromptComposer` reads `TEXT_SCENES` → renders `initial` of each as a clickable card.
- `ImageStarter` reads `IMAGE_SCENES` → renders thumbnails (and ships the image bytes from `/public/images/`).
- `EvolveScene` calls `findSceneForPrompt(snapshot.current_prompt)` → renders that scene's `evolutions` as one-click hot-swaps.

**Adding a new scene = one entry in `SCENES`.** No component changes. If the new scene has an `imageUrl`, drop the image bytes into `public/images/` and reference it.

### Prompts must be full paragraphs with explicit visual continuity

This is the most underrated part of building a real-time video frontend, and the #1 reason scenes look choppy when they should be smooth.

**Each prompt is a paragraph**, not a tagline. Describe the subject, the action, the environment, the lighting, AND the camera shot. Single-sentence prompts ("a cat chasing a butterfly") produce visually unstable output because the model has to invent everything else from scratch each chunk.

**Each evolution must re-establish the subject before introducing the change.** This is what lets the model hot-swap the prompt mid-stream without the scene visually resetting. Pattern:

```
Initial:    "A majestic lion named Leo stands regally in the heart of a
             dense jungle, embodying the essence of a king. Leo has a
             golden mane that flows gracefully around his broad shoulders…
             A medium close-up perspective emphasizing Leo's powerful
             stance and the regal aura surrounding him."

Evolution:  "Leo maintains his regal position on the rocky outcrop as
             the humid jungle air settles around his broad shoulders. He
             suddenly lowers his massive head to sniff a vibrant blue
             butterfly that has fluttered near his nose…
             A medium close-up perspective emphasizing Leo's powerful
             stance and the regal aura surrounding him."
```

The camera shot, subject identity, and environment are restated **verbatim**. Only the action changes. That stability is what produces smooth on-screen continuity.

### UI for long prompts

Render only the short `title` as the button label, with `line-clamp-2` over the dim `text` underneath. The full text reaches the model on click; the truncation is purely visual:

```tsx
<button onClick={() => setPrompt({ prompt: scene.initial.text })}>
  <div className="text-xs font-medium">{scene.label}</div>
  <p className="mt-0.5 line-clamp-2 text-[11px] text-zinc-500">
    {scene.initial.text}
  </p>
</button>
```

## Mid-stream prompt switching

This is Helios's signature capability. Once `started === true`, calling `setPrompt({ prompt })` is a **hot-swap on the next chunk** — no restart, no `start()` again. The scene continues from exactly where it was, with the new prompt influencing subsequent frames.

`app/components/EvolveScene.tsx` is the canonical pattern: find the active scene by matching `snapshot.current_prompt` against the library, render its evolutions as click handlers that call `setPrompt` directly:

```tsx
const current =
  typeof snapshot.current_prompt === "string" ? snapshot.current_prompt : "";
const scene = findSceneForPrompt(current);
if (!scene) return null;

// …
{
  scene.evolutions.map((e) => (
    <button
      key={e.title}
      disabled={current === e.text}
      onClick={() => setPrompt({ prompt: e.text })}
    >
      {e.title}
    </button>
  ));
}
```

Notice: **no `start()`** in the click handler. The model is already generating; we're just swapping the prompt. The snapshot updates on the next chunk, the active evolution is highlighted/disabled, and the model picks up the new prompt smoothly.

If you build a new control that mutates the live scene (e.g. a slider for image strength), follow the same pattern — call the typed method, don't touch `start()`.

## Capturing clips

The Reactor base SDK exposes a recording surface that works for every model: ask for the last N seconds of the live stream, get back a `Clip`, and either preview it with `<ClipPlayer>` or download it with `<ClipDownloadButton>`. The model SDK does not own this — it lives on `@reactor-team/js-sdk` because it is the same call for Helios, Lingbot, and every future model with recording enabled.

The example ships a drop-in [`app/components/SnapClip.tsx`](../app/components/SnapClip.tsx) panel that wires this together: a "Capture" button that calls `requestClip(durationSeconds)` off the store, opens a modal with the SDK's preview player, and offers an MP4 download. It is **model-agnostic** — the same file ships unchanged in every example.

### When to reach for `@reactor-team/js-sdk` directly

The default rule still applies: do everything via `@reactor-models/helios`. But the typed package only re-exports model-specific surface (events, messages, the typed provider/hook). The recording surface is base-SDK only, so for that one feature you import directly:

```tsx
import {
  ClipDownloadButton,
  ClipPlayer,
  RecordingError,
  useReactor,
  type Clip,
} from "@reactor-team/js-sdk";
```

When you scaffold a new component, ask: "Does this depend on Helios-specific events, messages, or commands?" If yes → typed package only. If no, and it would work the same on any model (recording, generic stats, generic connection state) → `@reactor-team/js-sdk` is fine.

### The pattern

`SnapClip` is small enough to read in one go. The shape that matters:

1. **Destructure the recording action off the store.** `useReactor((s) => s.requestClip)` is the canonical accessor as of `@reactor-team/js-sdk` ≥ 2.11.1 — `requestClip`, `requestRecording`, and `downloadClipAsFile` are first-class actions alongside `connect` / `disconnect` / `uploadFile`. No `s.internal.reactor` indirection.
2. **Gate on connection status.** `useReactor((s) => s.status)` — return `null` when status is not `"ready"`, so the panel disappears on disconnect just like every other live-only control.
3. **Catch `RecordingError`.** Recording can fail with typed reasons (`DISCONNECTED`, `RECORDER_DISABLED`, `INVALID_DURATION`, `REQUEST_TIMEOUT`). Surface them inline like `CommandError` does.
4. **Compose `<ClipPlayer>` + `<ClipDownloadButton>` in a modal, route their errors through callbacks.** Both accept `onError` (and `<ClipDownloadButton>` also accepts `onSuccess(blob)`); `SnapClip` threads them into the same inline error line that `requestClip` failures use. The SDK's components stay usable after disconnect, so the modal keeps working if the session ends mid-preview.
5. **No `getJwt` plumbing.** Both clip components auto-inherit the resolver from `<HeliosProvider getJwt={…}>` via React context (`@reactor-team/js-sdk` ≥ 2.10.1). That is the single source of truth for auth in this app — `SnapClip` doesn't need to know about the JWT route at all.

### The portal gotcha (Sonner toasts, headless modals)

The context-inheritance only works for components rendered **inside** the provider subtree. `SnapClip`'s modal is a normal child of the panel, so it inherits the resolver fine.

The trap is rendering clip UI through a React portal whose host lives _outside_ `<HeliosProvider>` — most commonly a Sonner `<Toaster />` mounted in `app/layout.tsx` as a sibling of `{children}`. The custom-toast tree has no `ReactorContext` in scope, the fallback returns `undefined`, and Coordinator answers the clip download with:

```
{"error":"Missing Authorization header"}
```

Fix: capture the resolver imperatively _inside_ the provider subtree, then thread it down as an explicit prop. The resolver outlives `disconnect()` by design, so the toast keeps minting fresh tokens even after the session ends.

```tsx
function HandlerInsideProvider() {
  // `requestClip` is a top-level store action; `internal.reactor`
  // stays in the escape-hatch slot for `getJwtResolver()` because
  // that one isn't lifted onto the store surface.
  const { requestClip, reactor } = useReactor((s) => ({
    requestClip: s.requestClip,
    reactor: s.internal.reactor,
  }));

  const onSnap = async () => {
    const clip = await requestClip(30);

    // Captured here — works because we're inside the provider.
    // The closure carries it across Sonner's portal boundary.
    const getJwt = reactor.getJwtResolver();

    toast.custom(() => <ClipReadyToast clip={clip} getJwt={getJwt} />);
  };
}
```

### hls.js is an optional peer

`<ClipPlayer>` plays HLS natively on Safari/iOS. On Chrome / Firefox / Edge it dynamically imports `hls.js` — which is why the example declares it as a direct dep (`hls.js@^1.6.0`). If `hls.js` isn't installed, the player surfaces an inline error and downloads still work; the dep keeps the preview path functional for the majority of users.

### Extending

The component takes optional props for `durationSeconds` (default 10), `filename`, and `label`. Most extensions are one prop:

```tsx
<SnapClip durationSeconds={30} label="Save 30s highlight" />
```

For multi-clip galleries, store an array of `Clip` instead of a single one, render a thumbnail per entry, and pass each to `<ClipPlayer>` / `<ClipDownloadButton>` on click. The headless [`useClipDownload`](https://docs.reactor.inc/api-reference/react-hooks#useclipdownload) hook is what to use if you want a custom progress UI instead of the default button.

### Full-session recordings

`requestRecording()` (no args, also on the store) grabs everything from the start of recording up to now, instead of a trailing window. Same `Clip` shape, longer manifest, larger MP4. Swap the call inside `SnapClip` if you want a "Save the whole session" button instead.

### Clips are short-lived

The URL on a `Clip` expires after a few minutes. Do not store `Clip` objects long-term, and do not hand `clip.playlistUrl` to your users for sharing. If you want a permanent link, download the MP4 (via `<ClipDownloadButton>` or `downloadClipAsFile(clip, null)` for a Blob) and host the result yourself.

### Clip downloads are social-media-ready (`@reactor-team/js-sdk` ≥ 2.10.1)

Behind the existing `<ClipDownloadButton>` / `reactor.downloadClipAsFile()` API the SDK now remuxes the fragmented MP4 the runtime ships into a flat MP4 with `start_time=0` and faststart layout, using `mp4box` as a bundled runtime dep. The transformation is `ffmpeg -c copy` style — no decode, no re-encode, the H.264 / AAC bitstream is bit-identical. The Blob you get back uploads cleanly to Twitter, Instagram, TikTok, YouTube; opens with `start_time=0` in QuickLook; and on the rare parse failure falls back silently to the previous fragmented bytes (logged via `console.warn`), so the download never fails outright. No knob, no API change — you get the right behaviour by being on 2.10.1+.

## Brand alignment — design tokens, not components

The app pulls Reactor's design tokens (fonts + brand colors) from `@reactor-team/ui`, but **does not** import its React components. Components ship interactive hooks under the hood — importing `<Button>` or `<CodeSnippet>` would force the consuming file into `"use client"` land. Design tokens have none of that baggage.

```css
/* app/globals.css */
@import "tailwindcss";

@theme {
  --font-sans: var(--reactor-font-sans);
  --font-mono: var(--reactor-font-mono);
  --color-brand: var(--reactor-color-light-gold);
  --color-brand-fg: var(--reactor-color-interstellar);
  --color-active: var(--reactor-color-flora-light);
}
```

```tsx
// app/layout.tsx
import "@reactor-team/ui/styles.css"; // fonts + brand CSS vars
import "./globals.css";
```

Use `bg-brand`, `text-brand`, `font-mono` etc. as plain Tailwind utilities — works in any component, server or client.

Reach for actual `@reactor-team/ui` components only when you need their behavior (e.g. a copy-on-click code block). Those usages are naturally Client Components anyway.

## Common mistakes when extending

1. **Reaching for `@reactor-team/js-sdk` directly.** Everything Helios-specific is on `@reactor-models/helios`. If you find yourself reaching for `useReactor((s) => s.internal.reactor)` for a Helios event or message, re-read the typed hooks list above. The one allowed exception is the recording surface — see [Capturing clips](#capturing-clips). And on 2.11.1+, even recording is a top-level store action (`s.requestClip` / `s.requestRecording` / `s.downloadClipAsFile`); reach for `s.internal.reactor` only for the few surfaces that aren't lifted yet (`getJwtResolver()`, raw `runtimeMessage` subscriptions).
2. **Aggregating events to reconstruct state.** Subscribe to `useHeliosState` and read fields off the snapshot. Stop folding `chunk_complete` + `generation_started` + `generation_paused` into your own boolean flags.
3. **Chaining `setImage + setPrompt + start` instead of using `setConditioning`.** Separate commands can be reordered on the wire — `start` slips past the still-resolving image upload and the first chunk renders with no image conditioning. When both pieces are known up front, use `setConditioning({ prompt, image })` for a single atomic message. Only fall back to `setImage` alone when the prompt arrives later from a different user action (the custom-upload flow).
4. **Overwriting the prompt on custom-image upload.** `set_image` is a conditioning tweak. Leave the user's prompt alone — let them type one in the textarea if they need to.
5. **Forgetting to clear the snapshot on disconnect.** The next session's UI will show stale state. Three lines of `useEffect` in any component that holds a snapshot.
6. **`if (snapshot?.started) return null` without a status check.** After disconnect, `snapshot.started` may still be true (stale until the effect clears it). Always gate on `status === "ready"` too.
7. **Connecting from a `useEffect` in your own component.** The Provider owns connection lifecycle. Don't fight it; configure it via the `connectOptions` prop instead.
8. **Importing `@reactor-team/ui` components into a Server Component.** They use hooks internally. Either keep them in Client Components, or use the design tokens via CSS vars instead.
9. **Single-line prompts and inconsistent evolutions.** The model needs paragraph-length prompts with explicit visual continuity. Short prompts produce choppy output; mismatched continuity makes scenes visibly reset.
10. **A new custom hook for one component.** Inline the pattern first. Extract when you have three call sites of the same logic.
11. **Breaking the file-per-phase rule.** A single component that does setup AND live work is hard to read. Split it.

## Checklist for new components

Before merging a new control or feature:

- [ ] Decided which phase it lives in (Setup, Live, or always-on)
- [ ] Early-return at the top matches that phase (`status === "ready" && snapshot?.started` to hide in live, etc.)
- [ ] If it subscribes to `useHeliosState`, it clears on disconnect via `useEffect`
- [ ] All interactive controls gate `disabled` on `status === "ready"`
- [ ] All event method calls use the typed wrappers (`setPrompt`, not `sendCommand("set_prompt", …)`) and are `await`ed
- [ ] If priming a session with prompt + image, uses `setConditioning({ prompt, image })` (atomic). If chaining a different slow conditioning step before `start()` that doesn't have an atomic wrapper, awaits its specific `*_accepted` message via a one-shot ref resolver
- [ ] Renders `command_error` somewhere visible (the existing `CommandError` component handles this automatically — don't suppress it)
- [ ] New prompts added to `app/lib/prompts.ts` are full paragraphs with continuity between `initial` and `evolutions`
- [ ] Brand colors via Tailwind utilities (`bg-brand`, `text-brand`), not hardcoded hex
- [ ] No imports from `@reactor-team/js-sdk` or `@reactor-team/ui` React components unless absolutely required (recording surface is the documented exception — see [Capturing clips](#capturing-clips))
