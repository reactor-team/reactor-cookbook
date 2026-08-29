---
name: building-lingbot-world-2-frontends
description: Extend this cloned LingBot World 2 example app — add new controls, scenes, motion patterns, or knobs on top of `@reactor-models/lingbot-world-2` without breaking the patterns the existing code uses. Covers the SDK's connection / events / messages model, the layered prompt system and its override store, the per-chunk `set_camera_pose` motion channel, the WASD + mouse-look driving model, the auth route, and clip capture.
---

# Building on this LingBot World 2 app

You've cloned this folder and now you want to extend it — a new control, a new scene, a new motion pattern, a different UX. This guide explains the patterns the existing code uses and the rules to follow so your additions feel native instead of bolted on.

All the code referenced below already exists in this folder. Read this guide alongside the source — especially [The camera-pose channel](#the-camera-pose-channel) and [Jump and crouch](#jump-and-crouch--the-button--event-model) before touching anything in the motion system.

## What LingBot World 2 actually is, in three sentences

LingBot World 2 is a **real-time interactive world model**. Given a starting image and a composed prose prompt, it streams an unending first/third-person video on a single track (`main_video`) — and while it generates, the client continuously steers it with movement actions (`set_move_longitudinal`, `set_move_lateral`), per-chunk camera deltas (`set_camera_pose`), and live prompt swaps (`set_prompt`).

The frontend's job is to (a) start the generation with a valid image + prompt, (b) translate keyboard/mouse/joystick input into those steering commands every chunk, and (c) keep the prose prompt in sync with what the user is doing.

## The four concepts you'll touch

| Concept        | What it is                                                                         | Hook / API                                                                                                                                                                                                                                                              |
| -------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Connection** | The lifecycle of the model session (`disconnected → connecting → waiting → ready`) | `useLingbotWorld2().status`, `.connect()`, `.disconnect()`                                                                                                                                                                                                              |
| **Events**     | Things you send TO the model. Always async.                                        | `useLingbotWorld2().setPrompt({...})`, `.setImage({...})`, `.setMoveLongitudinal({...})`, `.setMoveLateral({...})`, `.setCameraPose({...})`, `.setRotationSpeedDeg({...})`, `.setSeed({...})`, `.setAttnWindow({...})`, `.start()`, `.pause()`, `.resume()`, `.reset()` |
| **Messages**   | Things the model sends BACK to you — acks, chunk ticks, and the `state` snapshot.  | `useLingbotWorld2Message((m) => …)` (this app uses the catch-all; per-message typed hooks also exist)                                                                                                                                                                   |
| **Tracks**     | The model's video output, rendered as a live `MediaStreamTrack`.                   | `<LingbotWorld2MainVideoView />`                                                                                                                                                                                                                                        |

You almost never have to drop below this surface. If you find yourself reaching for `@reactor-team/js-sdk` directly, stop and re-read the typed hooks list — there's likely a typed hook you're missing. The one documented exception is the recording surface (see [Capturing clips](#capturing-clips) below), which is a base-SDK feature that the typed packages deliberately do not re-export.

Two commands the published schema doesn't declare yet (`set_kv_cache_reset`, `trigger_kv_cache_reset`) go through the raw `sendCommand("…", {...})` escape hatch — with a comment explaining why. When the schema catches up, migrate them to the typed methods and delete the comment.

## The UI phase model

A real-time session is a state machine, and the UI mirrors it with **two visible phases**:

```
       ┌──────────────┐   setImage → setPrompt → start   ┌────────────────┐
       │   WAITING    │ ───────────────────────────────▶ │   GENERATING   │
       │  (Setup UI)  │ ◀─────────────────────────────── │   (Live UI)    │
       └──────────────┘             reset                 └─────┬──────────┘
                                                               │ ▲
                                                          pause│ │resume
                                                               ▼ │
                                                         ┌────────────────┐
                                                         │     PAUSED     │
                                                         │   (Live UI)    │
                                                         └────────────────┘
```

| UI phase  | When                                    | What's visible                                                                                             |
| --------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Setup** | not generating (fresh page, post-reset) | Quick Start example cards · custom scene card · Start button with its blocker reason                       |
| **Live**  | `isGenerating === true`                 | WASD pad + joystick · mouse-look · jump/crouch · hold-key world events · prompt inspector · Advanced knobs |

Unlike the sibling examples (which split every phase into its own sidebar component), this app centralizes input handling in one component — [`LingbotWorldController.tsx`](../components/lingbot-world-2/LingbotWorldController.tsx) — because _every_ control feeds the same three output channels (movement actions, camera pose, composed prompt) and they must stay coherent. The controller returns `{ sidebar, controls }` and the page lays them out.

**When you add a new control, decide which channel it feeds first:**

- **A new steering input** (a key binding, a gamepad axis, a gesture) → extend the controller. It must participate in `sendCameraPoseChunk()` / the movement stacks so it composes with existing inputs instead of fighting them.
- **A new prompt ingredient** (a weather layer, a mood selector) → extend the layered scene model in [`lib/lingbot-world-prompts.ts`](../lib/lingbot-world-prompts.ts) and `composePrompt()`. Never call `setPrompt` from a random component — all prompt sends flow through `recomputePromptAndSend()` so the inspector and dedupe logic stay truthful.
- **A standalone panel** (stats, capture, diagnostics) → a separate component like [`SnapClip.tsx`](../components/SnapClip.tsx), gated on `status === "ready"`, dropped into the page layout.

## Auth — a memoizing `jwtToken` resolver + a scoped mint route

Two pieces work together: a Next.js GET route that mints a session-scoped JWT
server-side, and a `jwtToken` resolver on `<LingbotWorld2Provider>` that the SDK calls on
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
  if (cachedToken && Date.now() < cachedToken.expiresAtMs - TOKEN_REFRESH_SKEW_MS) {
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

`<LingbotWorld2Provider>` is initialized **without** `autoConnect`. The user clicks "Connect" so they see the `disconnected → connecting → waiting → ready` transitions. If you're shipping a polished product where you'd rather the connection happen on page load:

```tsx
<LingbotWorld2Provider jwtToken={fetchToken} connectOptions={{ autoConnect: true }}>
```

Just make sure your status indicator still surfaces the intermediate states (`connecting`, `waiting for GPU`) — sessions don't reach `ready` instantly, and you don't want users staring at an unexplained loading state.

## Messages — what the model sends back

The controller subscribes once with the catch-all `useLingbotWorld2Message` and switches on `msg.type`. The messages you'll handle:

| Message                                                | What it means / what the app does with it                                                                                                                                         |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state`                                                | Periodic full snapshot: `has_prompt`, `has_image`, `started`, `running`, `paused`, `camera_pose_active`. Mirrors into local state.                                                |
| `conditions_ready`                                     | Broadcast by `set_prompt` and `set_image` once each commits, so it reaches every connection. The `state` snapshot carries the same `has_prompt` / `has_image` picture.             |
| `generation_started`                                   | Generation began; carries `chunk_num`.                                                                                                                                            |
| `chunk_complete`                                       | **The heartbeat of the live phase.** Carries `chunk_index` and `active_action`. The app advances jump arcs, consumes crouch dips, and sends the next `set_camera_pose` from here. |
| `generation_complete`                                  | The run finished on its own.                                                                                                                                                      |
| `command_error`                                        | A command was rejected. **Always surfaced** — the app shows it as a 4-second toast (`errorToast`).                                                                                |
| `prompt_accepted` / `image_accepted`                   | Answers to `set_prompt` / `set_image`, so they reach **only the connection that sent the command** — read them off the awaited call. `image_accepted` carries the accepted `width` / `height`. |
| `generation_paused` / `resumed` / `reset`              | Answers to `pause` / `resume` / `reset`, so likewise **sender-only**. Clear per-session scene state off the resolved `reset()` call rather than waiting for the message.           |

Two rules to keep when extending:

1. **Clear everything on disconnect.** The controller has a `useEffect` on `status === "disconnected"` that resets every piece of session state — held keys, arcs, refs, previews. If you add new session state (a new ref, a new held-input), add its reset there too, or the next session starts haunted by the last one.
2. **Surface `command_error` for new commands automatically.** The existing toast handles any command, because `command_error` broadcasts. Don't reach for `try/catch` or `.catch()` around a command: they never reject, so a `catch` is dead code — the toast is the error path. A fire-and-forget command is written `void lw2.setSeed({ seed })`.

## Sending commands — where the result arrives

A command resolves when the model's handler has finished, and carries whatever
that handler answered with. **None of them reject:** a refusal arrives as a
broadcast `command_error` and resolves the call with `undefined`, and so does a
send that never completed, with the reason on `lastError`.

Which way the model answers decides where your code reads the result:

| The model | Reaches you | LingBot World 2 commands |
| --- | --- | --- |
| **answers** the command that asked | the awaited call's return value — and the **sending** connection's `message` event | `setPrompt`, `setImage`, `pause`, `resume`, `reset` |
| **broadcasts** to every connection | `useLingbotWorld2Message` and the per-message hooks | `state`, `chunk_complete`, `command_error`, `conditions_ready`, `generation_started` / `complete` |
| answers with **nothing** | the await resolves `undefined`; nothing reaches the message event | `start`, `triggerKvCacheReset`, and the movement / camera-pose / knob setters |

An answer is **addressed**: the runtime sends it to the one connection whose
command earned it, correlated by request id. On that connection it resolves the
awaited call *and* raises the `message` event, so a `msg.type ===
"image_accepted"` branch in the catch-all listener does run. It just runs only
here — a second client in the session never sees it — and it cannot tell you which
of several in-flight `setImage` calls it belongs to. Read the answer off the await
instead, and keep the listener for what the model genuinely broadcasts.

The canonical start flow lives in `applyScene()`:

1. Clear all held inputs (never carry a held key across a scene switch).
2. If generating, `await reset()` first. The resolved await **is** the
   confirmation the model cleared — there is no need to sleep after it.
3. `await uploadFile(file)` → `FileRef`, then
   `const accepted = await setImage({ image: ref })`. That resolves once the model
   has decoded the image, and carries the accepted `width` / `height`.
4. `await setPrompt({ prompt: composePrompt(scene, …) })`.
5. `await start()`.

```tsx
async function startScene(scene: StructuredScene, file: File) {
  const ref = await uploadFile(file);
  const accepted = await setImage({ image: ref });
  if (!accepted) return; // refused; the command_error toast already fired
  setImageInfo({ w: accepted.width, h: accepted.height });
  await setPrompt({ prompt: composePrompt(scene, false, []) });
  await start();
}
```

No settle delays anywhere in that chain, and no ack plumbing: each await already
waits for the handler it named.

### Movement commands are state-based — always send the neutral value on release

`set_move_longitudinal` / `set_move_lateral` hold their last value until you change it. Send `"forward"` and never send `"idle"` and the world keeps scrolling forever. The controller's key handlers maintain **key stacks** (`moveLStackRef` / `moveLatStackRef`) so overlapping presses resolve correctly (hold W, tap S, release S → back to forward, not idle). When you add a new binding, push/pop the stack — don't send raw values from your own handler.

## The camera-pose channel

This is the model's signature capability and the part most worth understanding before extending. Everything the driving UI does flows through three commands:

| Command                                      | Carries                                             | Notes                                                                  |
| -------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| `set_move_longitudinal` / `set_move_lateral` | WASD **action** (`forward`/`back`/`strafe`/`still`) | a discrete action string, not a vector                                 |
| `set_camera_pose`                            | per-step 6-DoF **deltas** `[rx,ry,rz,tx,ty,tz]`     | the interesting channel; drives look / roll / jump / crouch / joystick |
| `set_prompt`                                 | the composed **text prompt**                        | recomposed whenever anything changes                                   |

### The `camera_pose` contract (the physics everything obeys)

Four wire-observable facts shape every decision in the motion system:

1. **Granularity is per-step within a chunk, not per-chunk.** The app sends `CHUNK_LATENTS` (3) deltas per chunk — one per generation step, ≈12 pixel frames total. Send 3 deltas and each steers its own step; send 1 and the model repeats it across the chunk. The app always sends 3 so jump arcs and crouch dips can shape motion inside a single chunk.
2. **Axis convention is y-down: `−ty` = UP, `+ty` = DOWN.** In code this is `JUMP_UP_SIGN = -1`; "up" intent maps to a negative `ty`.
3. **Translation magnitude is normalized away per chunk.** Sending `ty = 0.3` vs `ty = 100` gives the same result — only the **sign** (direction) and the **within-chunk shape** survive. Consequences: a motion's _size_ is its **step count (duration)**, not a magnitude; a smooth velocity envelope cannot carry across chunks. Motion is therefore authored as small integer intent patterns (`+1` up / `0` still / `-1` down), not velocity curves.
4. **Rotation OVERRIDES the arrow-key look; translation ADDS to WASD movement.** A pose with `ty = down` and zero rotation does not erase forward motion — the model sums the WASD action in. This is why crouch-walking works without re-sending forward, and it drives every design decision in the jump/crouch system.

One consequence worth a warning: **a sustained vertical `hold` (jump or crouch `hold` mode) freezes arrow-key look.** The model's override rule applies whenever a pose is active, even when it carries zero rotation. Mouse-look is unaffected (it does carry rotation).

### One sender per chunk

The single send site is `sendCameraPoseChunk()`, called on every `chunk_complete` (and once when a control engages). Mouse-look, roll, orbit, joystick, jump arcs, and crouch dips all deposit their intent into refs, and that one function composes them into the next chunk's pose — rotation (`rx,ry,rz`) and horizontal translation (`tx,tz`) uniform across the 3 steps, only `ty` authored per-step. When nothing is active it sends an empty pose once, handing rotation back to the arrow keys and translation back to WASD. **Never add a second `setCameraPose` call site** — two senders per chunk means the second overwrites the first.

If you add a new motion (a dodge, a lean, a head-bob), follow the jump-arc shape: author it as a per-step intent pattern, store it in a ref, consume it inside `sendCameraPoseChunk()`, and advance/expire it on `chunk_complete`.

## Jump and crouch — the button → event model

Each button represents a triggerable **event**, and each event is a **(camera pose, prompt) pair**: a bit of motion on the `camera_pose` channel plus a sentence woven into `set_prompt`, delivered together so the prose matches the motion. Press and release are **independent triggers** — crouch fires a "down" dip + `crouchPrompt` on press and a mirrored "up" dip + `standPrompt` on release, making a button a small state machine rather than a momentary toggle.

### Symmetry — return to origin

Because magnitude is normalized away (fact 3 above), "equal up and down" means **equal counts of up-steps and down-steps**. `#up == #down` lands the camera back at its starting height; more up than down and the character ends up higher (never fully lands), more down and it sinks. This is why the default charge-level arcs are symmetric — L1 is `[1, 0, -1]`, L2 is `[1,1,0, -1,-1,0]`, L3 is `[1,1,1,1, 0, -1,-1,-1,-1]` (the `0`s are the hang at the apex and don't affect balance). Hand-edit an asymmetric pattern if you want deliberate vertical drift; just know that's what you're authoring.

### Trigger semantics — one-shot, held, and charge

- **Jump is one-shot and locked while airborne.** `onJumpDown` returns early while an arc is in flight (`jumpArcRef.length > 0`) — no double-jump.
- **Crouch can be held.** The press dip fires only on the idle → held transition (key auto-repeat is ignored); releasing fires exactly one release dip. Holding C while walking gives a crouch-walk because translation is additive.
- **Charge is hold-to-charge.** While held, the level meter steps through `NUM_CHARGE_LEVELS` discrete levels (dwelling `LEVEL_DWELL_MS` on each); nothing is emitted until release, which fires that level's arc + `jumpPrompt`.

Both dips last one chunk and are consumed on the next `chunk_complete`, so they're inherently transient.

### Modes, editors, and per-scene prompts

Jump and crouch each have a mode switch: `hold` (sustained translation while held — the way to move straight up/down), `prompt` (sentence only, no pose), and `charge` / `camera` (the arc/dip patterns). The patterns are **hand-editable per-step grids** (`Lvl` buttons for jump levels, `✎ dip` for crouch press/release) — click a cell to cycle `↑ up (+1) → ↓ down (−1) → · still (0)`; the number of still cells _is_ the hang duration. Edits persist in `localStorage` (`chargePatterns` / `crouchPatterns`), never hardcoded in logic.

The jump/crouch/stand sentences are **scene fields, not code constants** — `StructuredScene.jumpPrompt` / `crouchPrompt` / `standPrompt`, editable per example in the scene editor's "Vertical" tab. At runtime `recomputePromptAndSend()` picks the active vertical sentence and the inspector shows it as the vertical segment.

One keyboard gotcha to preserve: **crouch is on `C`, not `Ctrl`.** macOS reserves `Ctrl`+arrows for Mission Control and swallows the keydown before the page sees it — a `Ctrl`-held crouch would silently kill arrow-look, and no `preventDefault` can override a system shortcut. Don't move it back.

### Motion-system code map

| Concern                                       | Where                                                                                    |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| All input handling, pose emission, modes      | [`LingbotWorldController.tsx`](../components/lingbot-world-2/LingbotWorldController.tsx) |
| Per-chunk pose builder (per-step `ty`)        | `sendCameraPoseChunk()`                                                                  |
| Prompt composition + active vertical sentence | `recomputePromptAndSend()`                                                               |
| Jump press/release + charge meter + arc       | `onJumpDown` / `onJumpUp` / the charge-meter `useEffect`                                 |
| Arc / dip advancement + consumption           | `case "chunk_complete"` in the message handler                                           |
| Pattern defaults + grid editors               | `defaultChargePattern` / `defaultCrouchPatterns`, `cycleChargeCell` / `cycleCrouchCell`  |
| Prompt model + `composePrompt`                | [`lib/lingbot-world-prompts.ts`](../lib/lingbot-world-prompts.ts)                        |
| Prompt segments (incl. vertical)              | [`prompt-segments.ts`](../components/lingbot-world-2/prompt-segments.ts)                 |

Tunables sit at the top of the controller: `CHUNK_LATENTS`, `NUM_CHARGE_LEVELS`, `LEVEL_DWELL_MS`, `JUMP_SPEED`, `JUMP_UP_SIGN`, `CROUCH_DIP`.

## The layered prompt system

The model only ever sees one prose string, but the app authors it in layers — that's the `StructuredScene` in [`lib/lingbot-world-prompts.ts`](../lib/lingbot-world-prompts.ts):

| Layer                 | Role                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `base`                | World identity: subject, environment, style. Always present.                                 |
| `camera` / `movement` | Each has `static` and `dynamic` variants — selected by whether the user is currently moving. |
| `events[]`            | Hold-key detail clauses (keys 1–9). Stack while held, drop on release.                       |
| vertical              | The jump / crouch / stand sentence while those controls are engaged.                         |

`composePrompt(scene, isMoving, heldSlots, verticalPrompt)` flattens the active selection to prose. `recomputePromptAndSend()` calls it whenever any input changes, dedupes against the last sent string, and sends `set_prompt`. The prompt therefore always narrates what the controls are doing — that text/motion coherence is why driving feels responsive.

Rules when extending:

1. **All prompt mutations go through `recomputePromptAndSend()`.** If you send prompts from anywhere else, the dedupe ref (`lastSentPromptRef`) and the inspector both start lying.
2. **New layers get a segment in [`prompt-segments.ts`](../components/lingbot-world-2/prompt-segments.ts)** so the Show-prompt inspector and the editor preview render them. The inspector is the debugging surface for "why does the world look like that" — keep it complete.
3. **Write event clauses as single present-continuous sentences about the environment** ("rain begins to fall…"), not about the subject. They must compose onto any base without contradicting it.
4. **Keep prompts paragraph-length with explicit camera framing.** Short prompts produce choppy, unstable output. Look at the bundled scenes in [`lib/lingbot-cases/`](../lib/lingbot-cases/) for the density to match.

### Scenes, examples, and the override store

Curated examples live as JSON in [`lib/lingbot-cases/`](../lib/lingbot-cases/) (one file per scene) and are loaded by [`lib/lingbot-cases-examples.ts`](../lib/lingbot-cases-examples.ts). Starting images ship in [`public/lingbot-cases/`](../public/lingbot-cases/).

**Adding a new example = one JSON file + one image + one registry entry.** No component changes.

User edits to any example go into an **override store** (`overrides` state, persisted to `localStorage`), keyed by example id. The pristine JSON constants are never mutated. Rules the store enforces — keep them if you touch it:

- An override that is byte-equal to the pristine scene is dropped, not stored (otherwise cards show a misleading "edited" badge).
- Overrides survive `reset()` and disconnects — they're presets, not session state.
- `effectiveSceneFor(id)` is the only read path; it clones so callers can mutate freely.

## Capturing clips

The Reactor base SDK exposes a recording surface that works for every model: ask for the last N seconds of the live stream, get back a `Clip`, and either preview it with `<ClipPlayer>` or download it with `<ClipDownloadButton>`. The model SDK does not own this — it lives on `@reactor-team/js-sdk` because it is the same call for every model with recording enabled.

The app ships a drop-in [`components/SnapClip.tsx`](../components/SnapClip.tsx) panel that wires this together: a "Capture" button that calls `requestClip(durationSeconds)` off the store, opens a modal with the SDK's preview player, and offers an MP4 download. It is **model-agnostic** — the same file ships (modulo theme classes) in every example.

### When to reach for `@reactor-team/js-sdk` directly

The default rule still applies: do everything via `@reactor-models/lingbot-world-2`. But the typed package only re-exports model-specific surface (events, messages, the typed provider/hook). The recording surface is base-SDK only, so for that one feature you import directly:

```tsx
import {
  ClipDownloadButton,
  ClipPlayer,
  RecordingError,
  useReactor,
  type Clip,
} from "@reactor-team/js-sdk";
```

When you scaffold a new component, ask: "Does this depend on LingBot-World-2-specific events, messages, or commands?" If yes → typed package only. If no, and it would work the same on any model (recording, generic stats, generic connection state) → `@reactor-team/js-sdk` is fine.

### The pattern

1. **Destructure the recording action off the store.** `useReactor((s) => s.requestClip)` — `requestClip`, `requestRecording`, and `downloadClipAsFile` are first-class store actions.
2. **Gate on connection status.** Return `null` when `status !== "ready"`, so the panel disappears on disconnect like every other live-only control.
3. **Catch `RecordingError`.** Typed reasons: `DISCONNECTED`, `RECORDER_DISABLED`, `INVALID_DURATION`, `REQUEST_TIMEOUT`. Surface them inline.
4. **Compose `<ClipPlayer>` + `<ClipDownloadButton>` in a modal**, routing their `onError` / `onSuccess` callbacks into the same inline error line.
5. **No auth plumbing.** Both clip components inherit the resolver from `<LingbotWorld2Provider jwtToken={…}>` via React context. The one case where you'd pass it explicitly is a portal rendered _outside_ the provider subtree (e.g. a Sonner toast in `app/layout.tsx`) — capture `reactor.getJwtResolver()` inside the provider at action time and thread it down as a prop.

### hls.js

`<ClipPlayer>` plays HLS natively on Safari/iOS. On Chrome / Firefox / Edge it dynamically imports `hls.js` — which is why the app declares it as a direct dep (`hls.js@^1.6.0`). If it weren't installed, the player would surface an inline error (downloads still work); the dep keeps the preview path functional for the majority of users.

### Extending

`<SnapClip durationSeconds={30} label="Save 30s highlight" />` — most extensions are one prop. `requestRecording()` (no args, also on the store) grabs everything since recording started instead of a trailing window; swap the call for a "Save the whole session" button. The headless `useClipDownload` hook is what to use for a custom progress UI.

**Clips are short-lived.** The URL on a `Clip` expires after a few minutes. Don't store `Clip` objects long-term and don't hand `clip.playlistUrl` to users for sharing — download the MP4 and host the result yourself.

## Brand alignment — tokens through the theme, components from `components/ui`

The app pulls Reactor's design tokens (fonts + brand colors) from `@reactor-team/ui`'s stylesheet and maps them into the shadcn theme variables in [`app/globals.css`](../app/globals.css) — `--primary`, `--background`, `--font-sans`, etc. all resolve to Reactor brand values. The reusable primitives in [`components/ui/`](../components/ui) (button, input) are shadcn-generated and read those variables.

Rules:

- **Use the theme utilities** (`bg-primary`, `text-muted-foreground`, `font-mono`) or the existing `components/ui` primitives. Don't invent parallel color systems with raw hex values.
- **Don't import `@reactor-team/ui` React components.** They use hooks internally — importing one into a Server Component (like `SetupRequired`) dies at runtime, not build time. The stylesheet import gives you everything you need.
- The app is dark-only: the `dark` class is set on `<html>` in [`app/layout.tsx`](../app/layout.tsx) so server-rendered pages get the dark tokens too. Don't move it into a client effect.

## Common mistakes when extending

1. **Reaching for `@reactor-team/js-sdk` directly for model work.** Everything model-specific is on `@reactor-models/lingbot-world-2`. The one allowed exception is the recording surface — see [Capturing clips](#capturing-clips).
2. **Adding a second `setCameraPose` call site.** One sender per chunk (`sendCameraPoseChunk()`), fed by refs. A second sender silently overwrites the first and motion becomes non-deterministic.
3. **Sending prompts outside `recomputePromptAndSend()`.** Breaks dedupe and makes the Show-prompt inspector lie.
4. **Forgetting the neutral value on key release.** Movement axes hold their last value. Pair every `"forward"` with an `"idle"`, and use the key stacks so overlapping presses resolve correctly.
5. **Forgetting to reset new session state on disconnect.** Extend the `status === "disconnected"` effect in the controller with every new ref/state you add.
6. **Keyboard listeners that hijack typing.** Every handler early-returns when the event originates in an `INPUT` / `TEXTAREA` / contentEditable. Copy that guard into new bindings, and `preventDefault()` on handled keys so arrows don't scroll the page.
7. **Asymmetric vertical patterns by accident.** Equal up-steps and down-steps returns the camera to its starting height; unequal counts drift it. Fine if intentional — see [Symmetry](#symmetry--return-to-origin).
8. **Overwriting the user's scene edits.** Reads go through `effectiveSceneFor(id)`; writes go into the override store. Never mutate the pristine constants from `lib/lingbot-cases/`.
9. **Blocking `start()` on nothing.** `start` requires both a prompt and an image; the model rejects it with `command_error` otherwise. The Start button's `startBlockerReason` tells the user which half is missing — keep that pattern for new preconditions.
10. **Importing `@reactor-team/ui` React components into Server Components.** Runtime error, not build error. Use the theme tokens.
11. **One-line prompts.** Paragraph length minimum, explicit camera framing, present-continuous event clauses. The bundled scenes are the reference density.
12. **Storing `Clip` objects or sharing clip URLs.** They expire in minutes. Download the MP4 if you need an artifact.

## Checklist for new components

Before merging a new control or feature:

- [ ] Decided which channel it feeds (steering → controller; prompt → scene model + `composePrompt`; standalone → own component)
- [ ] Interactive controls gate on `status === "ready"` (and generation state where relevant)
- [ ] New steering inputs compose through `sendCameraPoseChunk()` / the movement stacks — no new send call sites
- [ ] New prompt ingredients render in the inspector via `prompt-segments.ts`
- [ ] All event calls use typed methods where the schema declares them; raw `sendCommand` only with a comment saying why
- [ ] New session state resets in the disconnect effect (and on `generation_reset` where appropriate)
- [ ] Keyboard handlers ignore events from inputs/textareas and `preventDefault()` handled keys
- [ ] Key-release sends the neutral value for any state-based axis
- [ ] `command_error` still surfaces (don't swallow it)
- [ ] New scenes are paragraph prompts with explicit camera framing; new event clauses are single environmental sentences
- [ ] Colors via theme utilities / `components/ui` primitives, not raw hex
- [ ] No `@reactor-team/js-sdk` imports outside `components/SnapClip.tsx` (recording is the documented exception)
