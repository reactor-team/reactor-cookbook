---
name: building-ltx2-frontends
description: Extend this cloned LTX example app — add controls, presets, or take-management UI on top of the Reactor JS SDK without breaking the patterns the existing code uses. Covers the typed @reactor-models/ltx2 client, the frozen-take/mutable-session model, reading valid_commands and queued_changes instead of deriving them, the stop-is-a-warm-restart remix loop and the take-chaining pattern behind a continuous performance, the avatar-image acknowledgement race, TTFF measurement and mid-take stall detection, crop-before-upload, capacity contention and idle billing, and the auth route.
---

# Building on this LTX app

You've cloned this folder and now you want to extend it. This guide explains the
patterns the existing code uses and the rules to follow so your additions feel
native instead of bolted on.

All the code referenced below already exists in this folder. Read this alongside
the source — especially [The take is frozen, the session is not](#the-take-is-frozen-the-session-is-not)
before you touch anything in the command path.

## What ltx2 is, in three sentences

LTX is a **streaming talking-head model**. The client uploads a still
image and sets a speech script; the model generates the voice and the lip-synced
video together and streams both back over WebRTC, window by window, at 640×352
and 24 fps. The frontend's job reduces to (a) collecting six conditions, (b)
sending `start` and the transport commands, and (c) mirroring the model's
`state_update` snapshot into the UI.

There are **no inbound tracks**. The face is a file upload; the speech is
generated from text. If you are looking for where to publish a camera, there
isn't one.

## The typed client

The typed client is `@reactor-models/ltx2`, generated from the model's
schema by `js-sdk-codegen`. It provides the `Ltx2Model` class and
command/message types plus the React surface (`Ltx2Provider`,
`useLtx2`, one hook per message, `<Ltx2MainVideoView>`), and only
depends on `@reactor-team/js-sdk`.

Import typed methods from it rather than reaching for raw `sendCommand("…")`
strings. The transport commands all pass through one choke point,
`runTransport` in `Ltx2App.tsx`, which is where the TTFF clock is
anchored on `start`; its `Record<TransportCommand, …>` map is exhaustive, so a
command added to the union without wiring there is a type error.

## The four concepts you'll touch

| Concept        | What it is                                                               | Hook / API                                              |
| -------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| **Connection** | The session lifecycle (`disconnected → connecting → waiting → ready`)    | `useLtx2()` → `status`, `connect`, `disconnect`         |
| **Commands**   | Things you send TO the model. Always async.                              | `useLtx2()` → `setScript({…})`, `start()`, `pause()`, … |
| **Messages**   | Things the model sends BACK — the `state_update` snapshot, acks, errors. | `useLtx2StateUpdate(…)`, `useLtx2CommandError(…)`       |
| **Tracks**     | `main_video` and `main_audio`, both outbound. Nothing inbound.           | `useLtx2()` → `tracks`                                  |

## The take is frozen, the session is not

Most Reactor models are continuously steerable: change the prompt mid-stream and
it lands at the next chunk. This one draws the line differently, and the
distinction is easy to get backwards. **The take in flight is immutable.** It is
generated from the conditions as they stood at `start`, and nothing you send
alters what you are currently watching. **The session's conditions are not
immutable.** All six setters stay valid during a run. The model accepts the
change, acks it as usual, and applies it to the next take.

So there are two different questions a component can ask, and the model answers
both itself:

| Question                                  | Field                         | Helper            |
| ----------------------------------------- | ----------------------------- | ----------------- |
| Would this command be accepted right now? | `state_update.valid_commands` | `validCommands()` |
| Will this field land now, or next take?   | `state_update.queued_changes` | `isQueued()`      |

Both helpers live in `app/lib/machine.ts` and both are thin on purpose.
`validCommands()` returns the server's list with exactly one addition, the thing
a snapshot cannot know:

```ts
if (status !== "ready") return new Set(); // no session at all
return new Set(ui.validCommands as Command[]);
```

**Keep it that way.** Every component asks these two functions; none of them
hard-codes an assumption about what is mutable when, and none re-derives the
state machine from `ui.generating`. Anything absent from `valid_commands` comes
back as `command_error`, so the list is not advice, it is the contract.

This is the second version of this file. The first one reproduced the state
machine client-side because the model did not report it, and every rule in it
became wrong when the model started to. If you find yourself writing
`ui.generating && …` in a component, you are rebuilding the thing that already
got deleted once.

Note `ui.ready`: the model reports "an image and a script are both set" as a
single flag rather than making clients recompute the precondition. Use it rather
than checking the two fields yourself.

### Queued, not staged

The `queued` chips are a **label on server state**, not a client-side buffer.
When you edit a field mid-run, the value goes to the model immediately and the
next snapshot carries it back; `queued_changes` simply names the fields whose
new values will not be heard until the next take. This is why `<TakePanel>` has
no Apply button and holds no pending-edit state: there is nothing to flush.

The one piece of local state it does keep is a per-field dirty set, so an
arriving snapshot cannot clobber half-typed input. That is a text-input concern,
not a staging mechanism. Note the resync effect's comment: a queued value is
already the value in the snapshot, so a run in flight needs no special case.
What the model will use next is what shows.

A trap worth naming: the preset rail is blurred and made `pointer-events-none`
during a run, and it would be easy to give the take panel the same treatment for
visual consistency. **Don't.** The preset rail is inert because a preset ends in
`start` and there is genuinely nothing valid to click. The take panel is the
opposite: every control in it is live mid-run, and disabling it would throw away
the model's best feature.

### Stop is a warm restart

`stop` ends the take and keeps every condition on the session: the image, the
script, the prompt, the pace, the seed, the duration. Nothing needs
re-sending, so the tightest loop the model supports is a remix:
`setScript({ script })`, then `start()`. The public demo at reactor.inc is
built entirely on that loop. Its production code adds one refinement worth
copying into any macro you write: it remembers the last value it sent for each
condition and skips the `set_*` for anything unchanged. Re-sending an
unchanged value is harmless, but every send is a round-trip plus a snapshot
echo, and the noise buries the command that mattered.

## Chaining takes into a continuous performance

The public demo at reactor.inc keeps an avatar talking for minutes at a time.
There is no long-generation mode behind that: it is the warm restart above,
run in a loop, with new script chunks written while the current take plays.
The content source is your product's business (an LLM, a playlist, a queue of
user submissions); the loop itself is model mechanics, and these are its
load-bearing details:

- **Drive the loop off the snapshot, never off your own sends.** The signal
  that a take ended is `generating` flipping false in `state_update`. When it
  does and a chunk is waiting, send `set_script` and `start`; the session is
  warm, so nothing else needs sending.
- **Settle before starting.** `start` is refused while a run is in flight. To
  cut a running take short, send `stop`, then poll the snapshot until
  `generating` reads false, on a bounded wait (the demo gives it 4 seconds),
  before the next `start`.
- **Guard every `start` with a grace timer.** A take that never starts
  streaming must not wedge the loop. The demo waits 25 seconds, then abandons
  that chunk and chains the next.
- **Know how much speech is left.** For the running take it is
  `effective_seconds - seconds_sent`; before the model reports
  `effective_seconds`, estimate from the script (`words / wpm * 60`). Ahead is
  that plus the same estimate over everything queued.
- **Refill early, write small.** The demo asks its writer for the next chunk
  whenever less than ~12 seconds of speech is ahead, and keeps chunks to two
  or three sentences that end where a sentence ends. A take boundary is a hard
  cut, and a cut inside a clause is heard.
- **Pin the conditions once.** The portrait, prompt, pace and seed are set
  before the first take and never re-sent, which is what keeps one face and
  one voice across every cut.
- **An interruption outranks the queue.** When the person redirects the
  conversation, drop every queued chunk; a line written for a conversation
  that has moved on is worse than the seam dropping it leaves. Let the take on
  stage keep playing while the reply is written, then stop and cut in the
  moment it is ready.

What this section deliberately leaves out is the writing side. The demo pairs
this loop with a realtime language model and a microphone, and none of that is
Reactor surface: the model streams the performance, and whatever supplies the
words is a choice this example does not make for you.

## The snapshot is the source of truth

`state_update` arrives on connect and after every observable change, carrying
the whole picture. `reduce()` (`app/lib/state.ts`) projects it into
`Ltx2UiState`, returning the previous object when nothing changed so React
can bail out — this model emits a snapshot after every window, so that bail-out
is doing real work.

Two rules:

1. **Only `state_update` mutates the reducer.** The discrete acks
   (`script_accepted`, `wpm_accepted`, …) are the correlated answers to the
   commands that earned them, so they never reach a listener at all — and each is
   followed by a snapshot carrying the same information anyway. Reconstructing
   state from acks would be a second, racier path to the same place.
2. **Clear session state on disconnect.** `Workspace` has an effect on
   `status === "disconnected"` that resets the reducer, the pending preset, the
   stage, and the TTFF anchor. The SDK emits no final `state_update`, so without
   it a reconnect renders the previous session's conditions.

`reset` gets one extra piece of handling: it clears every condition server-side,
but the take panel holds local drafts for fields being edited (so an incoming
snapshot can't clobber half-typed input). Those drafts would survive the reset.
The nonce that keys `<TakePanel>` is bumped to remount it and drop the drafts with
the state they mirrored — and it is bumped from the **resolved `reset()` call**,
not from `useLtx2GenerationReset`. `generation_reset` answers `reset`, so that
hook only ever fires on the connection that sent it; driving cleanup from the
resolved await keeps it tied to the call that caused it, and keeps working if a
second client attaches.

## Where a result arrives: the call, or a subscription

This is the subtlest thing in the app, and nothing about getting it wrong is a
compile error.

A command resolves when the model's handler has **finished**, and carries whatever
that handler answered with. Which way the model answers decides where you read it:

| The model | Reaches you | LTX commands |
| --- | --- | --- |
| **answers** the command that asked | the awaited call's return value — and the **sending** connection's `message` event | `setAvatarImage`, `setScript`, `setPrompt`, `setWpm`, `setSeed`, `setDurationSeconds`, `pause`, `resume`, `reset` |
| **broadcasts** to every connection | the per-message hooks | `state_update`, `command_error`, `generation_started` / `stopped` / `failed` / `complete`, `window_progress` |
| answers with **nothing** | the await resolves `undefined`; nothing reaches the message event | `start`, `stop` |

An answer is **addressed**: it goes to the one connection whose command earned it,
correlated by request id. There it resolves the awaited call *and* raises the
`message` event, so `useLtx2ScriptAccepted` and friends do fire — but only on this
connection, and with no way to tell which in-flight call they answer.

A **refusal is not an answer.** The handler broadcasts `command_error` and returns
without a value, so the awaited call resolves `undefined` and every connection
learns the reason through `useLtx2CommandError`. That is why `undefined` from a
reply-declaring command is the case to test for, and why the error banner is a
subscription rather than something read off a call.

> This is the contract from **model release 5.0.2** on. Up to 5.0.1 eight
> handlers *returned* `command_error` where their annotation promised the
> accepted message, so a refusal resolved the call truthy and a typed client
> unwrapped it as the success type. If you ever point this app at an older
> deployment, treat a reply whose `type` is not the expected `…_accepted` as a
> refusal.

### `set_avatar_image` is why this matters most

The model fetches and decodes the upload inside its handler, and a `start` racing
in behind an undecoded image generates the take with the **previous face** — which
looks like a caching bug and is maddening to chase.

Awaiting the command is what prevents that, because it resolves only once the
handler has run:

```ts
const ref = await uploadFile(file, { name });
const errorBefore = lastErrorRef.current;
const accepted = await sendAvatarImage({ avatar_image: ref });
if (accepted) return true;
// undefined: refused (command_error already surfaced) or the send never
// completed — only an error that appeared SINCE the snapshot is this call's.
return false;
```

Two things follow, and both are load-bearing:

1. **Do not build a message-waiter for this.** An earlier version of this app kept
   a module-scoped waiter registry that parked a predicate before sending and
   raced `avatar_image_accepted` against a `state_update` fallback, guarded on
   whether an image had been set before — because `has_avatar_image` is a **level,
   not an edge**, and this model emits a snapshot after every window, so an
   unguarded predicate resolved instantly against a snapshot describing the
   *previous* face. All of that existed to answer "has the handler run yet", which
   the await now answers directly. The registry is gone; do not reintroduce it.
   Parking a promise on `avatar_image_accepted` would technically resolve — an
   answer does raise the sending connection's `message` event — but it resolves
   for *any* `setAvatarImage` on this connection, which is the ambiguity the
   registry's guards existed to paper over. The await is tied to one call.
2. **Never `start` on an unconfirmed image.** Awaiting is only half of it —
   something has to hold Start for the length of the await. `setAvatarImage()`
   raises `imagePending` itself, in a `try/finally`, rather than leaving each call
   site to remember: the crop modal fires it and forgets it
   (`void onAvatarImage(…)`), so a gate at the call sites is one that eventually
   gets missed. `directPreset` additionally returns early when confirmation fails,
   and `presetPending` holds Start for the whole macro. Add an upload path of your
   own and it inherits the hold for free.

`imagePending` also stays a single boolean, which is why the panel allows one
upload at a time. Two concurrent uploads would each get their own correlated
answer and so could not confuse each other's confirmations, but the first to
finish would lower the shared flag while the second is still decoding.

### Telling a refusal from a bodyless answer

`undefined` means the model refused, **or** the send never completed. Telling them
apart needs `lastError`, which is a persistent record that success never clears —
so compare it across the call rather than reading it bare; only an error that
appeared since the pre-call snapshot belongs to this call. The app mirrors
`lastError` into a ref for exactly this, because the store field captured in an
async closure is a render-time snapshot. When `lastError` did not move, the model
refused and `command_error` already carries the reason.

## Time to first frame

The model is windowed and bidirectional, not frame-causal: nothing streams until
the leading window has denoised and decoded. Measuring that honestly takes care.

- **t0** is the moment `start` goes on the wire (`markStartSent`).
- **The clock stops** at the first newly-composited frame, observed with
  `requestVideoFrameCallback` on the `<video>` element — measured at the
  display, not inferred from a message.
- **The measurement must be armed by `generation_started`.** The WebRTC track
  stays live between takes and keeps compositing frames while the model is idle.
  Stopping the clock on the next frame after `start` therefore measures nothing:
  it reads a few milliseconds instead of a few seconds. This is why
  `ttffArmed` exists.

Expect roughly 3 s warm and considerably more on a cold pod.

This is also why `Stage.tsx` owns its own `<video>` rather than using the
generated `<Ltx2MainVideoView>`: that, plus the need to carry
`main_video` and `main_audio` in **one** `MediaStream`. The two tracks share a
sample clock; play them from separate elements and they drift.

## The stream can stall mid-take

The snapshot arrives over the data channel and the frames over the media
transport, and the two fail independently: a take can die mid-sentence while
`state_update` keeps reporting `generating: true`. No message announces it.
The only client-side signal is the one TTFF already uses:
`requestVideoFrameCallback` goes quiet.

`Stage.tsx` watches for that. While the snapshot says frames should be flowing
(`generating`, not `paused`, and `seconds_sent > 0`, so warm-up never counts),
a new frame should composite many times a second; when none arrives for 8
seconds the stage shows a stalled notice. Recovery is the transport the user
already has: `stop` stays in `valid_commands` for the whole run and the
conditions survive it, so a fresh take is one Stop and one Start away. The
threshold is the one the production demo runs.

## Crop before upload

The model fits whatever you upload to its 640×352 canvas — a wide frame. Hand it
an ordinary portrait photo, taller than it is wide, and the fit takes the top of
the head off. Since the avatar image defines the face for the entire take, that
is not a subtle degradation.

`CropModal.tsx` therefore sits between the file picker and the upload: it offers
the largest 640:352 region that fits, defaults the framing from the browser's
`FaceDetector` where available (top-center otherwise, since faces sit in the top
third of portraits), lets the user drag, and uploads only those pixels. If you
add another way to supply a face, route it through the same modal.

Cropping is one of two valid answers. The reactor.inc sandbox letterboxes
instead: it scales the whole image onto the wide canvas and fills the margins
with black, trading bars for keeping every pixel. Both work because both hand
the model an image already in its aspect; what fails is handing it a tall
portrait and letting the server's fit decide. Pick one and route every
face-supplying path through it.

## Capacity contention

A session holds a whole B200, so a deployment may serve **one session at a
time**. The second person to open the app gets `429 no available capacity`, and
without handling that reads as "the app is broken".

`StatusBadge.tsx` prints the error verbatim under the connection row and leaves
the Connect button in reach. Keep the verbatim line: for an engineering audience
it is the difference between a mystery and a diagnosis. It deliberately does
_not_ auto-retry on a backoff — an example that silently reconnects hides the
very state machine it exists to show, and a reader who wants that behaviour
should add it on purpose.

A session is as expensive to hold as it is to get: a connected session
occupies the GPU and bills every second, including idle time between takes.
This example leaves disconnect manual for the same reason it does not
auto-retry, but a surface meant to be left open should end idle sessions
itself. The reactor.inc sandbox disconnects after 30 idle seconds, keeps the
composed conditions client-side so the next generate reconnects and replays
them, and raises a "planned disconnect" flag before calling `disconnect()` so
its connection-lost handling stays quiet for a disconnect it chose.

Separately from capacity, two `connect()` rejections are transient races worth
one silent retry if you wrap the call: a message naming `pollSessionReady`
(the readiness poll raced the session) and one containing `Already connected`
(a previous attempt's teardown had not finished). The production webapp
handles both the same way: `disconnect()`, wait 250 ms, `connect()` once, and
let a second failure surface normally.

## Auth — a memoizing `jwtToken` resolver + a scoped mint route

Two pieces work together: a Next.js GET route that mints a session-scoped JWT
server-side, and a `jwtToken` resolver on `<Ltx2Provider>` that the SDK calls on
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

## Presets are macros, not a wrapper

`directPreset` sends the real command sequence, in order, with nothing hidden:

```
set_avatar_image → set_script → set_prompt → set_wpm → set_seed
                 → set_duration_seconds → start
```

That is the point of them. A reader should be able to click a preset, then
reproduce the same take by hand from the take panel. Keep new presets to the
same rule: if it isn't expressible in the panel, it doesn't belong in the rail.

Seeds are pinned so takes reproduce. `set_duration_seconds: 0` is sent
explicitly to clear a duration a previous take may have pinned — presets derive
their length from the script.

The three presets shipped here are the public demo's cast, trimmed to a
subset: each script, voice prompt, pace and seed is the production record,
validated against the live model, and each prompt goes out inside the camera
lock described below. Treat them as reference conditions; a new preset should
earn its place the same way, by running well live, not by reading well.

Portraits are not committed (see `public/presets/README.md`). When one is
missing the preset applies everything except the image and says so, rather than
failing; the row falls back to the preset's monogram. That fallback probes the
image client-side rather than using `<img onError>`, because an SSR'd img can
fail before hydration and the error event never reaches React.

## Locking the camera in the scene prompt

Left loose, the model drifts the camera over a take: a slow push in, a
reframe. The public demo pins every scene prompt with the same wrapper, and it
held in production:

> Locked-off tripod shot, fixed framing from the first frame to the last.
> _[the scene description]_ The camera never moves, never pans, never zooms,
> and never pushes in; the framing at the end of the video is identical to the
> framing at the start.

Stating it twice, before and after the scene, is deliberate. Wrap generated or
added prompts the same way unless camera motion is the point.

## Recording

Recording is enabled in the model's manifest (h264 640×352 + aac, 4 s chunks,
300 s cap) and `requestClip()` is accepted — but on the current deployment the
clip never materializes: the playlist is polled indefinitely and the request
neither fails nor times out.

The save button ships anyway. `SnapClip.tsx` is copied unchanged from the
sibling examples and rendered in the sidebar as the **Capture** panel; it is
base-SDK surface that imports only `@reactor-team/js-sdk` and needs no
model-specific code, so the day the deployment produces clips it works with no
client change. Don't read the dead button as a missing feature and don't
rebuild it — the defect is behind the API, not in this file.

The one adaptation still worth making sits at the call site. The panel ships
with a fixed window (`durationSeconds={30}`), but this model produces discrete
takes rather than a continuous stream, so sizing the request to the take —
`requestClip(ui.secondsSent + margin)` — beats both the fixed window and
`requestRecording()`, which would return every take plus the idle gaps between
them.

## Common mistakes when extending

1. **Sending `start` before the avatar image is confirmed.** The take renders
   with the previous face. Go through `setAvatarImage()` and check its result.
2. **Inferring session state from clicks.** Gate on the reduced
   `Ltx2UiState`; only `state_update` mutates it.
3. **Making the take panel inert during a run.** Every control in it is valid
   mid-run. Only the preset rail is inert.
4. **Scattering `ui.generating` checks through components** instead of asking
   `validCommands()`. The model already answers this; a local copy will drift.
5. **Measuring TTFF off the next frame after `start`.** The idle stream keeps
   compositing; arm the measurement on `generation_started`.
6. **Playing `main_video` and `main_audio` from separate elements.** They share
   a sample clock and will drift. One `MediaStream`, one element.
7. **Uploading an uncropped portrait.** The model's fit beheads tall images.
8. **Swallowing `command_error`.** It is what you get for sending a command
   absent from `valid_commands`; surface every one.
9. **Forgetting the disconnect reset.** New session state must be cleared in the
   `status === "disconnected"` effect, or the next session starts haunted.
10. **Assuming a condition cannot change mid-run.** All six setters stay valid
    while generating; the change lands on the next take. What you must not
    assume is that it affects the take already playing.
11. **Hand-rolling command strings** instead of the typed methods off
    `useLtx2()`.
12. **Treating a `command_error` on `set_wpm` as a validation bug.** The model
    refuses a pace outside the deployment's `wpm_min`–`wpm_max`, correctly.
13. **Leaving a session connected while nobody is generating.** It holds the
    GPU and bills every second, takes or no takes. A long-lived surface needs
    an idle disconnect.

## Checklist for new components

- [ ] Gated on `status === "ready"` plus `validCommands()` — not ad-hoc flags
- [ ] Reads model state from `Ltx2UiState`, never from local click history
- [ ] New message handling lives in `Workspace` via the typed per-message hooks
- [ ] New session state resets in the disconnect effect
- [ ] Command validity comes from `valid_commands`, never re-derived locally
- [ ] Fields that can change mid-run show their `queued` state via `isQueued()`
- [ ] `command_error` still surfaces
- [ ] Any new face-supplying path goes through `CropModal` and `setAvatarImage`
- [ ] Colors via the theme utilities (`bg-brand`, `border-edge`, …), not raw hex
- [ ] Typed surface imported from `@reactor-models/ltx2`
