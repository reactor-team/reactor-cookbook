# LTX

A Next.js + TypeScript reference frontend for **LTX** — real-time
streaming talking-head generation on Reactor.

Give it a still image and a script. It generates the voice and the lip-synced
video _together_ and streams both back over WebRTC, window by window, while you
watch. There is no inbound media track: the face is a file upload, the speech is
generated from text, and the whole take is shaped by six settings you can see
and change.

```
┌──────────────────┬────────────────────────────────────────────┐
│  Connect         │                                            │
│  ──────────────  │                                            │
│  Transport       │                                            │
│   next take →    │           live A/V output                  │
│     Start        │                                            │
│   live →         │                                            │
│     Pause Resume │                                            │
│     Stop         │                                            │
│   session →      │                                            │
│     Reset        │                                            │
│  ──────────────  │                                            │
│  Presets ×3      │                                            │
│  ──────────────  │                                            │
│  Direct the take │                                            │
│   avatar image   │                                            │
│   script         │                                            │
│   scene prompt   │                                            │
│   pace · seed    │                                            │
│   duration       │                                            │
│  ──────────────  ├────────────────────────────────────────────┤
│  Capture         │  status line · time to first frame         │
└──────────────────┴────────────────────────────────────────────┘
```

Every control lives in the sidebar and the stage carries nothing but the take,
which is the same split every other example in this repo uses.

## Quick start

> **Heads up — this model is not listed publicly yet.** It runs on production
> (`https://api.reactor.inc`), so a normal `rk_` key from the dashboard is the
> right kind of key, but the account behind it has to be granted access to
> `reactor/ltx2` first. Ask the Reactor team; a `connect` that 404s with
> `model not found` is what missing access looks like.

> **Start a standalone project:** `npx create-reactor-app my-app --model=ltx2` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

```bash
cp .env.example .env.local   # add your REACTOR_API_KEY
pnpm install
pnpm dev                     # http://localhost:3000
```

Then press **Connect**. The app does not connect on load, deliberately: a
session holds a whole B200, so opening the page should not claim one. The click
doubles as the user gesture browsers require before video may play with sound.
Disconnect when you are done; a connected session bills every second it holds
the GPU, including idle time between takes.

The app defaults to `https://api.reactor.inc`; override with
`NEXT_PUBLIC_REACTOR_API_URL`. The key stays on the server — the browser only
ever sees a short-lived JWT minted by `app/api/reactor/token/route.ts` (see
[docs.reactor.inc/authentication](https://docs.reactor.inc/authentication)).

**Preset portraits are not committed.** Faces are a licensing and consent
question, so the preset rows show a monogram until you add your own; see
[`public/presets/README.md`](public/presets/README.md). Everything else works
without them — upload a face in the take panel and press Start.

## The one thing to understand: the take is frozen, the session is not

Most Reactor models let you steer generation while it runs: swap the prompt
mid-stream and the change lands at the next chunk. This model draws the line in
a different place, and it is narrower than it first looks. **The take you are
watching is fixed at `start`, and nothing changes it.** But the session's
conditions stay editable throughout. Send `set_script` mid-run and the model
accepts it, acks it, and applies it to the _next_ take.

Two fields on the `state_update` snapshot carry this, and both are
authoritative:

- **`valid_commands`** lists what the session would accept right now; anything
  absent comes back as `command_error`. `validCommands()` (`app/lib/machine.ts`)
  passes it through, adding only the one thing the snapshot cannot know:
  whether there is a session at all. The app derives no validity rules of its
  own.
- **`queued_changes`** lists the condition fields changed during the run in
  flight. Those values are _already_ in the snapshot. The field only says "you
  will see this on the next take, not the one playing."

So the take panel stays live during a run, each changed field wears a `queued`
chip, and there is no Apply button, because nothing is waiting client-side to be
flushed. The preset rail does dim, because a preset ends in `start` and there is
genuinely nothing valid to click.

The mistake worth avoiding: do not rebuild either rule client-side. An earlier
version of this app mirrored the state machine inside `validCommands()` and held
mid-run edits in React state. Both had to be deleted once the model began
reporting the truth itself, and anything you derive locally will drift the same
way.

## What you can do with it

- **Run a preset** — each row fires the real command sequence
  (`set_avatar_image → set_script → set_prompt → set_wpm → set_seed →
set_duration_seconds → start`) in order, with a pinned seed so the take
  reproduces. Presets are macros over the take panel's form; there is
  nothing in them you cannot send by hand. The three characters are the
  public demo's cast, trimmed to a subset: scripts, voice prompts, paces and
  seeds are the production records, validated against the live model.
- **Bring a face** — upload any photo. The model fits the image to its 640×352
  canvas, which decapitates a tall portrait, so the app crops first: drag to
  frame, and only those pixels go up (`CropModal.tsx`, with the browser's
  `FaceDetector` picking the default framing where available).
- **Write a script** — the run's length derives from `words / wpm`, or pin it
  explicitly with duration. The model reports the length it will actually use as
  `effective_seconds`.
- **Change the pace** — 80 to 220 wpm by default, though the range is
  deployment-configured and the slider reads it off `wpm_min` / `wpm_max` on the
  snapshot rather than hard-coding it. Each character carries the pace it was
  tuned at (110 · 120 · 160); drag the slider and the delivery moves with it.
- **Pause and resume** — the stream freezes on the last frame while generation
  keeps running ahead into a buffer, so resume continues instantly with no
  warm-up. Try it mid-sentence on the Teddy Bear.
- **Reroll** — same image, script, prompt, wpm and duration with the same seed
  give you the same take. `set_seed` is the "another take, same setup" knob.
- **Remix** — `stop` ends the take but keeps every condition on the session,
  so the fastest loop is: type a new script, press Start. Nothing else needs
  re-sending.

## Architecture at a glance

The model is the **source of truth**: it broadcasts a full `state_update`
snapshot on connect and after every observable change, and the whole UI renders
from a reduction of that snapshot (`app/lib/state.ts` → `Ltx2UiState`).
The app never infers session state from its own button clicks.

Commands this app sends: `set_avatar_image`, `set_script`, `set_prompt`,
`set_wpm`, `set_duration_seconds`, `set_seed`, `start`, `pause`, `resume`,
`stop`, `reset`, which is the model's full command set. Which of them the
session will accept at any moment comes from `state_update.valid_commands`,
never from the app's own reckoning, so a deployment that gates or adds a command
needs no change here.

Tracks out: `main_video` and `main_audio`, in lockstep on one sample clock.
Tracks in: none.

The typed client comes from `@reactor-models/ltx2`, generated from the
model's schema. `<Ltx2Provider jwtToken={fetchToken}>` bakes in the model
name and tracks; `useLtx2()` exposes status plus typed commands
(`setScript`, `setWpm`, `start`, …); and per-message hooks
(`useLtx2StateUpdate`, `useLtx2CommandError`, …) replace a
hand-rolled message switch.

> **Time to first frame.** The model is windowed, not frame-causal: nothing
> streams until the leading window has denoised and decoded. The figure in the
> status line under the stage measures the real thing — `start` going on the
> wire to the first frame composited _after_ `generation_started`. Measuring
> the next frame after `start` instead would read a few milliseconds, because
> the WebRTC track keeps compositing between takes; see `markFirstFrame` in
> `app/Ltx2App.tsx`.
>
> Deliberately no target number here. This is a pre-release model on a dev
> pod and the number moves; treat what you see as a reading, not a spec.

## Code tour

| Path                             | What it is                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `app/Ltx2App.tsx`                | Provider shell, all message handling, the action layer, TTFF anchoring                    |
| `app/lib/machine.ts`             | The `valid_commands` gate, the queued-field lookup, the status line. **Read this first.** |
| `app/lib/state.ts`               | `state_update` → `Ltx2UiState` reducer                                                    |
| `app/lib/types.ts`               | The reduced UI state, take fields, and the model's limits                                 |
| `app/lib/presets.ts`             | The public demo's cast, cut to three: script, voice prompt, pace, pinned seed             |
| `app/components/Stage.tsx`       | The `<video>` carrying both tracks, TTFF measurement, the stall watch, status line        |
| `app/components/Transport.tsx`   | Start / Pause / Resume / Stop / Reset, gated by `validCommands()`                         |
| `app/components/SnapClip.tsx`    | Clip capture + download. Copied unchanged from the sibling examples                       |
| `app/components/ui/`             | Shared primitives (`Panel`, `Button`, `Icon`) — the same module every example uses        |
| `app/components/TakePanel.tsx`   | The six conditions and their `queued` chips. Holds no pending-edit state                  |
| `app/components/PresetRail.tsx`  | The persona rows; each click is a real command sequence                                   |
| `app/components/CropModal.tsx`   | Frame-before-upload so the model's fit doesn't behead the subject                         |
| `app/components/StatusBadge.tsx` | Connection state and the Connect / Disconnect entry point                                 |
| `app/api/reactor/token/route.ts` | Mints the short-lived JWT server-side                                                     |

## Known broken: saving a take

The **Capture** panel is wired and correct, and it does nothing useful yet.

`requestClip()` is accepted and returns a `Clip`, but the clip never
materializes on the current dev deployment: the playlist never becomes
playable, so the preview sits on "waiting for clip" indefinitely. The request
neither fails nor times out, which is the unhelpful part — there is nothing to
surface as an error.

The panel ships anyway, because recording is base-SDK surface
(`SnapClip.tsx` is copied unchanged from the sibling examples and imports only
`@reactor-team/js-sdk`), so when the deployment starts producing clips this
works with no client change. Verified still broken on 2026-08-09.

## Not in this example

- **`say()` / live mode.** Not in this release of the model.

## Going further

`skill/SKILL.md` documents the patterns this app uses: reading `valid_commands`
and `queued_changes` instead of deriving them, the avatar-image acknowledgement
race, TTFF measurement, the take-chaining loop behind a continuous
performance, and the capacity story. Point your coding agent at it before you
extend this app.

## Tech stack

Next.js 15 · React 19 · TypeScript · Tailwind v4 · `@reactor-team/js-sdk` ·
`@reactor-team/ui`
