# XMAX X2

A Next.js + TypeScript reference frontend for [**XMAX X2**](https://docs.reactor.inc/model-api-reference/xmax/overview) - real-time streaming video **editing** on Reactor.

Where generation models produce video from a prompt alone, X2 edits the video you bring: describe a change in plain text and the model makes that change while everything you don't mention carries through from the source untouched. Point your webcam at it and watch yourself transformed live, pick a clip and stream it into the model frame for frame, or feed it a still image and drag on the output to animate it. Re-prompt mid-stream and the new edit lands at the next chunk boundary.

```
┌──────────────────────┬───────────────────────────────────────┐
│  Status / errors     │                                       │
│  Source              │           edited output               │
│   • webcam self-view │   webcam mode shows just this;        │
│   • or video clip    │   video / image mode puts your        │
│   • or still image   │   source on the left and the edit     │
│   • keep backlog     │   on the right — drag on the output   │
│     · reset          │   to steer the subject                │
│  Prompt + presets    ├───────────────────────────────────────┤
│  Pointer readout     │   status row (generating · resolution │
│  Reference image     │            · ref ✓ · prompt)          │
│  Snap clip           │                                       │
└──────────────────────┴───────────────────────────────────────┘
```

## Quick start

> **Start a standalone project:** `npx create-reactor-app my-app --model=x2` scaffolds this example into a fresh app — no clone needed. The steps below are for running it in-place from a monorepo checkout.

```bash
cp .env.example .env.local  # then add your REACTOR_API_KEY
pnpm install
pnpm dev                    # http://localhost:3000
```

Get a **production** API key (`rk_...`) from the [Reactor dashboard](https://reactor.inc) - the app targets `https://api.reactor.inc`. The key stays on the server (`REACTOR_API_KEY` is the only required env var); the browser only ever sees a short-lived JWT minted by `app/api/reactor/token/route.ts`, scoped to `xmax/x2` sessions via `authorization_details`.

## What you can do with it

- **Webcam** - your webcam is published to the model's `source` input track and transformed in real time. Edited frames come back on `main_video`; the model picks its output resolution per session from your source stream.
- **Video** - pick a preset clip or a local file. Instead of uploading it, the app **plays it and streams its frames into the same `source` track**, so the model edits it on its live path — the source pane and the edited pane share one feed and can't drift apart.
- **Image** - pick a still image and the app repeats it as a constant feed. Set a prompt and drag on the output to animate the scene.
- **Steer the prompt** - setting a prompt is what arms generation; re-prompt mid-stream at any time and the new edit lands at the next chunk boundary, with no re-render and no break in the stream. Prompts are editing instructions, not scene descriptions - the [prompt guide](https://docs.reactor.inc/model-api-reference/xmax/prompt-guide) covers how to write edits that land where you aim them.
- **Reference image** - upload an image the model conditions on (a face, an outfit, a style target). The upload rides `uploadFile()` and lands as a `set_reference_image` command; swapping it mid-stream restarts generation automatically. Uploads are floored to mod-4 dimensions first: the runtime streams the reference as driving video, and its GStreamer pipeline scrambles the color channels on any frame whose width or height isn't a multiple of 4.
- **Drag to steer** - press and drag on the edited output to drive the model's pointer (`set_pointer`, normalized 0..1 output-frame coordinates). Releasing deactivates it. Works in every source mode; the sidebar's Pointer panel shows the raw `pointer_x` / `pointer_y` / `pointer_active` values as the model echoes them back in `state_update`, so you can watch exactly what the API receives.
- **Keep backlog** - a checkbox on the source panel toggles `set_keep_backlog`: keep every source frame queued (edit all of them, latency grows) or drop stale frames to stay real-time.
- **Snap a clip** - capture the last N seconds of the stream (model-agnostic recording).

> **Aspect ratios — keep the source and reference on the same one.** The model
> derives its output resolution from the source stream, so a landscape source
> gives a landscape edit. Conditioning that landscape output on a **portrait**
> reference image (or vice-versa) distorts the result badly — a stretched or
> squashed character. Every preset here is landscape 16:9 (clips, webcam, and
> the reference images) so they line up. If you bring your own reference, match
> the source's orientation. The app does **not** normalize the reference for you
> yet; it uploads what you give it.

## Architecture at a glance

The model is the **source of truth**: it broadcasts a full `state_update` snapshot on connect and after every observable change, and the UI gates entirely off the reduced `X2UiState` (`app/lib/types.ts`). All three source modes stream into the `source` track, so generation is always the model's live path. Commands out: `set_prompt`, `set_reference_image`, `set_pointer`, `set_keep_backlog`, `reset`. Tracks: input `source`, output `main_video`. The full wire surface - every command, message, and the `state_update` payload - is documented in the [schema reference](https://docs.reactor.inc/model-api-reference/xmax/schema).

> **Streaming a clip, not uploading it.** A selected video is played in a `<video>` element and captured with `captureStream()`; that track is published as `source`. What you see in the source pane is literally what the model receives. A still image works the same way — it's painted to a canvas and captured as a constant 24 fps stream.

The typed client comes from the published [`@reactor-models/x2`](https://www.npmjs.com/package/@reactor-models/x2) package. `<X2Provider jwtToken={fetchToken}>` bakes in the model name and tracks; `useX2()` exposes status plus typed commands (`setPrompt`, `setReferenceImage`, `setPointer`, `setKeepBacklog`, `reset`, `uploadFile`); per-message hooks (`useX2StateUpdate`, `useX2GenerationStopped`, …) replace a hand-rolled message switch; and `<X2MainVideoView />` renders the live output.

## Code tour

| Path                                   | What it is                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `app/X2App.tsx`                        | `X2Provider` shell + layout + one-shot handlers for the discrete model events    |
| `app/lib/state.ts` / `types.ts`        | `X2UiState` and the reducer projecting `state_update` snapshots into it          |
| `app/components/SourcePanel.tsx`       | Source toggle (webcam / video / image), keep-backlog, reset                      |
| `app/components/WebcamSource.tsx`      | Webcam capture + self-view (in the panel); produces the `source` track           |
| `app/components/VideoSource.tsx`       | Plays a chosen clip and streams it into `source` via `captureStream()` (stage)   |
| `app/components/ImageSource.tsx`       | Repeats a still image as a constant canvas-captured feed (stage)                 |
| `app/components/VideoPicker.tsx`       | Preset + local-file picker for the video source                                  |
| `app/components/ImagePicker.tsx`       | Preset + local-file picker for the image source                                  |
| `app/components/useSourcePublisher.ts` | Single owner of the `source` slot; reconciles the wire to the latest track       |
| `app/components/Prompt.tsx`            | Prompt textarea + Apply, preset chips, active prompt                             |
| `app/components/ReferenceImage.tsx`    | Reference upload (`uploadFile` → `set_reference_image`), preview, accepted dims  |
| `app/components/PointerOverlay.tsx`    | Drag-to-steer overlay mapping pointer to output-frame coords (`set_pointer`)     |
| `app/components/PointerPanel.tsx`      | Sidebar readout of the model-echoed `pointer_x` / `pointer_y` / `pointer_active` |
| `app/components/Stage.tsx`             | Edited output — single in webcam mode; split with the source in video / image    |
| `app/api/reactor/token/route.ts`       | Mints the short-lived, session-scoped JWT server-side                            |
| `app/components/SnapClip.tsx`          | Model-agnostic clip recording on `@reactor-team/js-sdk` (drop-in)                |

## Going further

`skill/SKILL.md` documents the patterns this app uses - the typed client surface, the connection and state model, how all three sources stream into the `source` track, the single-owner publish reconciler, the reference-image upload path, and the pointer protocol. Point your coding agent at it when you build on top.

The published docs cover the model itself: the [overview](https://docs.reactor.inc/model-api-reference/xmax/overview) for the conceptual model and quick start, the [schema](https://docs.reactor.inc/model-api-reference/xmax/schema) for every command and message, the [prompt guide](https://docs.reactor.inc/model-api-reference/xmax/prompt-guide) for writing edit instructions, and a [tutorial](https://docs.reactor.inc/model-api-reference/xmax/tutorial) built around this app.

## Tech stack

Next.js 15 · React 19 · TypeScript · Tailwind v4 · [`@reactor-models/x2`](https://www.npmjs.com/package/@reactor-models/x2) (typed SDK) · `@reactor-team/js-sdk` (recording primitives) · `@reactor-team/ui` (design tokens)
