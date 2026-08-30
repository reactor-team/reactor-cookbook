# FastH3

A queue of prompt-driven video clips with synchronized audio. Clients enqueue
generation requests — a prompt plus their own metadata, each answered with a
UUID — the model builds them in order into a bounded buffer, and playback is a
separate, explicit step: `play` streams one built clip at 768p over WebRTC,
and when it ends the stream holds on black until the next `play`. Nothing
plays on its own.

Reach for this when something else decides what plays and when: a frontend
that lets people submit prompts and curates the order, a playlist that is
assembled faster than it is watched, a controller that wants clips ready
before they are needed. The model is the queue handler and the renderer; the
scheduling brain sits on the client side of the API. If you want one finished
clip returned as a file, this is the wrong shape — clips are streamed, not
returned.

[FastH3 Preview v1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is MiniMax-H3 (35B) distilled by FastVideo with data-free DMD2 down to **four
transformer forwards**, with 90% sparse video attention on Blackwell. It
generates video and stereo audio jointly from text. Text-to-video-and-audio is
the only task this checkpoint was distilled for; first/last-frame and reference
conditioning are not.

## Prerequisites

- **Four NVIDIA B200s** — FastVideo's tested default for this checkpoint. A
  15 s clip builds in about 15.5 s on four and 12.9 s on eight; playback is
  explicit and clips are pre-built, so the GPU count only moves the
  enqueue-to-ready wait. The count must divide H3's 56 attention heads
  (1, 2, 4, 7, 8 …), and each rank holds its own ~63 GB text encoder plus a
  66/N GB transformer shard, so fewer than four wants offloading.
- **CUDA 13.** The VSA-H3 sparse kernel and the FA4 CuTe kernels are both cu130
  builds, which is why this model's `build.cuda_version` differs from every
  other model here.
- **The weights bundle**, roughly 148 GB, placed under `runtime.weights_path`.
  Nothing is downloaded at load — see below.

## Weights

One Hugging Face snapshot carries every component:

```
~/.cache/reactor_registry/fasth3/
└── FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/   # ~148 GB
    ├── modular_model_index.json
    ├── transformer/        # ~70 GB, the 35B DiT
    ├── text_encoder/       # ~69 GB, Qwen3-VL
    ├── vae/  audio_vae/
    ├── scheduler/  audio_scheduler/
    └── tokenizer/  processor/
```

FastVideo resolves each component as a subdirectory of `model_path` and ignores
the repo ids inside `modular_model_index.json`, so the bundle loads fully
offline — which is what `HF_HUB_OFFLINE=1` in the manifest relies on. `load()`
checks every component directory up front, so an incomplete bundle stops
startup rather than surfacing as a loader traceback on the first clip.

## Run it

```sh
# Render the client-facing contract — no weights, no GPU.
python -m reactor_runtime.schema --path . --out /tmp/schema.json

# The CPU-only structural tests.
PYTHONPATH=. python -m pytest tests/ -q

# Build the image and serve (needs the weights and four GPUs).
reactor build
reactor run

# Drive the served model end to end with the reference client (saves .mp4s).
python client/client.py
```

`load()` warms one throwaway clip per configured canvas before the pod reports
ready, so the first real build runs at warm speed. Every distinct frame count
and canvas is a separate one-time compile cost, which is why
`inference.warmup_aspects` in [`fasth3.yaml`](./fasth3.yaml) is a deliberately
short list and why the first build at a non-default `set_clip_seconds` pays a
one-off stall.

Before blaming the adapter for slow builds, baseline the recipe itself with
FastVideo's own `examples/inference/basic/basic_fasth3.py` at the same settings.
Its median is the number this model should match; a gap is the adapter's fault,
not the model's.

## The mental model

```
enqueue ──► [ queue, oldest first, up to inference.queue_size ] ──► play ──► tracks
              build build build ... (in order, one at a time)          │
              ready clips wait in host memory                          ▼
                                                            clip ends or stop:
                                                            flush → black, wait
```

- **`enqueue` is the only way in.** Each request snapshots the session's
  conditions (`set_clip_seconds`, `set_seed`, `set_canvas`) as they stand, gets
  a UUID, and joins the back of the queue. The queue is bounded
  (`inference.queue_size`, default 10); a full queue refuses further enqueues.
- **Builds run through the queue front to back**, one at a time, whenever an
  audience is connected — including while another clip is playing. A finished
  build turns the entry `ready: true`, announced on `queue_update`.
- **`play` is the only way out** — unless autoplay is on. Bare `play` takes the
  oldest ready clip; a `clip_id` takes that specific one. Playing consumes the
  entry. When the clip ends — or `stop` cuts it — the output flushes to black
  and the session waits for the next `play`. With `set_autoplay` on, the
  oldest ready clip starts on its own whenever nothing is playing, so a
  steadily fed queue plays through hands-free; `stop` then acts as a skip.
- **Everything a clip is travels with every mention of it.** `clip_queued`,
  `queue_update`, `clip_started`, `clip_finished`, `clip_stopped` and
  `clip_failed` all embed the full `ClipInfo` structure, so a client never has
  to join a UUID against an earlier message.

## The `ClipInfo` structure

| Field | Type | Meaning |
|---|---|---|
| `clip_id` | string | UUID assigned at `enqueue`; every later reference uses it. |
| `prompt` | string | What the clip shows, exactly as enqueued. |
| `metadata` | string | Opaque client string, echoed back untouched — see below. |
| `frames` | int | Clip length in frames, fixed at enqueue time. |
| `seconds` | float | The same length in seconds (`frames / 24`). |
| `seed` | int | Seed this clip generates from. |
| `ready` | bool | Whether the clip is built and can be played. |

**Metadata is for the frontend, not the model.** The model stores it and echoes
it back on every message that references the clip; it never parses it. Use it to
carry whatever your application needs to track — which request produced the
clip, who asked for it, which group of enqueues it belongs to, text to show
while it plays. Up to 2000 characters; JSON fits if you want structure.

## Tracks

| Track | Direction | Kind | Rate | Payload |
|---|---|---|---|---|
| `main_video` | out | video | 24 fps, fixed | RGB frames at the session's canvas, e.g. 1344×768 |
| `main_audio` | out | audio | 48 kHz | Mono, synchronized frame-for-frame with `main_video` |

There are no inbound tracks: the model reads no camera and no microphone. The
video track keeps one size — `set_canvas` chooses it and is only accepted while
the queue is empty and nothing is playing, since queued clips are built at the
size in force.

## Commands

| Command | Parameters | Effect | Rejected when |
|---|---|---|---|
| `enqueue` | `prompt` (≤ 800 chars), `metadata` (≤ 2000 chars), `seed` (optional, ≥ 0), `seconds` (optional, 5.167–14.375) | Queues one generation; replies `clip_queued` with the full `ClipInfo`. Without a seed the session's advancing default is used; without `seconds` the session's default length. | queue full, empty prompt |
| `play` | `clip_id` (optional UUID) | Streams the oldest ready clip, or the named one. Emits `clip_started` as frames begin. | already playing, unknown id, clip not ready |
| `pop` | `clip_id` (UUID) | Removes that clip from the queue, freeing its slot; a build in flight for it is discarded. Replies `clip_popped`. | unknown or missing id |
| `stop` | — | Cuts the playing clip to black; the queue is untouched. With autoplay on, acts as a skip. Emits `clip_stopped`. | nothing playing |
| `get_queue` | — | Replies with the full queue — the same payload as `queue_update`. | — |
| `set_autoplay` | `enabled` (bool) | On, the oldest ready clip starts on its own whenever nothing is playing. Off (default), the stream holds until `play`. Replies `autoplay_accepted`. | — |
| `set_clip_seconds` | `seconds` (5.167–14.375) | Default length for enqueues that carry no `seconds`, snapped to what the model can produce; the effective value returns in `clip_length_accepted`. | — |
| `set_seed` | `seed` (≥ 0) | Default seed for enqueues that carry none; each such enqueue advances it by one. Replies `seed_accepted`. | — |
| `set_canvas` | `aspect` (`16:9`, `1:1`, `9:16`, `4:3`) | Video size for the session. Replies `canvas_accepted`. | clips queued or playing, unsupported aspect |
| `reset` | — | Drops the whole queue, cuts any playing clip, restores every default. Replies `session_reset`. | — |
| `get_state` | — | Replies with the full `state_update` snapshot. | — |

A rejected command has no effect and is answered by a broadcast
`command_error` naming the command and the reason.

## Messages

| Message | Reaches | When |
|---|---|---|
| `state_update` | everyone | On connect, and after every change. A complete snapshot minus the queue's contents — render from this plus `queue_update` alone. |
| `queue_update` | everyone | On connect, and whenever the queue changes: an enqueue, a clip turning ready, a clip leaving to play, a reset. Carries every `ClipInfo`, oldest first. |
| `clip_queued` | the caller | Reply to `enqueue`. The full `ClipInfo`, UUID included. |
| `clip_started` | everyone | A clip's first frames reach the tracks. |
| `clip_finished` | everyone | A clip was fully sent; the stream is now black until the next `play`. |
| `clip_stopped` | everyone | `stop` (or `reset`) cut the clip; the rest of it is discarded. |
| `clip_failed` | everyone | A build failed; the clip left the queue and the queue moves on. |
| `clip_popped` | the caller | Reply to `pop`. The clip left the queue and its slot is free. |
| `clip_length_accepted` | the caller | Reply to `set_clip_seconds`. Carries the snapped value. |
| `seed_accepted` | the caller | Reply to `set_seed`. |
| `autoplay_accepted` | the caller | Reply to `set_autoplay`. |
| `canvas_accepted` | the caller | Reply to `set_canvas`. Carries the exact pixel size. |
| `session_reset` | the caller | Reply to `reset`. Says how many clips were dropped. |

## Session lifecycle

```
  session starts (no clients yet)
    |
    v
  client connects       -> state_update + queue_update (to this client)
    |
  ┌───────────────────────────────────────────────────────────────┐
  │ IDLE (black screen)                                           │
  │ Valid: enqueue, set_clip_seconds, set_seed, reset, get_queue, │
  │        get_state; set_canvas while the queue is empty;        │
  │        play once a clip is ready                              │
  │ Builds run in the background whenever the queue has work      │
  └───────────────────────────┬───────────────────────────────────┘
                              v  play
  ┌───────────────────────────────────────────────────────────────┐
  │ PLAYING one clip                                              │
  │ Valid: enqueue, set_clip_seconds, set_seed, stop, reset,      │
  │        get_queue, get_state                                   │
  │ Messages: clip_started, then clip_finished or clip_stopped    │
  │ Builds keep running behind the playout                        │
  └───────────────────────────┬───────────────────────────────────┘
                              v  clip ends / stop / reset
                    (flush to black, back to IDLE)
```

**Single session, shared state.** Several clients may attach to one session and
they all see the same queue and the same stream: an `enqueue` or a `stop` from
any client affects everyone, and every client receives every `state_update` and
`queue_update`. Generation is gated on having an audience — with nobody
connected no new build starts, though a build already running finishes into the
queue.

`state_update.valid_commands` names exactly what the session would accept at
that moment, so a frontend enables and greys out controls from the snapshot
instead of re-deriving these rules.

## What to expect from the timing

- **Enqueue-to-ready** is one build, plus the wait behind earlier queued
  builds; `queue_update` reports the clip turning `ready`. On the measured
  sm100a profile a 14.375 s clip builds in about 15 s on four B200s (13 s on
  eight); on the portable profile this deployment currently runs (triton VSA,
  offloaded text encoder — see `fasth3.yaml`) the measured number is 31–35 s.
- **Play-to-first-frame** is near-instant for a ready clip — the frames are
  already in host memory; the only latency is the transport.
- **`stop`** cuts to black within a fraction of a second: the emitter checks the
  flag every slice (about an eighth of a second) and whatever the transport
  still holds is flushed. A build in flight for another clip is unaffected —
  and cannot be cancelled, so `reset` may keep the GPUs busy for a few more
  seconds finishing a clip it will then discard.

Playout is a strict 24 fps metronome and the audio is sample-clocked against
the same rate, so the two tracks stay locked for the length of any clip.

## Clip boundaries are hard cuts

Every clip is generated independently. There is no continuity of subject,
framing, or voice from one clip to the next, even with identical prompts — and
the stream holds on black between plays. This checkpoint has no continuation
path, and inventing one is not an adapter's decision. It is why
`runtime.recording` is left disabled: a recording would carry those cuts too.

The geometry it accepts is narrow, and [`fasth3_clip_plan.py`](./fasth3_clip_plan.py)
encodes it: 24 fps, a frame count of the form `17n + 5`, a duration between 5
and 15 seconds, a short edge of 768, at most 768×1344 pixels, and both sides a
multiple of 32. The duration cap has a sharp edge — 15.0 s is 360 frames, which
aligns *up* to 362 (15.083 s) and is then rejected, so **the longest clip this
model can make is 345 frames, 14.375 s**.

## Determinism

Each enqueued clip carries its own seed — passed explicitly on `enqueue`, or
taken from the session's advancing default (`set_seed` fixes it; each seedless
enqueue advances it by one, and explicit seeds leave it untouched).
Re-enqueuing the same prompts with the same seeds, clip length and canvas
reproduces the same clips. Reproduction is approximate rather than bit-exact:
the deployment runs fused and compiled kernels that can reorder floating-point
operations.

## Notes on the code

**`ReactorModel`, not `ReactorPipeline`.** The generator base is shaped for
frame-per-step models, where a `yield` is the natural unit of work. Here the
unit is a whole clip produced by one blocking call, so the generator would buy
nothing. Under `ReactorModel`, command handlers and lifecycle hooks run on
background coroutines *concurrent* with `run()`, so:

- session state is plain attributes reset in `_reset_session_state`, called from
  `@session_started` (not `@connected`: a client rejoining mid-session keeps the
  queue it joined);
- one persistent worker thread serialises every build and gives teardown a
  single handle to wait on;
- refusals and accepts answer immediately, even mid-build and mid-play.

**Built clips live in host memory.** A ready clip is about 1 GB of uint8 pixels
at the 16:9 canvas and the longest length, and the queue can hold
`inference.queue_size` of them — size `resources.memory` in the manifest
together with that knob.

**Refusals are broadcast, not raised.** A handler returns only the message its
annotation names and reports failure by broadcasting `command_error`. A raised
runtime `CommandError` has its failure frame withheld from v0 clients, so the
broadcast is what reaches every SDK generation.

**Nothing is vendored.** FastVideo is a published package, so the whole upstream
tree arrives through `requirements.txt` and an upgrade is a one-line bump.

## Layout

| Path | What it is |
|---|---|
| `fasth3.py` | The `ReactorModel`: commands, lifecycle, the queue/playout loop |
| `fasth3_types.py` | Everything a client sees — tracks, `ClipInfo`, messages |
| `fasth3_queue.py` | The bounded clip queue and its entries |
| `fasth3_backend.py` | The FastVideo engine, its worker thread, warm-up, audio conversion |
| `fasth3_assets.py` | Config parsing and weights-bundle validation |
| `fasth3_clip_plan.py` | Clip geometry: valid lengths, frame counts, canvases |
| `fasth3_session_rules.py` | Which commands each session state accepts |
| `fasth3.yaml` | `inference:` the recipe and queue size, `runtime:` weight layout and engine shape |
| `reactor.yaml` | The manifest: identity, version, resources, runtime, image build |
| `tests/` | Structural tests that need no GPU |
| `client/` | Reference SDK client that drives the whole queue contract and saves what it receives |

## Open questions for bring-up

- **The sm100a VSA kernel does not launch.** fastvideo-kernel 0.3.5's CUDA
  block-sparse kernel fails with `block_sparse_sm100a launch failed: invalid
  argument` on driver 595 / CUDA 13.1 / B200, eager and compiled alike, so
  `fasth3.yaml` runs the triton route at ~2.5x the build time. Retest each
  fastvideo-kernel release and switch back — it also re-enables regional
  compile, the other half of the measured fast profile.
- **Host memory.** Every rank loads its own text encoder, so CPU-offloading it
  across four ranks would want ~250 GB of host memory *and* pay a
  transfer per clip. `fasth3.yaml` therefore keeps it resident on the GPU, which
  the FSDP-sharded transformer leaves room for. Confirm against measured VRAM
  before changing either flag, and re-size `resources.memory` in the manifest
  from what real hardware uses — the values there are an opening estimate that
  must also cover the built-clip buffer (`queue_size` × ~1 GB).
- **Post-decode cost.** Asking FastVideo for frames in memory
  (`return_frames=True`) also allocates a full fp32 mirror of the decoded video
  and copies into it — several GB per clip that nothing reads, on top of the
  uint8 conversion that is actually needed. The per-clip log line carries the
  stage timings; if `PostDecodeFrameProcessStage` dominates the build, the fix
  belongs upstream in FastVideo rather than here.
- **The dependency closure is not yet exact.** `requirements.txt` lists what the
  model's own code needs and leaves torchvision/torchaudio to the cu130 index.
  Regenerate it from a `pip freeze` of the first successfully built image so a
  rebuild is byte-comparable.
