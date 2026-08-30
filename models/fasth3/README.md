# FastH3

An endless, prompt-driven video and audio channel. Set a prompt and the model
streams 768p video with synchronized audio over WebRTC continuously, always
building the next clip while the current one plays, so the stream never runs
dry. Change the prompt live and the channel follows it.

Reach for this when you want a *channel* rather than a file: a always-on
backdrop, a live visual that responds to typed direction, a stream a room full
of people watches together. If you want one finished clip returned to you, this
is the wrong shape — the model never stops on its own.

[FastH3 Preview v1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is MiniMax-H3 (35B) distilled by FastVideo with data-free DMD2 down to **four
transformer forwards**, with 90% sparse video attention on Blackwell. It
generates video and stereo audio jointly from text. Text-to-video-and-audio is
the only task this checkpoint was distilled for; first/last-frame and reference
conditioning are not.

## Prerequisites

- **Eight NVIDIA B200s.** Eight is load-bearing, not a performance preference:
  sequence parallelism spans all of them, and a 15 s clip builds in 12.88 s on
  eight against 15.5 s on four. Only the former is faster than the video plays,
  which is the whole premise of a continuous channel.
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

# Build the image and serve (needs the weights and eight GPUs).
reactor build
reactor run
```

`load()` warms one throwaway clip per shape before the pod reports ready, so the
first real clip streams at warm speed. Every distinct frame count and canvas is
a separate one-time cost, which is why `inference.warmup_aspects` in
[`fasth3.yaml`](./fasth3.yaml) is a deliberately short list.

Before blaming the adapter for a slow channel, baseline the recipe itself with
FastVideo's own `examples/inference/basic/basic_fasth3.py` at the same settings.
Its median is the number this model should match; a gap is the adapter's fault,
not the model's.

## Tracks

| Track | Direction | Kind | Rate | Payload |
|---|---|---|---|---|
| `main_video` | out | video | 24 fps, fixed | RGB frames at the session's canvas, e.g. 1344×768 |
| `main_audio` | out | audio | 48 kHz | Mono, synchronized frame-for-frame with `main_video` |

There are no inbound tracks: the model reads no camera and no microphone. The
video size is fixed for the life of a channel — `set_canvas` chooses it and is
only accepted while the channel is idle, so the track never changes size
mid-stream.

## Commands

| Command | Parameters | Effect | Rejected with |
|---|---|---|---|
| `set_prompt` | `prompt` (≤ 800 chars) | Sets what the channel shows. Empty clears it. Replies `prompt_accepted`. | — |
| `set_clip_seconds` | `seconds` (5.167–14.375) | Sets clip length; the value is snapped to what the model can produce and the effective one is returned in `clip_length_accepted`. | — |
| `set_seed` | `seed` (≥ 0) | Seed the next channel starts from; each clip advances it by one. Replies `seed_accepted`. | — |
| `set_canvas` | `aspect` (`16:9`, `1:1`, `9:16`, `4:3`) | Sets the video size for the session. Replies `canvas_accepted`. | `channel_running`, `unsupported_aspect` |
| `start` | — | Begins the channel. Emits `channel_started`, then streams. | `already_running`, `missing_prompt` |
| `pause` | — | Freezes playout. Replies `channel_paused`. | `not_running`, `already_paused` |
| `resume` | — | Continues playout. Replies `channel_resumed`. | `not_paused` |
| `stop` | — | Ends the channel, keeping every condition. Emits `channel_stopped`. | `not_running` |
| `reset` | — | Returns every condition to its default and clears the stream. Replies `channel_reset`. | — |
| `get_state` | — | Replies with the full `state_update` snapshot. | — |

A rejected command has no effect. Rejections carry a stable `code` (the values
above) plus a readable message, so a client can branch on the code.

`stop` halts the stream within about a second. The clip already being built
cannot be cancelled, so the model can take several more seconds to go fully
idle; `state_update` reports `running: false` when it has.

## Messages

| Message | Reaches | When |
|---|---|---|
| `state_update` | everyone | On connect, and after every change. A complete snapshot — render from this alone. |
| `prompt_accepted` | the caller | Reply to `set_prompt`. Carries where the prompt lands. |
| `clip_length_accepted` | the caller | Reply to `set_clip_seconds`. Carries the snapped value. |
| `seed_accepted` | the caller | Reply to `set_seed`. |
| `canvas_accepted` | the caller | Reply to `set_canvas`. Carries the exact pixel size. |
| `channel_started` | everyone | `start` accepted. No frames yet. |
| `clip_started` | everyone | A clip begins streaming — this is where the picture and sound cut. |
| `clip_complete` | everyone | A clip has been fully sent. |
| `channel_paused` / `channel_resumed` | the caller | Replies to `pause` / `resume`. |
| `channel_stopped` | everyone | The channel ended via `stop`. |
| `channel_failed` | everyone | The channel ended early because something went wrong; the model returns to idle. |
| `channel_reset` | the caller | Reply to `reset`. |

Anywhere a prompt appears on the wire — `state_update.prompt`,
`prompt_accepted.prompt` — **"not set" is `null`, never an empty string**, so a
client can test one thing rather than two.

## Session lifecycle

```
  session starts (no clients yet)
    |
    v
  client connects            -> state_update (a full snapshot, to this client)
    |
  ┌──────────────────────────────────────────────────────────────┐
  │ IDLE                                                         │
  │ Valid: set_prompt, set_clip_seconds, set_seed, set_canvas,   │
  │        reset, get_state, and start once a prompt is set      │
  │ No frames on either track                                    │
  └───────────────────────────┬──────────────────────────────────┘
                              v  start
  ┌──────────────────────────────────────────────────────────────┐
  │ BUILDING THE FIRST CLIP                                      │
  │ channel_started is emitted immediately; NO frames yet        │
  │ Lasts several seconds — show progress, not a stalled player  │
  └───────────────────────────┬──────────────────────────────────┘
                              v
  ┌──────────────────────────────────────────────────────────────┐
  │ STREAMING                                                    │
  │ Valid: set_prompt, set_clip_seconds, set_seed, pause/resume, │
  │        stop, reset, get_state                                │
  │ Messages: clip_started, clip_complete, state_update          │
  │ Tracks: main_video + main_audio, continuously                │
  └───────────────────────────┬──────────────────────────────────┘
                              v  stop / reset / last client leaves
                        (back to IDLE)
```

**Single session, shared state.** Several clients may watch one session, and
they all see the same channel: a `set_prompt` or a `stop` from any client
affects everyone, and every client receives every `state_update`. Generation is
gated on having an audience — when the last client leaves the channel winds down
at the next clip boundary, so nothing is generated with nobody watching.

A prompt is required before `start`; everything else has a default.

## What to expect from the timing

Two numbers matter, and they are different things:

- **Time to first frame.** Nothing streams until the first clip is fully built.
  The channel deliberately opens with a *shorter* clip (`inference.ramp_seconds`),
  which builds proportionally faster, so this is a few seconds rather than a
  full clip build. `channel_started` fires immediately and carries
  `first_clip_seconds`; treat it as the cue to show progress.
- **Steady state.** From then on the model builds faster than the video plays,
  so the stream should not stall. If the hardware cannot keep up, the symptom is
  a brief freeze at a clip boundary, not a dropped or corrupted stream. `seam
  late` in the log is the gate: it fires when a clip was not ready by the time
  its predecessor ran out, and a clean multi-hour run should never print it. The
  levers, in order, are longer clips, regional compile, and replicated rather
  than sharded weights.

Output is a strict 24 fps metronome and the audio is sample-clocked against the
same rate, so the two tracks stay locked over hours. `pause` freezes the stream
within about an eighth of a second; the model keeps building ahead while paused,
so `resume` continues instantly rather than skipping forward to catch up.

## Clip boundaries are hard cuts

Every clip is generated independently. **The picture and the sound cut at each
boundary** — there is no continuity of subject, framing, or voice from one clip
to the next, even with the prompt unchanged. `clip_started` marks each cut, so a
UI can anticipate it.

This checkpoint has no continuation path, and inventing one is not an adapter's
decision. It is why [`fasth3.yaml`](./fasth3.yaml) defaults to the longest clip
the model supports, and why `runtime.recording` is left disabled: a recording
would carry those cuts too.

The geometry it accepts is narrow, and [`fasth3_clip_plan.py`](./fasth3_clip_plan.py)
encodes it: 24 fps, a frame count of the form `17n + 5`, a duration between 5
and 15 seconds, a short edge of 768, at most 768×1344 pixels, and both sides a
multiple of 32. The duration cap has a sharp edge — 15.0 s is 360 frames, which
aligns *up* to 362 (15.083 s) and is then rejected, so **the longest clip this
model can make is 345 frames, 14.375 s**.

## Changing the prompt mid-channel

The model is always one clip ahead. When clip *k* is playing, clip *k+1* has
already been built with the prompt as it stood earlier, so a prompt sent now
first appears on clip *k+2*.

You never have to work this out. Both `prompt_accepted` and `state_update` carry
`prompt_effective_clip_index` (the clip this prompt will first appear on) and
`prompt_effective_in_seconds` (how much already-built video plays before it).
Render the second one directly — "new prompt in ~21 s". Shorter clips
(`set_clip_seconds`) shorten this wait, at the cost of cutting more often.

## Determinism

`set_seed` fixes the seed the channel starts from, and each clip advances it by
one, so the same seed with the same prompt, clip length and canvas reproduces
the same sequence of clips. Reproduction is approximate rather than bit-exact:
the deployment runs fused and compiled kernels that can reorder floating-point
operations.

## Notes on the code

**`ReactorModel`, not `ReactorPipeline`.** The generator base is shaped for
frame-per-step models, where a `yield` is the natural unit of work. Here the
unit is a whole clip produced by one blocking call, so the generator would buy
nothing and cost the usual workarounds — `pause` holding by yielding `Idle`,
polling the handoff with `get_nowait()`, and session state living in a
runtime-built `InputState`. Under `ReactorModel`, command handlers and lifecycle
hooks run on background coroutines *concurrent* with `run()`, so:

- session state is plain attributes reset in `_reset_session_state`, called from
  `@session_started` (not `@connected`: a client rejoining mid-session keeps the
  channel it joined);
- one persistent worker thread serialises every clip and gives teardown a single
  handle to wait on;
- refusals and accepts answer immediately, even mid-build.

**The lookahead is exactly one clip.** `_run_channel` submits clip *k+1* the
moment clip *k* is dequeued, then paces clip *k* out at a strict 24 fps while
*k+1* builds. The handoff is a `Queue(maxsize=1)`, which is what bounds it —
deeper would only bury prompt changes further behind.

**Refusals are broadcast, not raised.** A handler returns only the message its
annotation names and reports failure by broadcasting `command_error`. A raised
runtime `CommandError` has its failure frame withheld from v0 clients, so the
broadcast is what reaches every SDK generation.

**Nothing is vendored.** FastVideo is a published package, so the whole upstream
tree arrives through `requirements.txt` and an upgrade is a one-line bump.

## Layout

| Path | What it is |
|---|---|
| `fasth3.py` | The `ReactorModel`: weight loading, commands, and the `run()` loop |
| `fasth3_types.py` | Everything a client sees — output tracks and messages |
| `fasth3_clip_plan.py` | Clip geometry: valid lengths, frame counts, canvases |
| `fasth3_session_rules.py` | Which commands each session state accepts |
| `fasth3.yaml` | `inference:` the generation recipe, `runtime:` weight layout and engine shape |
| `reactor.yaml` | The manifest: identity, version, resources, runtime, image build |
| `tests/` | Structural tests that need no GPU |

## Open questions for bring-up

- **Host memory.** Every rank loads its own text encoder, so CPU-offloading it
  across eight ranks would want several hundred GB of host memory *and* pay a
  transfer per clip. `fasth3.yaml` therefore keeps it resident on the GPU, which
  the FSDP-sharded transformer leaves room for. Confirm against measured VRAM
  before changing either flag, and re-size `resources.memory` in the manifest
  from what real hardware uses — the values there are an opening estimate.
- **Post-decode cost.** Asking FastVideo for frames in memory
  (`return_frames=True`) also allocates a full fp32 mirror of the decoded video
  and copies into it — several GB per clip that nothing reads, on top of the
  uint8 conversion that is actually needed. The per-clip log line carries the
  stage timings; if `PostDecodeFrameProcessStage` eats the channel's slack, the
  fix belongs upstream in FastVideo rather than here.
- **The dependency closure is not yet exact.** `requirements.txt` lists what the
  model's own code needs and leaves torchvision/torchaudio to the cu130 index.
  Regenerate it from a `pip freeze` of the first successfully built image so a
  rebuild is byte-comparable.
