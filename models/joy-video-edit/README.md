# Edit a live camera stream with JoyAI-Video-Edit

Serve [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit)
([paper](https://arxiv.org/abs/2608.03974)) as a real-time Reactor model. A
client streams video on `camera`, sets a natural-language edit instruction
(for example "Turn the video into a watercolor painting"), optionally uploads a
reference image, and calls `start`. The edited video streams back on
`main_video` at 1248x720, chunk by chunk, as the frames arrive: the model never
sees future frames and needs no video length.

Reach for it when the input is live (a webcam, a screen, a video played in real
time) and the edit is open-ended: style transfer, local object edits, adding,
removing or replacing a subject, background changes, or reference-guided
edits.

This folder carries a modified copy of the upstream inference code, tuned for
one GPU. On one B200 a steady chunk of 8 frames takes about 0.2 s, about 40
edited frames per second, from the first chunk of a session on.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA B200 or B300 GPU, its driver, and the NVIDIA Container Toolkit.
  The kernels are built for the Blackwell datacenter family; the recipe is
  tested on one B200.
- About 50 GB of persistent storage for the checkpoints, plus space for the
  image (about 22 GB) and its build cache.
- 128 GB of host memory: loading the 16B DiT checkpoint peaks at about 63 GB.

## Run

This directory is a `reactor` workspace. Its `Dockerfile` builds the image:
CUDA 13.2, PyTorch 2.14, Reactor Runtime 3.6.0 (keep the Dockerfile's
`RUNTIME_VERSION` equal to `build.runtime_version` in `reactor.yaml`), and the
`joyomni_ops` kernels compiled from `./joyomni_ops` against CUTLASS.

```sh
cd models/joy-video-edit
reactor validate
reactor build
reactor run --gpus device=0
```

First startup downloads the pinned checkpoints into the CLI-mounted weights
cache (`runtime.weights_path`, `~/.cache/reactor_registry/joy-video-edit` by
default), about 48 GB, then warms up: it runs whole editing sessions and
captures every CUDA graph a session can use, so the first client's first chunk
is as fast as its hundredth. The compile caches are kept under the weights
cache, so later starts are faster. Public downloads need no token; to forward
one without putting it on the command line:

```sh
export HF_TOKEN=hf_your_read_token
reactor run --gpus device=0 -e HF_TOKEN
```

If you already have the checkpoints in the layout `joy_video_edit.yaml`
describes, set `checkpoint_dir` to that absolute directory and nothing is
downloaded.

## Commands

[`model_behaviour.md`](./model_behaviour.md) has the full client contract: the
states, which commands each accepts, message ordering, and SDK sequences.

| Command | When | Effect |
| --- | --- | --- |
| `set_prompt` | any time | Stores the edit instruction. `start` reads it; a change during a run applies from the next `start`. Replies `prompt_accepted`, broadcasts `session_state`. |
| `set_reference_image` | while not generating | Stores a reference image (PNG or JPEG) whose appearance guides the edit. It is fitted, aspect kept, into the nearest of a square, 4:3 or 3:4 frame and padded with grey. Replies `reference_image_accepted`, broadcasts `session_state`; `command_error` while generating. |
| `set_seed` | any time | Stores the noise seed. `start` reads it. |
| `start` | while not generating | Begins a run with the current instruction, reference image and seed. Replies `generation_started`, broadcasts `session_state`; `command_error` if already generating. |
| `reset` | any time | Ends the run and returns to waiting; the instruction, reference image and seed are kept. While generating, broadcasts `generation_reset` then `generation_complete`; always broadcasts `session_state`. |

Frames start flowing once `camera` delivers video. `set_prompt`'s instruction
and the reference image are moderated when the deployment moderates.

## Messages

| Message | Sent |
| --- | --- |
| `session_state` | On connect, after each command above except `set_seed`, and when a run ends: `started`, `prompt`, `has_reference_image`, `seed`. |
| `chunk_complete` | Before each chunk's frames go out on `main_video`: the chunk's index in the run, its frame count (usually 8; the first chunk of a run, and an occasional later one, carry 1), and the milliseconds it took. |
| `generation_complete` | When a run ends, by `reset` or when the last client disconnects: the run's chunk and frame totals. |
| `prompt_accepted`, `reference_image_accepted`, `generation_started`, `generation_reset`, `command_error` | As listed under Commands. |

## How it is served

- **One run, one conditioning.** `start` fixes the instruction, the reference
  image and the seed for the run. The model edits in chunks: the first chunk
  of a session takes 1 camera frame and every later chunk 8, and the model
  half tells the application how many frames the next step needs.
- **Latest frames first.** Each chunk reads the newest 8 camera frames and
  drops older queued ones, so the output follows the input with about one
  chunk of delay. `camera_read: fifo` in `joy_video_edit.yaml` keeps every
  frame instead, dropping only a backlog beyond `camera_backlog_chunks`.
- **Adaptive playout.** `main_video` plays each chunk at its frame count over
  the time it took to edit, so a slower chunk plays slower instead of
  freezing on its last frame.
- **Bounded drift.** Every 600 camera frames (`kv_reset_frames`) the run opens
  a fresh session on the same conditioning; the chunk after it carries 1
  frame.
- **One retry.** A chunk that fails part-way is retried once on a fresh
  session of the same run; a second failure in a row ends the session with an
  error.

The application half (`joy_video_edit.py`) owns the client contract, the
camera reads, and the messages. The model half (`joy_video_edit_model.py`)
owns the weights and the streaming session and imports nothing from Reactor
Runtime; every GPU call runs on one persistent compute thread. They meet on
`JoyVideoEditInput` and `JoyVideoEditResult`.

## Source and model assets

`xvideo/` and `joyomni_ops/` are vendored from upstream's `deploy/` directory
and modified; their `PROVENANCE.md` files name the upstream commit and list
every change. Both are Apache-2.0. `scripts/prove_parity.py` runs this folder
and upstream side by side on the same clips, inside the model image, and
compares their latents and pixels chunk by chunk.

`joy_video_edit.yaml` pins the checkpoints:

- [`jdopensource/JoyAI-Video-Edit`](https://huggingface.co/jdopensource/JoyAI-Video-Edit)
  at `39491dd`: the `joyai_video_edit_dit_0811.pth` DiT and the VAE
  (Apache-2.0).
- [`XiaomiMiMo/MiMo-VL-7B-RL-2508`](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508)
  at `4bfb270`: the multimodal condition encoder (MIT).

## Tests

The tests drive the application half with a fake model half and the model
half with the GPU work faked, so they need neither a GPU nor the weights:

```sh
pip install "reactor-runtime==3.6.0" pillow pytest
PYTHONPATH=. python -m pytest tests/ -q
```

## Notes

- Recording is off in `reactor.yaml`: the session recorder encodes video on
  the same GPU the model edits on, which costs most of the model's throughput.
- `reset` does not cut `main_video`; the last chunk plays out.
- Stop `reactor run` to remove the container and release its GPU memory.
