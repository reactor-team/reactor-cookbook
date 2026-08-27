# Play OpenDreamer through Reactor Runtime

Run the public [OpenDreamer world model](https://github.com/next-state/open-dreamer)
as an interactive Minecraft Reactor backend. Use this recipe when a client
needs to start from a dataset demo or uploaded Minecraft frame, apply native
keyboard and mouse actions, stream generated video, and record the session.

The adapter uses an exact tested OpenDreamer revision in the model image and
calls its public inference components directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  intentionally unsupported.
- About 8 GB in Reactor's weights cache for the checkpoint, plus Docker image
  space for CUDA dependencies and the 184 MB public VPT sample.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` declares the model,
runtime, recording, and complete CUDA-capable image build. `requirements.txt`
contains only the model's Python dependencies. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported YAML fields.

Run the workspace and expose one GPU to the container:

```sh
cd models/open-dreamer
reactor run --gpus device=0
```

The first run builds the image and downloads the model weights, so it takes a
while. Once the model is ready it serves at `http://localhost:8080`, and later
runs start faster because they reuse the built image and the downloaded
weights.

Use `reactor build` when you want to build without starting the service, or to
rebuild after editing anything baked into the image:

```sh
reactor build
reactor run --gpus device=0
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)** and `http://localhost:8080`, or point the
[JS SDK](https://docs.reactor.inc/sdk-reference/using-the-sdk) at it with
`local: true`. A quick liveness check:

```sh
curl -s localhost:8080/health
```

## Controls

- `set_key_state(key, pressed)` holds or releases `w`, `a`, `s`, `d`,
  `space`, `shift`, `ctrl`, `e`, `q`, `escape`, `f`, `1`-`9`, or
  `f3`.
- `set_mouse_button_state(button, pressed)` holds or releases `left`,
  `right`, or `middle`.
- `mouse_move(delta_x, delta_y)` supplies relative camera movement for one
  generated frame.
- `mouse_wheel(delta)` supplies a `-1`, `0`, or `1` hotbar scroll for one
  frame.
- `set_demo(demo)` starts from one reproducible dataset window.
- `random_demo` starts from a randomly selected dataset window.
- `set_conditioning_image(image)` starts from one uploaded Minecraft frame.
- `reset(seed)` clears the incremental caches and optionally changes the seed.

Keyboard keys and mouse buttons remain held until released. Mouse movement and
wheel input are consumed after one successful generation step. Ending the
session releases all controls.

## Start from a dataset demo

Each session selects one of three dataset demos at random. Each demo is a
16-frame window with frame-aligned VPT actions from one public OpenAI recording.
`set_demo` selects `demo_1`, `demo_2`, or `demo_3`; `random_demo`
chooses another window randomly. Both reset the model's incremental KV caches.

The model image includes the paired MP4 and JSONL in the upstream checkout.
They are internal dataset conditioning assets; clients do not upload them.

## Start from an image

Ready-to-upload dataset frames live in [`example_images`](example_images).
Upload one through the client, then call `set_conditioning_image` with its
upload reference:

```js
const image = await uploadFile(file);
await sendCommand("set_conditioning_image", { image });
```

The adapter center-crops the image to 640x360, pads it to OpenDreamer's 640x368
tokenizer shape, repeats it for the 16-frame context, and pairs each frame with
a neutral VPT action. The next inference boundary resets the caches and starts
automatically from that image.

A single image has no motion history, so its rollout may be less stable than a
dataset demo backed by consecutive frames and aligned actions. Arbitrary images
are accepted, but Minecraft frames from the model's training distribution give
the most reliable results.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `action_changed` contains the originating control, held keys and
  mouse buttons, and the mouse or wheel movement received.
- `conditioning_changed` contains the selected demo or uploaded image
  filename.
- `rollout_reset` contains the selected seed and retained conditioning source.
- `state_update` is a complete snapshot of the held controls,
  seed, and conditioning selection. A joining viewer receives one immediately;
  successful state changes broadcast another.

Message delivery stays outside the synchronous inference loop.

Each autoregressive turn emits one RGB frame. Playback adapts to measured
inference throughput, and the output queue holds that complete one-frame turn.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and
allows clips up to five minutes. The model image includes FFmpeg. OpenDreamer
does not emit audio.

## Notes

- `opendreamer.yaml` pins the upstream source, checkpoint, sampling schedule,
  and demo offsets. The `reactor.yaml` build installs the same source revision
  under `/opt/open-dreamer` and sets `OPENDREAMER_PATH` inside the image.
- The adapter follows the upstream CUDA 12 dependency lock and keeps
  training-only packages out of the runtime environment.
- Selecting the same demo and reset seed reproduces the same rollout. The
  automatic demo selected for a new session is intentionally random.
- Ending a session releases its controls and uploaded conditioning. Stop
  `reactor run` to remove the container and release its GPU memory.
