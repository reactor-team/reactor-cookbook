# Play YUME-1.5 through Reactor Runtime

Run the public [YUME-1.5 world model](https://github.com/stdstu12/YUME) as an
interactive Reactor backend. Start from text, an uploaded image, or an uploaded
video; change the prompt; hold compatible translation and view keys; stream the
generated video; and record the session.

The adapter loads the distilled public
[`stdstu123/Yume-5B-720P`](https://huggingface.co/stdstu123/Yume-5B-720P)
checkpoint and calls the pinned upstream inference components directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation),
  Docker, and the NVIDIA Container Toolkit.
- One NVIDIA GPU. CPU inference is unsupported; the deployment manifest requests
  one NVIDIA B200.
- Sufficient persistent storage for the source checkout, 5B checkpoint, Hugging
  Face cache, uploaded media, and generated-kernel cache, plus container image
  and build-cache space.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` declares the model,
Reactor Runtime release, GPU resources, recording settings, Python and CUDA
versions, system packages, and Python dependencies. No handwritten Dockerfile
is required. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported manifest fields.

Build the model image, expose one GPU, and start Runtime:

```sh
cd models/yume-1-5
reactor build
reactor run --gpus device=0
```

`reactor run` serves on `http://localhost:8080` and reuses the image produced by
`reactor build`. It builds automatically when the local image is missing. To use
another host GPU and port:

```sh
reactor run --gpus device=4 --port 18095
```

The first start clones the pinned YUME source and downloads the pinned public
checkpoint. Later starts validate and reuse both from `runtime.weights_path`.
Check readiness and inspect the generated client contract with:

```sh
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/schema
```

## Runtime boundary

YUME-1.5 is an autoregressive video world model. Model weights load once at
process startup, while each session owns one rolling clean-latent history and
random generator. Each inference turn denoises one native eight-latent tail and
emits 29 RGB frames on `main_video`. The generated tail is retained as context
for the next turn.

The public 5B checkpoint does not expose a Transformer KV cache. The adapter
preserves YUME's full clean latent context and its upstream temporal compression
instead of introducing a separate cache. Prompt changes affect the next chunk
without discarding that visual history.

Generation follows the upstream synchronous inference path. Model messages are
sent outside that path at command and chunk boundaries.

## Controls

Generation waits for one of the scene-selection commands:

- `set_image(image, prompt, seed)` starts a fresh world from an uploaded image.
  A blank prompt uses the configured neutral continuation description.
- `set_video_scene(video, prompt, seed)` starts from the first 33 frames of an
  uploaded video and requires a non-empty prompt.
- `set_text_scene(prompt, seed)` starts without reference media and requires a
  non-empty prompt.
- `set_prompt(prompt)` changes text conditioning at the next chunk boundary
  without clearing visual history.
- `set_key_state(key, pressed)` presses or releases a persistent translation or
  view key.
- `release_controls` releases every held key for forthcoming chunks.
- `reset(seed)` restarts the selected scene, releases all controls, and
  optionally selects another seed.

Translation uses `w`, `a`, `s`, and `d`; view control uses `arrow_left`,
`arrow_right`, `arrow_up`, and `arrow_down`. Compatible keys remain held and may
be combined: forward/backward can pair with left/right translation, pan can pair
with tilt, and one translation combination can run simultaneously with one view
combination. Opposite directions are rejected with `command_error`.

YUME encodes these controls in its text condition. Stationary chunks explicitly
set movement distance, angular change rate, and view rotation speed to zero;
active translation and view controls use the released caption vocabulary and
control values.

## Image and video uploads

`set_image` accepts JPEG, PNG, WebP, BMP, or TIFF files up to 25 MiB and 100
million pixels through Reactor's upload protocol. The image anchors the first
frame and is fitted to YUME's 1280×704 output.

`set_video_scene` accepts a decodable video up to 500 MiB with at least 33
frames. The first 33 frames initialize the continuation. This recipe does not
provide a default image; generation begins only after the client selects text or
uploads media.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `scene_queued` reports an accepted text, image, or video scene.
- `prompt_changed`, `action_changed`, and `rollout_reset_queued` report accepted
  prompt, held-key, and reset changes and their first affected chunk.
- `chunk_completed` reports the prompt, exact YUME control condition, movement,
  view, frame count, and wall-clock generation time for one completed chunk.
- `state_update` is a complete snapshot of the selected mode and media, prompt,
  held keys, seed, reset and generation flags, and chunk progress. A joining
  client receives one immediately, and each successful mutation broadcasts
  another.

## Public source and model assets

`yume.yaml` pins upstream source revision
`111c3fab7fb020d1e261a68be6ec78a3fecc8d5b` and checkpoint revision
`b117c778405246a0b932d7ec5873b01a4840da3a`. Downloads resume into the
CLI-mounted weights directory, and startup reuses valid existing assets.

The checked-in `runtime.weights_path` is
`~/.cache/reactor_registry/yume-1-5`. To keep that stable CLI path while placing
its contents on a larger volume, create a symlink before the first run:

```sh
export YUME_ROOT=/path/to/large-volume/yume-1-5
mkdir -p "$YUME_ROOT" "$HOME/.cache/reactor_registry"
ln -s "$YUME_ROOT" "$HOME/.cache/reactor_registry/yume-1-5"
```

The container engine stores image layers and build cache separately from model
weights. Configure the engine's storage on a large volume when the system disk
is constrained.

## Recording

`reactor.yaml` records `main_video` by default and allows clips up to five
minutes. YUME-1.5 emits video without audio. Stop `reactor run` to remove the
serving container and release its GPU memory.
