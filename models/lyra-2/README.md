# Play Lyra 2.0 through Reactor Runtime

Run NVIDIA's public [Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2)
autoregressive video world model as an interactive Reactor backend. Start from
an uploaded or built-in image, change the prompt without resetting the world,
and explore with continuous six-axis camera motion.

This recipe selects the released four-step DMD video model. It does not run
Lyra's 3D-asset generation workflow.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation),
  Docker, and the NVIDIA Container Toolkit.
- One NVIDIA GPU with substantial memory. The tested B200 configuration peaks
  near 66 GB of allocated VRAM; an 80 GB or larger GPU is recommended. CPU
  inference is unsupported.
- Enough persistent storage for the Lyra source, checkpoints, and model image.
  The tested image is about 12 GB before model weights.

## Run

This directory is a Reactor workspace. There is no Dockerfile: the `build:`
section in `reactor.yaml` declares the Python 3.12, CUDA 12.8.1, system-package,
Python-package, and pinned upstream source build. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported YAML fields.

Validate the workspace, build the generated image, and expose one GPU to the
container:

```sh
cd models/lyra-2
reactor validate
reactor build
reactor run --gpus device=0
```

`reactor run` reuses the image from `reactor build` and automatically builds it
when the local tag is missing. Rebuild after changing the adapter, manifest, or
dependencies. `--gpus device=4` selects host GPU 4 and presents it as device 0
inside the container.

The default endpoint is `http://localhost:8080`. Choose another port when
needed:

```sh
reactor build && reactor run --gpus device=4 --port 18086
```

The service reports healthy only after the model and reconstruction components
are resident:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
```

## Runtime boundary

Lyra 2.0 is an autoregressive video model. The adapter calls the released
`autoregressive_step` once per Reactor inference turn and emits exactly one
native 80-frame RGB chunk. Model weights load once, while the full latent
history, streaming VAE encoder and decoder caches, camera trajectory, DA3 depth
state, and `Sparse3DCache` remain alive across chunks.

The four-step DMD schedule is enabled. Diffusion, VAE, CLIP, DA3, and UMT5 stay
on the GPU; the optional Qwen image captioner is disabled because every rollout
already has an explicit prompt. Completed RGB frames move to host memory only
at the Runtime output boundary.

The adapter retains Lyra's default memory horizon instead of shortening its
history. Prompt changes are sampled at the next chunk boundary without clearing
the current world. Selecting an image or calling `reset` starts a fresh rollout
without reloading model weights.

## Controls

Generation waits for `set_image` or `random_image`, then starts continuous
playback.

- `set_image(image, prompt, seed)` starts a fresh world from an uploaded image.
  A blank prompt uses the configured generic continuation prompt.
- `random_image` starts from a randomly selected public Lyra sample and its
  paired prompt.
- `set_prompt(prompt)` changes text conditioning at the next chunk boundary
  while preserving the current autoregressive history.
- `set_camera_motion(forward, strafe, vertical, pitch, yaw, roll)` atomically
  sets all six held camera axes for forthcoming chunks.
- `release_camera` returns all six axes to neutral.
- `reset(seed)` restarts from the selected image and prompt, optionally using a
  new non-negative seed.

Every camera value is normalized to `[-1, 1]`, with zero neutral. `forward`,
`strafe`, and `vertical` control translation; `pitch`, `yaw`, and `roll` control
rotation. Values remain active until changed or released and are sampled as one
complete trajectory at a chunk boundary.

## Image uploads

`set_image` uses Reactor's upload protocol and accepts JPEG, PNG, WebP, or BMP
files up to 25 MiB. The adapter validates the bytes before replacing the current
world and resizes the image to Lyra's 768×448 generation resolution.

A schema-driven client can reserve an upload slot, write the raw file to its
returned URL, and pass the resulting upload reference to `set_image`. There is
no default image; generation remains idle until a client uploads an image or
calls `random_image`.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `image_selected`, `prompt_queued`, `camera_changed`, and `reset_queued`
  report accepted changes and their first affected chunk.
- `chunk_completed` reports the one-based chunk index, 80-frame output count,
  sampled prompt, and wall-clock generation time.
- `state_update` is a complete snapshot of the selected image, queued and active
  prompts, seed, generation state, completed chunks, and all six camera axes. A
  joining viewer receives one immediately, and every successful mutation
  broadcasts another.

Message delivery stays outside the synchronous chunk-generation path.

## Public source and model assets

The manifest builds an exact tested Lyra revision and its VIPE and Depth
Anything 3 submodules into the image. Runtime expects the released Lyra 2.0
checkpoints beneath `runtime.weights_path`; the adapter reuses those files on
later starts.

The checked-in weights path is `~/.cache/reactor_registry/lyra-2`. Point that
path at a high-capacity volume before the first run when the system disk is
small. Container image layers and BuildKit cache are managed by Docker and
should be moved separately through Docker's `data-root` configuration.

## Inference performance

The tested B200 path uses Transformer Engine 2.7 and FlashAttention 2.8.1. A
typical 80-frame chunk takes roughly 30 seconds after warmup. The diffusion
step, CUDA VAE decode, camera conditioning, and DA3 spatial-memory update all
run once per chunk; camera and spatial-memory work can increase as the rollout
history grows.

The output queue holds one complete native chunk. Reactor does not impose a
model FPS in the adapter; Runtime paces the emitted frames for WebRTC playback
and recording.

## Recording

`reactor.yaml` records `main_video` as H.264 by default in five-second segments
and allows clips up to ten minutes. Lyra 2.0 emits video without audio.

## Notes

- `lyra2.yaml` controls the default prompt and per-frame translation and
  rotation scale.
- The pinned FlashAttention version is required by the tested Transformer
  Engine combination.
- Ending a session clears rollout caches while keeping loaded model weights
  available for the next session.
- Stop `reactor run` to remove the container and release its GPU memory.
