# Play LingBot-World v1 Fast through Reactor Runtime

Run the public [LingBot-World](https://github.com/robbyant/lingbot-world)
camera model as an interactive Reactor backend. A client can start from a
built-in scene or uploaded image, change the prompt, apply six-axis camera
motion, stream generated video, and record the session.

The adapter loads an exact tested LingBot-World revision from the persistent
weights cache and applies a stateful source extension that exposes its native
chunk boundary.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  unsupported. The recipe requests one B200; a tested 13-chunk rollout used
  about 48 GB of VRAM, so a GPU with at least 64 GB is recommended.
- About 90 GB on persistent storage for the Fast checkpoint, VAE, UMT5 assets,
  source checkout, and worker environment, plus space for the model image and
  build cache.

## Run

This directory is a `reactor` workspace. `reactor.yaml` names the model,
controls its Reactor Runtime 3.2.5, CUDA 12.8.1, Python 3.12, system packages,
and Python dependencies. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Validate the workspace, build the model image, and expose one GPU to the
container:

```sh
cd models/lingbot-world-v1-fast
reactor validate
reactor build
reactor run --gpus device=0
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container. `reactor run` reuses the image from `reactor build` and builds it
automatically when the local tag is missing. Rebuild after changing adapter
code, dependencies, or the manifest.

First startup clones the pinned public source, prepares an isolated upstream
environment, builds FlashAttention 2, and downloads the selected model assets.
Later starts reuse all of them from the CLI-mounted weights cache. Public
downloads require no token, but a Hugging Face token can be forwarded without
putting its value on the command line when needed:

```sh
export HF_TOKEN=hf_your_read_token
reactor run --gpus device=0 -e HF_TOKEN
```

The default endpoint is `http://localhost:8080`. Pass `--port` to use another
port:

```sh
reactor run --gpus device=0 --port 18087
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)**, or check readiness directly:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
```

## Controls

Generation waits for `set_image` or `random_image`, then begins a fresh
continuous rollout.

- `set_image(image, prompt)` starts a fresh world from an uploaded image and
  optionally replaces the prompt.
- `random_image` selects another built-in scene and its matching prompt.
- `set_prompt(prompt)` applies a non-empty text condition at the next chunk
  boundary without clearing visual self-KV history.
- `set_forward`, `set_strafe`, and `set_vertical` control translation.
- `set_pitch`, `set_yaw`, and `set_roll` control rotation.
- `reset(seed)` starts the selected image and prompt again without reloading
  model weights.

Every camera value is normalized to `[-1, 1]`, with zero neutral. A WASD
frontend maps W/S to forward ±1 and A/D to strafe ∓1. Pointer, arrow, touch, or
gamepad input can drive yaw and pitch; additional controls can expose vertical
translation and roll. Axes are independent, so translation and rotation can be
combined in the same chunk.

## Start from an image

`set_image` uses Reactor's upload protocol and accepts decoded JPEG, PNG, WebP,
or BMP files up to 25 MiB and 100 million pixels. Uploading an image starts a
fresh world and continuous generation from its first chunk.

LingBot-World's camera path expects calibrated intrinsics. An uploaded image
uses the first public sample's calibration until `random_image` selects another
one, matching the upstream fixed 480×832 inference path. The public lakeside
and Great Wall anchors arrive in the pinned source checkout; their locations
are documented in [`example_images`](example_images).

## Runtime boundary

LingBot-World Fast is an autoregressive video model. The included
`InteractiveFastRollout` runs its native three-latent, four-timestep boundary as
a resumable session: model weights load once, and self-KV, absolute RoPE
position, prompt cross-attention, random generator, and causal VAE feature
cache remain alive for the next request.

The first three-latent chunk decodes to 9 RGB frames because it includes the
image anchor. Every later chunk decodes to 12 frames. Each turn is submitted as
one frame batch; playback adapts to measured inference throughput, and the
output queue holds one complete 12-frame chunk. The official 81-frame Fast
invocation contains 21 VAE latents; the adapter keeps that exact rolling native
KV window for every session.

Prompt changes replace only cross-attention conditioning at the next chunk
boundary. Visual self-KV remains intact, so future video responds to the new
text while preserving generated visual history.

## Model messages

Every accepted command returns its own typed result — `image_selected`,
`prompt_queued`, `camera_motion_changed`, or `rollout_reset_queued` — and
broadcasts a complete `state_update` snapshot for the client timeline. The
snapshot contains the active image, prompt, seed, all six camera axes,
completed chunk count, next affected chunk, frame count, rollout limit, and
most recent generation time. A joining viewer receives the same state
immediately.

`rollout_limit_reached` reports when generation stops at the configured safe
RoPE boundary. `reset`, `set_image`, or `random_image` starts a fresh timeline.
Message delivery remains outside the synchronous inference loop.

## Public source and model assets

`lingbot_world_v1.yaml` pins the source and both Hugging Face snapshots. First
load downloads the base VAE, UMT5 encoder and tokenizer, and Fast model shards.
Downloads resume after interruption, and marker files record the immutable
revisions.

Reactor Runtime and LingBot-World require different NumPy major versions. A
persistent Python 3.12 worker provides dependency isolation while remaining the
only model process and loading one copy of the weights. The source extension
delegates each chunk to the upstream scheduler, four denoising timesteps,
self-KV eviction, camera Plücker embedding, and VAE decoder.

The CLI bind-mounts `runtime.weights_path` at the same absolute path inside the
container and exports it as `REACTOR_WEIGHTS_PATH`. The checked-in default is
`~/.cache/reactor_registry/lingbot-world-v1-fast`. To keep that stable CLI path
while placing its contents on a larger volume, create a symlink before the
first launch:

```sh
export LINGBOT_ROOT=/path/to/large-volume/lingbot-world-v1
mkdir -p "$LINGBOT_ROOT/weights" "$HOME/.cache/reactor_registry"
ln -s "$LINGBOT_ROOT/weights" \
  "$HOME/.cache/reactor_registry/lingbot-world-v1-fast"
```

The container engine stores image layers and build cache separately from model
weights. Configure that storage on a large volume before `reactor build` when
the system disk has limited space.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes. The model emits video without audio.

## Notes

- `stream.context_latents: 21` preserves the upstream Fast memory length; each
  inference turn requests one native three-latent chunk.
- `stream.max_chunks: 320` keeps absolute RoPE positions inside the upstream
  1024-latent table. A fresh image or reset starts another timeline without
  reloading the checkpoint.
- Ending a session releases KV, cross-attention, camera, and causal VAE caches
  while retaining loaded model weights for the next session.
- Stop `reactor run` to remove the container and release its GPU memory.
