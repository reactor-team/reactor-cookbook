# Play DreamX-World through Reactor Runtime

Serve the public [DreamX-World-5B](https://github.com/AMAP-ML/DreamX-World)
distilled autoregressive world model through Reactor Runtime. Start from an
uploaded or built-in image, describe the world with a prompt, and move through
it with DreamX's native composable keyboard camera controls.

The adapter calls the model, scheduler, camera conditioning, denoiser, VAE, and
rolling KV-cache implementation from a pinned upstream checkout. Model weights
and causal state stay resident across chunks.

DreamX generates three latent frames per turn. Streaming VAE decode emits 9 RGB
frames for the first turn and 12 for every later turn. Playback adapts to
measured inference throughput, and the output queue holds one complete 12-frame
chunk. Prompt and held-key state are sampled at the chunk boundary, so a command
received during an in-flight turn applies to the following one.

A new session starts without choosing a scene. Upload an image with
`set_image`, or call `random_image` to select one of the public DreamX examples.
Either command resets the autoregressive state and starts continuous generation
from chunk 1.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, recent driver, and NVIDIA Container Toolkit. The manifest
  requests one B200; a compatible GPU with at least 48 GB of VRAM is sufficient
  for local development. CPU inference is unsupported.
- Public network access to GitHub and Hugging Face for first-run source and
  model downloads. Neither checkpoint repository is gated.
- At least 80 GB on a fast filesystem for the model image, build cache, and
  selected model assets.

## Run

The `build` block in `reactor.yaml` controls the model image: Reactor Runtime
3.2.5, Python 3.12, CUDA 12.8.1, system packages, and `requirements.txt`. See
Reactor's [build configuration](https://docs.reactor.inc/deploy/platform/build)
for the supported fields. The host needs only the prerequisites above.

Build the model image, expose one GPU, and choose a free port:

```sh
cd models/dreamx-world
reactor build
reactor run --gpus device=0 --port 18085
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container. `reactor run` reuses the image produced by `reactor build`, and
builds it automatically when the local tag is missing. Rebuild after changing
model code or dependencies.

Check readiness after the model finishes loading:

```sh
curl -s localhost:18085/health
curl -s localhost:18085/schema
```

Stop `reactor run` with Ctrl-C to remove the container and release its GPU
memory.

## Keep large data off the system disk

First startup downloads the 21 GB DreamX checkpoint and the required Wan2.2
text encoder, tokenizer, and VAE.

`runtime.weights_path` controls the host directory that `reactor run` mounts for
source and model assets. Its checked-in value is portable. On a
space-constrained host, change it to an absolute path on a larger volume before
the first run:

```yaml
runtime:
  weights_path: /mnt/fast/reactor/dreamx-world
```

The container engine stores image layers and build caches separately from the
workspace. Configure that storage on the same large volume when the system disk
is small.

## Public source and model assets

On first load the adapter clones DreamX-World at the immutable revision in
`dreamx_world.yaml` and downloads the pinned `GD-ML/DreamX-World-5B` checkpoint
plus the required public files from `Wan-AI/Wan2.2-TI2V-5B`. Neither repository
is gated. Interrupted Hugging Face downloads resume on the next start, and
completion markers let later runs verify and reuse the assets.

The source checkout lives under `runtime.weights_path`, outside both the image
and cookbook checkout. Startup verifies its exact revision and rejects tracked
modifications. To reuse an existing clean checkout, place it inside the mounted
weights directory and pass its absolute mounted path:

```sh
reactor run --gpus device=0 --port 18085 \
  -e DREAMX_WORLD_PATH=/mnt/fast/reactor/dreamx-world/DreamX-World
```

The adapter and upstream DreamX source are Apache-2.0. Downloaded checkpoints
retain the terms published by their source repositories.

## Controls

- `set_image(image, prompt)` accepts an uploaded image and optional prompt,
  resets the world, and starts continuous generation. A blank
  prompt uses the active prompt or configured upload default.
- `random_image()` selects one configured upstream example with its original
  evaluation prompt and starts continuous generation.
- `set_prompt(prompt)` changes the cross-attention condition at the next chunk
  boundary without discarding visual KV history.
- `set_key_state(key, pressed)` holds or releases one native camera key. `W`/`S`
  move forward/backward, `A`/`D` strafe left/right, `I`/`K` tilt up/down, and
  `J`/`L` pan left/right.
- `reset(seed)` restarts from the selected image and prompt, clearing the
  autoregressive, cross-attention, camera, and streaming VAE caches. `-1`
  retains the active seed.

Key states persist until released, and translation plus view keys compose at
chunk boundaries.

## Start from an image

`set_image` declares an `UploadedFile` in Runtime's schema, allowing a
schema-driven client to send bytes through the session upload protocol. JPEG,
PNG, WebP, and BMP are accepted up to 25 MiB and 100 million decoded pixels.
DreamX resizes the image to its native 1280×704 inference resolution.

[`example_images/`](./example_images) contains three unmodified public DreamX
demo images ready for manual upload. Their upstream paths, immutable revision,
hashes, and license are recorded in
[`example_images/README.md`](./example_images/README.md). These convenience
copies are independent of `random_image`, which loads the paired images and
prompts from the pinned source checkout.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `image_selected` identifies the uploaded or built-in image and the effective
  prompt for the fresh rollout.
- `prompt_queued` reports the normalized prompt and first chunk expected to use
  it.
- `action_changed` contains the complete held-key set.
- `rollout_reset_queued` confirms rollout transitions before inference consumes them.
- `chunk_generated` reports the sampled prompt and keys, frame count, chunk
  number, and wall-clock inference time.
- `state_update` is a complete snapshot of image, prompt, held keys, seed,
  chunk progress, and generation state. A joining viewer receives one
  immediately; successful state changes broadcast another.

Message delivery stays outside the blocking GPU inference call.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes. DreamX-World does not emit audio.

## Autoregressive state and long sessions

The adapter advances exactly one upstream-native chunk per inference turn and
bounds Runtime's output queue at one chunk. It preserves DreamX's 12-latent
rolling visual KV window, three-latent sink, four-step denoising schedule,
context-cache commit, chunk-relative PRoPE camera condition, and cached VAE
decode.

Prompt changes invalidate only the prompt-specific cross-attention cache. Past
visual tokens stay in the rolling causal cache, so a new event prompt affects
the next chunk without replacing the current world.

`stream.max_chunks_per_rollout` defaults to 512. After chunk 512 the adapter
broadcasts `rollout_reset_queued`, releases held keys, and starts a fresh
rollout from the selected image, active prompt, and seed without reloading model
weights. Chunk numbering then returns to 1, bounding causal and camera state
while keeping the Reactor session alive.
