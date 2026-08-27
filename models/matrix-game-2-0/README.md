# Play Matrix-Game-2.0 through Reactor Runtime

Run SkyworkAI's public
[Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game) universal distilled
model as an interactive Reactor backend. Use this recipe when a client needs to
start from a public demo or uploaded image, apply native keyboard and
mouse-camera controls, stream autoregressive video, and record the session.

The adapter loads an exact tested source revision and checkpoint from Reactor's
mounted weights cache and calls the upstream autoregressive components
directly. The `build:` block in `reactor.yaml` controls the model image.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  intentionally unsupported. The published resource request targets one B200.
- About 12 GB in Reactor's weights cache, plus approximately 30 GB of Docker
  image and build working space.

When the system disk is small, place Reactor's weights cache and the container
engine's image storage on a high-capacity volume. `runtime.weights_path`
controls the host directory mounted for source and checkpoints.

## Run

This directory is a `reactor` workspace. `reactor.yaml` names the model,
configures its Reactor Runtime 3.2.5 image, mounts the persistent weights cache,
and enables recording. `requirements.txt` contains the inference dependencies.
See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Check the host, build the image, and expose one free GPU to the container:

```sh
cd models/matrix-game-2-0
reactor doctor
reactor build
reactor run --gpus device=0 --port 8080
```

The configured image contains Reactor Runtime 3.2.5, Python 3.12, CUDA 12.8,
the model dependencies, and the required system packages. `reactor run` reuses
the local image, building it automatically when its tag is absent.

On first load the adapter sparsely clones the pinned `Matrix-Game-2` source and
downloads the universal distilled checkpoint, Wan 2.1 VAE, image encoder, and
tokenizer files from the pinned Hugging Face snapshot. Interrupted downloads
resume from the mounted cache, and later runs reuse the same assets.

Check readiness and inspect the generated command schema with:

```sh
curl -s http://localhost:8080/health
curl -s http://localhost:8080/schema
```

Rebuild after editing files copied into the model image:

```sh
reactor build && reactor run --gpus device=0 --port 8080
```

## Controls

A session starts with no scene selected. Selecting an image starts
continuous generation from the first chunk.

- `set_key_state(key, pressed)` holds or releases `w`, `a`, `s`, or `d` for
  forthcoming chunks. Held keys persist and can be combined; W+A and W+D become
  the same multi-hot actions used by the official universal model.
- `set_pitch(pitch)` holds normalized look-down or look-up velocity in
  `[-1, 1]`.
- `set_yaw(yaw)` holds normalized turn-left or turn-right velocity in
  `[-1, 1]`.
- `release_controls` returns keyboard and camera conditions to neutral.
- `reset(seed)` clears every autoregressive cache, rebuilds the selected image
  conditioning, and automatically generates a fresh first chunk. Pass `-1` to
  keep the active seed.

Keyboard and camera values are sampled together at the next chunk boundary.
Changing a control while one chunk is in flight applies it to the following
chunk. Ending the session releases all controls.

## Start from a public demo

`random_image` selects one of the public universal example images in the pinned
Matrix-Game checkout. It clears the active rollout and all controls, rebuilds
image conditioning, and automatically queues one chunk. Repeated calls choose a
different configured image when possible.

## Start from an image

Upload a JPEG, PNG, WebP, or BMP through Reactor's upload protocol, then pass its
upload reference to `set_image`. The adapter applies EXIF orientation, validates
the decoded image, and uses the official centered aspect-ratio crop and 352x640
resize before creating the visual condition.

Uploads are limited to 25 MiB and 100 million decoded pixels. Uploaded bytes and
their rollout state are session-scoped and released when the session ends.

## Autoregressive inference

The adapter loads the official model, image VAE/CLIP encoder, causal VAE decoder,
and universal distilled checkpoint in the Runtime process. Each inference turn advances
the same native three-latent block as upstream `inference_streaming.py` and
continues its incremental state.

The active rollout preserves the diffusion model's 30 rolling KV caches, the
keyboard and mouse action KV caches, the visual cross-attention cache, and all
32 causal VAE decoder cache tensors. The official `local_attn_size: 6`,
three-step distilled denoising schedule, context-cache commit, and 360-latent
horizon remain unchanged.

The first causal decode emits 9 RGB frames and each later decode emits 12.
Playback adapts to measured inference throughput, and the output queue holds
one complete 12-frame chunk. One rollout therefore provides 120 interactive
chunks before the adapter idles and requires `reset`, `set_image`, or
`random_image`.

## Model messages

Commands and generation publish typed messages for the client timeline:

- `action_changed` contains the addressed key, whether it is held, the complete
  held-key set, and the first chunk that will sample it.
- `camera_motion_changed` contains both camera axes and the first chunk that will
  sample them.
- `state_update` is a complete snapshot of image selection, rollout progress,
  seed, and active controls. A joining viewer receives one immediately;
  successful state changes broadcast another.
- `chunk_complete` contains the chunk index, decoded frame count, sampled
  controls, and measured inference time.
- `rollout_limit_reached` reports the exhausted official chunk horizon.

Message delivery stays outside the synchronous inference path.

## Recording

`reactor.yaml` records `main_video` as H.264 in four-second chunks and allows
clips up to five minutes. The model image includes FFmpeg. Matrix-Game-2.0
does not emit audio.

## Notes

- `matrix_game_2.yaml` pins the upstream source revision, checkpoint snapshot,
  universal distilled variant, 360-latent horizon, seed, and public demo images.
- Set `MATRIX_GAME_2_PATH` to an existing `Matrix-Game-2` source directory to
  reuse a checkout. Its Git revision must match the pinned revision.
- The released Matrix-Game-2.0 checkpoint accepts image and action
  conditioning. The command schema exposes those native inputs.
- The image uses Matrix-Game-2.0's native Flash Attention 2.8.3 dependency.
- Stop `reactor run` to remove the container and release its GPU memory.
