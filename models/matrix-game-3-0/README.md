# Play Matrix-Game 3.0 through Reactor Runtime

Run the public [Matrix-Game 3.0 world model](https://github.com/SkyworkAI/Matrix-Game)
as an interactive first-person Reactor backend. Use this recipe to start from an
uploaded image or a bundled example, optionally condition the world with text,
apply native movement and camera controls, stream autoregressive video, and
record the session.

The adapter loads an exact tested Matrix-Game revision from its mounted weights
cache and calls the upstream interactive pipeline directly.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  intentionally unsupported. The distilled recipe is tested on a B200; allow
  roughly 90 GB of GPU memory for an active rollout.
- About 40 GB in Reactor's weights cache for the distilled checkpoint and
  source checkout, plus about 18 GB of Docker image space.
- Network access to GitHub and Hugging Face during the first model load.

## Run

This directory is a `reactor` workspace. `reactor.yaml` names the model and
controls its Reactor Runtime 3.2.5, CUDA 12.8.1, Python 3.12, system packages,
and Python dependencies. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Build the image, then expose one GPU to the container:

```sh
cd models/matrix-game-3-0
reactor validate
reactor build
reactor run --gpus device=0
```

`reactor build` prepares the configured model image. `reactor run` reuses that
image, building it automatically if none exists, and bind-mounts the local
weights cache into the container. The first model load clones the pinned
upstream revision and downloads the pinned distilled checkpoint subset; later
runs reuse both.

The runtime reports ready on `http://localhost:8080`. Rebuild after editing
anything baked into the image:

```sh
reactor build && reactor run --gpus device=0
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)** and `http://localhost:8080`, or point a Reactor SDK client at
that URL with `local: true`. A quick liveness check:

```sh
curl -s localhost:8080/health
```

## Controls

- `set_image(image, prompt="")` starts from an uploaded image and optional
  scene description.
- `random_image` selects another bundled image and its paired prompt.
- `set_prompt(prompt)` replaces the text condition and starts a fresh rollout
  from the selected image.
- `set_key_state(key, pressed)` holds or releases native `w`, `a`, `s`, or `d`
  movement. Perpendicular pairs produce diagonal movement.
- `set_pitch(pitch)` and `set_yaw(yaw)` set normalized camera axes in
  `[-1, 1]` for the next chunk.
- `reset(seed)` clears rollout memory, optionally changes the seed, and starts
  again from the selected image and prompt.

Keyboard and camera values remain held until changed. Changing the conditioning
source, ending a session, and reaching the rollout limit release
all controls.

## Start from an image

Ready-to-upload frames live in [`example_images`](example_images). Upload one
through the client, then call `set_image` with its upload reference:

```js
const image = await uploadFile(file);
await sendCommand("set_image", { image });
```

The prompt is optional. Omitting it or sending an empty string lets Matrix
condition on the image and controls without injecting a synthetic description:

```js
await sendCommand("set_image", {
  image,
  prompt: "A desert town under a clear blue sky.",
});
```

Image selection starts continuous playback from the first 57-frame chunk. JPEG,
PNG, WebP, and BMP uploads are accepted up to 25 MiB and 100
million pixels. Images close to Matrix-Game's training distribution give the
most reliable rollouts.

## Autoregressive rollout

One Reactor chunk is one original Matrix iteration. The first chunk contains
57 frames and each later chunk contains 40. The configured upstream limit is
12 chunks, or 497 frames total. Playback adapts to measured inference
throughput, and the output queue holds the largest complete chunk of 57 frames.

The adapter loads `MatrixGame3Pipeline` once and runs its original interactive
`generate()` method. It replaces only the blocking terminal action reader and
per-iteration video writer at runtime. The upstream loop continues to own and
reuse its causal image condition, denoised memory latents, camera-aware memory
selection, keyboard and pose history, and 34-entry streaming VAE cache.

Matrix encodes text before entering that loop. `set_prompt` therefore starts a
fresh rollout from the selected image. Movement and camera changes preserve the
active autoregressive state.

## Model messages

Commands return typed, command-correlated messages for the client timeline:

- `controls_changed` contains the originating keyboard or camera command, the
  full held control state, and the chunk where it will apply.
- `state_update` contains the selected image and prompt, seed, restart state,
  held controls, completed chunks, next chunk size, and rollout limit.
- `rollout_limit_reached` marks completion of the final native chunk.

A joining viewer receives a complete `state_update`, and successful state
changes broadcast another. Message delivery stays outside synchronous model
inference.

## Recording

`reactor.yaml` records `main_video` by default in four-second chunks and allows
clips up to five minutes. The model image includes FFmpeg. Matrix-Game 3.0
does not emit audio.

## Storage

Local CLI runs use the workspace's `weights/` cache and mount it at the same
absolute path inside the container. Clone the cookbook onto a disk with enough
capacity, or set `runtime.weights_path` locally to an existing directory on a
larger disk. The cache is excluded from Git and the image build context.

Docker image layers are independent of Reactor's weights cache. Configure the
Docker daemon's data root on a large disk when the system disk is constrained.

## Notes

- `matrix_game_3_0.yaml` pins the upstream source, distilled checkpoint,
  sampling schedule, INT8 attention projections, MG-LightVAE v2 decoder, and
  native rollout length.
- The first load downloads the distilled 5B DiT, UMT5 assets, Wan2.2
  encoder VAE, and the 0.75-pruned MG-LightVAE v2 decoder.
- The upstream model imports its attention function directly. The adapter
  routes that symbol through the adjacent upstream dispatcher so `fa_version: 0`
  consistently uses SDPA.
- Stop `reactor run` to remove the container and release its GPU memory.
