# Matrix-Game-3.5 example

Serve the public Matrix-Game-3.5 distilled first-person model through Reactor
Runtime. Matrix is conditioned on an anchor image, a text prompt, and camera
trajectories. This adapter exposes the image and prompt through Runtime commands
and expands normalized six-axis camera motion into the camera-to-world matrices
consumed by the model.

## Runtime boundary

The 5B model is loaded once in a persistent worker process. The worker is
separate because Matrix and Reactor both expose a top-level Python package named
`examples`; process isolation lets each repository retain its public imports.

One worker rollout remains alive for the Reactor session. Every request supplies
one native three-latent causal chunk, equivalent to 12 RGB camera slots, and
receives 12 decoded frames. The rollout preserves its rolling KV cache, absolute
RoPE/PRoPE timeline, generated dynamic visual context, and FrustumHandler Patch
Memory across requests. `context_chunks: 7` remains the rolling KV window; it
keeps each interactive request at one native chunk.

The anchor image initializes Matrix's causal visual state, so `set_image` starts
a fresh rollout at chunk 1 while keeping model weights loaded. A new session
waits for an image. Each successful upload starts continuous
generation. Text conditioning
is sampled per causal chunk. `set_prompt` re-encodes only the text context for
the next chunk while retaining the current camera pose, rolling KV cache,
dynamic visual context, and Patch Memory. As the rolling window advances,
chunks made under the new prompt naturally replace older cached chunks. The
worker retains one active encoded prompt across repeated prompt changes.

The adapter submits each finished chunk as one 12-frame batch. Playback adapts
to measured inference throughput, and the output queue holds one complete chunk
while camera axes are sampled again before the next expensive turn begins.

## Run with the Reactor CLI

This directory is a `reactor` workspace. The manifest names the model and its
B200 resource, and its `build` block defines the complete Python 3.12 serving
image with Reactor Runtime 3.2.5. `requirements.txt` contains the model
dependencies. The host needs the
[`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
the NVIDIA Container Toolkit, and a compatible NVIDIA GPU. Matrix requires
Linux, CUDA, and approximately 40 GB of VRAM at 704x1280. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

Build the Python 3.12 serving image, expose one GPU, and start Runtime:

```sh
cd models/matrix-game-3-5
reactor build
reactor run --gpus device=0
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container; `--gpus all` exposes every GPU. The manifest requests one B200 when
the workspace is deployed. `reactor run` reuses the configured image and builds
one on the first run when no local image exists. It serves WebRTC signaling on
`http://localhost:8080` by default. Rebuild after changing code or dependencies.
To use a different port:

```sh
reactor build && reactor run --gpus device=0 --port 18011
```

The model repositories are public. If Hugging Face requests authentication,
forward a host token while keeping its value out of the container command line:

```sh
export HF_TOKEN=hf_your_read_token
reactor run --gpus device=0 -e HF_TOKEN
```

Startup waits for the Matrix worker to prepare its assets and load the weights.
Check readiness using the port passed to `reactor run`:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
```

A Reactor client consumes the `main_video` WebRTC track. Recording is enabled
for that video track.

## Public source and model assets

The CLI bind-mounts `runtime.weights_path` from `reactor.yaml` and exports that
directory as `REACTOR_WEIGHTS_PATH`. The checked-in default stores all large
resources under `~/.cache/reactor_registry/matrix-game-3-5`, outside both the
image and this Git checkout, so container rebuilds retain them.

On first load the adapter performs the complete public setup:

- clone the pinned Matrix-Game-3.5 source revision;
- apply the included resumable-rollout patch;
- install a managed Python 3.10 worker environment with the CUDA 12.8 PyTorch
  wheels and upstream dependencies;
- download the pinned distilled checkpoint, Wan2.2 TI2V 5B model and UMT5
  tokenizer, and Depth-Anything-3 model.

The patch adds the resumable chunk boundary used by the worker. The upstream
model forward pass, scheduler, cache policy, memory queries, and registration
logic remain in Matrix. Later starts verify and reuse the checkout, environment,
and completed snapshots. Interrupted Hugging Face downloads resume on the next
start. Startup reports a clear revision mismatch for any other checkout.

Allow roughly 48 GB for checkpoints, plus the worker environment and download
cache. `source.path` in `matrix_game_3_5.yaml` is relative to the CLI weights
mount; every worker, model, tokenizer, inference, and sample path is derived from
that one root. To relocate everything, change only `runtime.weights_path` in
`reactor.yaml`.

The adapter provides a scene-neutral generic default prompt, so users can leave
the prompt empty. The default intrinsics and camera pose arrive in the pinned
source checkout. The demo image is copied into `example_images` for convenient
manual upload. Every new session waits for the client to upload its anchor
image.

## Controls

- `set_image` accepts uploaded JPEG, PNG, WebP, or BMP bytes plus an optional
  prompt, then starts a fresh continuous rollout and generates its
  first 12-frame chunk. An empty prompt uses the generic default from
  `matrix_game_3_5.yaml`.
- `set_prompt` applies a new non-empty text condition at the next chunk boundary
  without resetting visual history. `StateUpdate.next_chunk` identifies that
  boundary.

The camera API matches AlayaWorld's normalized six-axis convention. Every value
is in `[-1, 1]`, where zero is neutral:

- `set_forward`: backward to forward translation
- `set_strafe`: left to right translation
- `set_vertical`: down to up translation
- `set_pitch`: down to up pitch
- `set_yaw`: left to right yaw
- `set_roll`: counterclockwise to clockwise roll

The frontend owns device mapping. A WASD frontend sends `set_forward(1)` for W,
`set_forward(-1)` for S, `set_strafe(-1)` for A, and `set_strafe(1)` for D. On
key release it recomputes that axis from the currently held keys and sends the
new value. Pointer, arrow, touch, and gamepad input can drive yaw and pitch;
vertical and roll can use additional keys or analog controls. Simultaneous
translation and rotation axes are normalized independently.

Every command returns `StateUpdate`, a complete snapshot containing the prompt,
anchor image, seed, six axes, completed chunk count, next control boundary,
and configured limit. A joining viewer receives the same
snapshot, and the model broadcasts another after every completed chunk. Clients
can consume each complete snapshot directly.

The `reset` command restores the selected anchor and prompt; an optional
non-negative `seed` selects the next reproducible rollout.

Camera axes are sampled at a chunk boundary and apply to the next 12 camera slots,
or 0.75 seconds at 16 FPS. Commands received during inference or playback affect
the following chunk. An in-flight CUDA chunk cannot be interrupted; reset and
disconnect take effect when inference returns to Runtime.

`stream.max_chunks` bounds the preallocated PRoPE camera timeline. The default
512 chunks cover 6.4 minutes. After the final chunk, generation stops and emits
`RolloutLimitReached` followed by a `StateUpdate` with `limit_reached: true` and
`next_chunk: null`. Camera commands then return `rollout_limit_reached` until
`reset` or `set_image` starts a fresh timeline.
Reset releases the prior KV and memory state while preserving loaded weights.

Ending a session also releases its rollout caches and request workspace while
keeping model weights resident for the next session. A client disconnect within
a live session releases held camera axes but preserves the shared world.

## Image uploads

The `set_image` command declares an `UploadedFile` parameter in Runtime's
schema. A Reactor client reserves a session upload slot, writes the raw bytes to
its returned URL, and sends the resulting upload reference with the command. A
schema-driven frontend can render a file picker while the upload channel carries
the binary data.

Each session waits for `set_image`. A successful upload starts continuous
generation and broadcasts completion after each 12-frame chunk.

A ready-to-upload copy of the public first-person demo input lives in
[`example_images`](example_images).

Uploads are limited to 25 MiB and 100 million pixels. Runtime verifies the
declared media type, actual JPEG/PNG/WebP/BMP codec, dimensions, and decodability
before a rollout reset. An empty `set_image` prompt keeps the current prompt;
an empty `set_prompt` is rejected because Matrix requires text conditioning.
