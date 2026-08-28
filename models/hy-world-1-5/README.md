# HY-World 1.5 example

Serve the public
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) distilled
autoregressive image-to-video model through Reactor Runtime. The adapter keeps
HY-World's upstream denoising, text and visual KV caches, dual action
representation, causal latent history, and Reconstituted Context Memory intact
while exposing image, prompt, and camera controls to Reactor
clients.

HY-World predicts four causal latents per turn at 480p. The first turn decodes
to 13 RGB frames because its first latent is the image anchor; every later turn
decodes to 16 frames. Playback on `main_video` adapts to measured inference
throughput, with one complete model turn in the 16-frame output queue.

A new session starts without choosing a scene for the user. Upload
an image with `set_image`, or invoke `random_image` to select an official
example. Either command starts continuous generation from the first chunk.

## Run

This directory is a `reactor` workspace. Its `reactor.yaml` declares the model,
runtime entry point, Reactor Runtime 3.2.5, CUDA and Python versions, system
packages, and Python requirements. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

The host needs the
[`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
the NVIDIA Container Toolkit, and a compatible NVIDIA GPU. From the recipe
directory, validate the manifest, build the image, and expose one GPU to the
container:

```sh
cd models/hy-world-1-5
reactor validate
reactor build
reactor run --gpus device=0 --port 18089
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container; `--gpus all` exposes every GPU. `reactor run` reuses the image from
`reactor build`, and builds it automatically when the local image is absent.
Rebuild after changing adapter code, dependencies, or the manifest.

Check readiness after the model finishes downloading and loading. Use the port
passed to `reactor run`:

```sh
curl -s localhost:18089/health
curl -s localhost:18089/schema
```

The preferred SigLIP image encoder comes from the gated public
`black-forest-labs/FLUX.1-Redux-dev` repository. After accepting its terms, a
Hugging Face read token can be forwarded without putting its value on the
Docker command line:

```sh
export HF_TOKEN=hf_your_read_token
reactor run --gpus device=0 --port 18089 -e HF_TOKEN
```

When that gated asset is unavailable, startup prepares the matching public
SigLIP SO400M architecture from Google and keeps image-semantic conditioning
enabled.

For a hosted release, choose a unique `model.name`, bump `model.version`, and
review the resources in `reactor.yaml`. The publish command uses the same
manifest:

```sh
reactor model publish
reactor model deploy
```

## Play it in the browser

The [Reactor Sandbox](https://reactor-sandbox.vercel.app/) can connect to the
local endpoint with **Local (Direct)**. The generated controls include a real
file picker for `set_image`, the built-in `random_image` action, prompt and
camera inputs, and reset. The model-message timeline reports every
accepted state change and completed chunk.

## Public source and model assets

The CLI bind-mounts `runtime.weights_path` from `reactor.yaml` and exports that
directory as `REACTOR_WEIGHTS_PATH`. The checked-in default keeps all large
resources under `~/.cache/reactor_registry/hy-world-1-5`, outside both the image
and this Git checkout, so container rebuilds retain them.

On first load the adapter performs the complete public setup:

- clone the pinned HY-WorldPlay source revision;
- download the HunyuanVideo 1.5 480p transformer, VAE, and scheduler;
- download the distilled action checkpoint, Qwen and ByT5 text encoders, Glyph
  encoder, and SigLIP vision encoder;
- assemble the relative model layout expected by the upstream pipeline.

Interrupted Hugging Face downloads resume on the next start. Later launches
verify and reuse the prepared resources. The official image and caption pairs
used by `random_image` arrive in the pinned source checkout, so no separate
sample dataset is required.

Allow room for the download cache, base model, encoders, and approximately 34
GB distilled checkpoint. To keep the stable CLI path while placing its contents
on a larger volume, create a symlink before the first launch:

```sh
export HY_WORLD_WORK=/path/to/large-volume/hy-world-1-5
mkdir -p "$HY_WORLD_WORK/weights" "$HOME/.cache/reactor_registry"
ln -s "$HY_WORLD_WORK/weights" \
  "$HOME/.cache/reactor_registry/hy-world-1-5"
```

The container engine stores image layers and build cache separately. Configure
that storage on a large volume when the system disk has limited space.

To reuse an existing public checkout or prepared asset, place it somewhere
visible inside the container and forward one or more of these environment
variables through `reactor run -e NAME`. Local `reactor run` mounts only
`runtime.weights_path`, so keeping reused assets beneath that root gives the
host and container the same absolute path:

```sh
export HY_WORLDPLAY_PATH=/path/to/HY-WorldPlay
export HY_WORLDPLAY_BASE_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_WORLDPLAY_ACTION_MODEL_PATH=/path/to/HY-WorldPlay-checkpoints
export HY_WORLDPLAY_VISION_ENCODER_PATH=/path/to/siglip-layout
```

`HY_WORLDPLAY_VISION_ENCODER_PATH` must contain `image_encoder/` and
`feature_extractor/`. The action model path must contain
`ar_distilled_action_model/model.safetensors`. Existing source checkouts must
match the immutable revision in `hy_world_1_5.yaml`; startup reports a clear
revision mismatch for any other checkout.

## Controls

HY-World's public first-person action path trains four controls. `set_camera`
updates all four atomically, and every value is normalized to `[-1, 1]`:

- `forward`: backward to forward translation
- `strafe`: left to right translation
- `pitch`: look down to look up
- `yaw`: turn left to turn right

The model does not train vertical translation or camera roll on this path, so
the schema does not invent those controls. The frontend owns keyboard, pointer,
touch, and gamepad mapping; a WASD frontend can map W/S to `forward` and A/D to
`strafe`, while pointer or arrow input drives `pitch` and `yaw`.

Additional commands:

- `set_image` validates an uploaded JPEG, PNG, WebP, or BMP, starts a fresh
  world, and queues the visible initial chunk.
- `random_image` selects a different official image and matching caption.
- `set_prompt` replaces text conditioning at the next chunk boundary without
  discarding generated latent or geometric memory.
- `release_camera` returns all four camera values to zero.
- `reset` rebuilds the world from the selected image and prompt while
  resuming continuous generation; its optional seed selects a reproducible rollout.

Commands received during an in-flight CUDA turn apply at the next chunk
boundary. Every successful control returns a typed confirmation and broadcasts
a complete `StateUpdate`. `ChunkCompleted` records the prompt, camera values,
frame count, and generation time used by each finished chunk.

## Image uploads

The `set_image` command declares an `UploadedFile` parameter in Runtime's
schema. A Reactor client reserves a session upload slot, writes the raw bytes to
its returned URL, and sends the resulting upload reference with the command.
This lets schema-driven clients render a file picker without embedding binary
data in a command message.

Uploads are limited to 25 MiB and 100 million decoded pixels. The adapter
verifies the declared media type, actual JPEG/PNG/WebP/BMP codec, dimensions,
and decodability before starting a fresh rollout. EXIF orientation is applied
before the upstream image-conditioning path prepares its 832x480 input.

## Autoregressive memory and long sessions

After the initial image-conditioned chunk, the backend selects the official
20-frame context for every turn: the recent 12 latent frames plus
geometry-aligned historical memory, including the initial chunk. The distilled
four-step denoiser and official memory settings are fixed and validated because
changing them would alter memory evaluation.

Inference uses the upstream PyTorch attention path and eager execution from the
pinned model source.

`stream.max_chunks` defaults to 512. At the limit, generation idles and emits
`RolloutLimitReached`; `reset`, `set_image`, or `random_image` starts a fresh
world without reloading model weights. Ending a session releases its causal KV
cache, VAE cache, latent history, camera history, and geometric memory while
keeping the loaded checkpoint resident for the next session.

## Upstream license

HY-WorldPlay source, weights, and generated outputs remain governed by the
[Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/LICENSE).
Review its terms before downloading or serving the model.
