# HY-World 1.5 example

The adapter uses Reactor Runtime 3.10's native step loop. `hy_world_1_5.py`
owns commands and camera planning; `hy_world_1_5_model.py` owns inference and
causal state. Frozen inputs and results carry one native chunk per step, with
a world ID acknowledging when a new reference has been applied.

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
runtime entry point, Reactor Runtime 3.10.0, CUDA and Python versions, system
packages, and Python requirements. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields.

The host needs the
[`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
the NVIDIA Container Toolkit, and a compatible NVIDIA GPU. First build the
weights folder. It takes about 95 GB, every source is public and ungated, and
no token is needed:

```sh
cd models/hy-world-1-5
pip install huggingface-hub modelscope pyyaml
python prepare_weights.py --output ./weights
```

Then validate the manifest, build the image, and expose one GPU to the
container. `runtime.weights_path` points `reactor run` at `./weights`:

```sh
reactor validate
reactor build
reactor run --gpus device=0 --port 18089
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container; `--gpus all` exposes every GPU. `reactor run` reuses the image from
`reactor build`, and builds it automatically when the local image is absent.
Rebuild after changing adapter code, dependencies, or the manifest.

Check readiness after the model finishes loading. Use the port passed to
`reactor run`:

```sh
curl -s localhost:18089/health
curl -s localhost:18089/schema
```

To deploy, choose a unique `model.name`, bump `model.version`, review the
resources in `reactor.yaml`, and push the same folder as the weight set, so a
new instance does not download the checkpoints:

```sh
reactor deploy production --weights ./weights
```

## Play it in the browser

The [Reactor Sandbox](https://reactor-sandbox.vercel.app/) can connect to the
local endpoint with **Local (Direct)**. The generated controls include a real
file picker for `set_image`, the built-in `random_image` action, prompt and
camera inputs, and reset. The model-message timeline reports every
accepted state change and completed chunk.

## Public source and model assets

`reactor build` clones the pinned HY-WorldPlay source into the image at
`/opt/HY-WorldPlay`; `HY_WORLDPLAY_PATH` selects another checkout, which must
be at the same revision. The official image and caption pairs that
`random_image` uses come with that checkout, so no separate sample dataset is
required.

`prepare_weights.py` writes each checkpoint as a plain file at the path the
upstream pipeline reads. `HunyuanVideo-1.5/` holds the 480p transformer, VAE,
and scheduler, with the encoders in its `text_encoder/` and `vision_encoder/`
subfolders. `HY-WorldPlay/` holds the distilled action checkpoint. The script
skips the files of these repositories that the model does not read, and a
second run downloads only the files that are missing.

| Source | Revision | Size |
| --- | --- | --- |
| [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) source (in the image) | `1588e1336e842b03b0a7860c654ebd7c46bb065e` | 35 MB |
| `tencent/HunyuanVideo-1.5` (480p transformer, VAE, scheduler) | `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038` | 38.4 GB |
| `tencent/HY-WorldPlay` (distilled action checkpoint) | `f4c29235647707b571479a69b569e4166f9f5bf8` | 34.2 GB |
| `Qwen/Qwen2.5-VL-7B-Instruct` (text encoder) | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 16.6 GB |
| `google/siglip-so400m-patch14-384` (image encoder) | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` | 3.5 GB |
| `google/byt5-small` (glyph text encoder) | `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` | 1.2 GB |
| ModelScope `AI-ModelScope/Glyph-SDXL-v2` (glyph checkpoint) | `b51a8604c03c9a4e71a389f4c7deeb526320f2e6` | 0.9 GB |

Upstream reads its SigLIP image encoder from the gated
`black-forest-labs/FLUX.1-Redux-dev` repository. Those tensors are the vision
tower of the public Google SigLIP SO400M checkpoint, rounded to bfloat16, so
the weights folder carries the public checkpoint and keeps image-semantic
conditioning without a token.

Nothing downloads at runtime (`HF_HUB_OFFLINE=1`), and the weights folder is
read-only. A missing file stops `load()` with an error that names the file and
`prepare_weights.py`. A source checkout at another revision stops `load()` with
a request to rebuild the image. The model writes its own state under
`~/.cache/hy-world-1-5`.

The container engine stores image layers and build cache separately. Configure
that storage on a large volume when the system disk has limited space.

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

## Finite camera paths

After selecting an image, `set_camera_path` accepts a JSON upload containing
`viewmats` (N × 4 × 4 world-to-camera matrices), `intrinsics` (N × 3 × 3 normalized
camera matrices), and `actions` (N native integer labels in 0–80). Each row is
one latent time step; N must be divisible by four. The path queues a fresh world
and produces N / 4 chunks: 13 frames initially, then 16 per chunk.
`reset` replays the path; selecting an image restores live camera controls.
The `camera_path_selected` response reports the exact chunk and frame budget.

A camera path is sampled the way upstream's offline `pipe(...)` call (and
WBench's `examples/hy_worldplay/generate_navigation.py`) samples the same
video: path values stay float64 for memory selection, the noise for all N
latents is drawn in one call from the seeded CUDA generator, and frames are
truncated to uint8 like upstream `save_video`. Live controls draw noise one
chunk at a time because their length is unknown. Like upstream, the
geometric-memory point cloud comes from the unseeded global CPU generator, so
memory selection after the first 20 latents can vary between processes.

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
