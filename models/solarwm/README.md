# Play SolarWM 5B Stage2 through Reactor Runtime

Run the public [SolarWM](https://github.com/Junchao-cs/SolarWM) Wan2.2
TI2V-5B Stage2 checkpoint as an interactive Reactor backend. A client uploads
an anchor image, optionally supplies a prompt, applies six-axis camera motion,
and streams generated video.

The adapter loads an exact tested SolarWM revision and preserves the upstream
autoregressive sampler, camera conditioning, and rolling cache boundary.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and
  Docker.
- An NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit. CPU inference is
  unsupported. A tested rollout used about 58 GB of VRAM, so a GPU with at
  least 64 GB is recommended.
- Access approval for the gated
  [`junchaoh-cs/SolarWM`](https://huggingface.co/junchaoh-cs/SolarWM) repository.
- About 75 GB of persistent storage for the Wan2.2-5B base assets, Stage2
  checkpoint, source checkout, and runtime files, plus space for the model
  image and build cache.

## Run

This directory is a `reactor` workspace. Its `reactor/v2` manifest defines the
model, persistent weights mount, and complete CUDA-capable image
build. No Dockerfile is required. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported YAML fields.

Validate the workspace, build the model image, and expose one GPU to the
container:

```sh
cd models/solarwm
reactor validate
reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 inside the
container. `reactor run` reuses the image from `reactor build` and builds it
automatically when the local tag is missing. Rebuild after changing adapter
code, dependencies, or the manifest.

First startup prepares the pinned source and downloads the model assets. Later
starts reuse `/opt/dlami/nvme/.cache_hf`. Forward a Hugging Face read token
without putting its value on the command line:

```sh
export HF_TOKEN=hf_your_read_token
reactor run --gpus device=0 -e HF_TOKEN
```

The default endpoint is `http://localhost:8080`. Pass `--port` to use another
port:

```sh
reactor run --gpus device=0 --port 18087 -e HF_TOKEN
```

Connect from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) using
**Local (Direct)**, or check readiness and the generated contract directly:

```sh
curl -s localhost:8080/health
curl -s localhost:8080/schema
```

## Controls

Generation waits for `set_image`; there is no default or random image.

- `set_image(image, prompt)` starts a fresh world from an uploaded image. The
  prompt is optional; an empty value preserves the active prompt or uses the
  default configured in `solarwm.yaml` when none is active.
- `set_prompt(prompt)` starts a fresh world from the selected image with a
  non-empty text condition.
- `set_forward`, `set_strafe`, and `set_vertical` control translation.
- `set_pitch`, `set_yaw`, and `set_roll` control rotation.
- `release_camera` returns every held camera axis to neutral.
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
fresh world and continuous generation from its first chunk. There is no
configured default image, so generation remains idle until the client uploads
one.

SolarWM resizes and center-crops the upload to its native 864×480 resolution.
Images resembling the model's training distribution give the most stable
interactive rollouts.

## Runtime boundary

SolarWM Stage2 is an autoregressive video model. The included backend runs the
upstream four-step self-forcing sampler as a resumable session: model weights
load once, and self-KV, absolute position, prompt conditioning, random
generator, and causal video decoder state remain alive for the next request.

The first native three-latent chunk decodes to 9 RGB frames because it includes
the image anchor. Every later chunk decodes to 12 frames. Each turn is
submitted as one frame batch; playback adapts to measured inference throughput,
and the output queue holds one complete 12-frame chunk. No fixed Reactor FPS is
declared.

SolarWM fixes prompt conditioning for one rollout. `set_prompt` therefore
starts again from the selected image, so chunk one uses the acknowledged prompt
without mixing it with the previous world's cached context.

## Model messages

Every accepted command returns its own typed result — `image_selected`,
`prompt_queued`, `camera_motion_changed`, or `rollout_reset_queued` — and
broadcasts a complete `state_update` snapshot for the client timeline. The
snapshot contains the active image, prompt, seed, all six camera axes,
completed chunk count, next affected chunk, frame count, rollout limit, and
most recent generation time. A joining viewer receives the same state
immediately.

`rollout_limit_reached` reports when generation stops at the configured safe
timeline boundary. `reset` or `set_image` starts a fresh timeline. Message
delivery remains outside the synchronous inference loop.

## Public source and model assets

`solarwm.yaml` pins the SolarWM source revision, Wan2.2-5B base assets, and
Stage2 checkpoint. First load downloads missing files from Hugging Face and
validates the source revision. Existing valid files are reused on later
starts.

The CLI bind-mounts `runtime.weights_path` at the same absolute path inside the
container and exports it as `REACTOR_WEIGHTS_PATH`. This recipe uses
`/opt/dlami/nvme/.cache_hf` so the SolarWM checkout can reuse shared Wan2.2
assets without copying them onto the system disk. `reactor.yaml` also directs
UV, Hugging Face, CUDA, and general runtime caches to the NVMe volume. Docker's
image layers and build cache must be configured separately on the host; this
server uses `/opt/dlami/nvme/.cache_docker` as Docker's data root.

## Notes

- `stream.context_latents: 18` preserves the upstream rolling self-KV window;
  each inference turn requests one native three-latent chunk.
- `stream.max_chunks: 320` bounds the absolute camera timeline. A fresh image or
  reset starts another timeline without reloading the checkpoint.
- Ending a session releases rollout, camera, and video decoder caches while
  retaining loaded model weights for the next session.
- Stop `reactor run` to remove the container and release its GPU memory.
