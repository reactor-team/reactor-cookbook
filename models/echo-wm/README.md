# Echo-WM Flash

Explore a prompt-, image-, and camera-controlled audiovisual world with
[Echo-WM](https://github.com/jd-opensource/JoyAI-Echo/tree/main/echo_wm) and
Reactor.

## Runtime boundary

The adapter keeps one Echo-WM rollout alive across interactive turns. An image
anchors the first frame, a prompt conditions the world, and four pure-camera
axes control motion. Each inference turn advances the released causal model by
one three-latent block and preserves its bounded audio-video KV caches for the
next turn.

This recipe uses the four-step
[`echo-wm-flash.safetensors`](https://huggingface.co/Echo-Team/Echo-WM) release
at 1280×704 and 24 FPS. It retains Echo-WM's 19-frame local video cache,
7-frame sink, aligned audio caches, and `[1000, 750, 500, 250]` timestep
schedule. The pinned JoyAI-Echo checkout remains an unmodified upstream source
tree; the Reactor integration lives entirely in this directory.

One model output contains synchronized `main_video` and `main_audio` tracks.
The first output carries the anchor plus 24 generated frames and 50,000 mono
audio samples. Later outputs carry 24 frames and 48,000 samples, representing
one second of generated world time per chunk. Playout uses Echo-WM's native
24 FPS rate so the 48 kHz audio samples remain aligned with their video frames.

## Run with the Reactor CLI

This directory is a Reactor workspace. Its `reactor.yaml` manifest defines the
Python 3.12 serving image, Reactor Runtime 3.2.5, CUDA and system dependencies,
the model entry point, GPU resource, recording tracks, and persistent weights
directory. See the [Reactor CLI installation
guide](https://docs.reactor.inc/deploy/platform/installation) and [build
configuration guide](https://docs.reactor.inc/deploy/platform/build) for host
setup and supported manifest fields.

The default configuration targets Linux and one NVIDIA B200. Install the
NVIDIA Container Toolkit, accept the public Gemma 3 license on Hugging Face,
and expose a read token before the first asset download:

```sh
cd models/echo-wm
export HF_TOKEN=hf_your_read_token

reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 in the serving
container. `reactor run` serves on `http://localhost:8080` by default. Choose a
different port when needed:

```sh
reactor build && reactor run --gpus device=0 --port 18091 -e HF_TOKEN
```

Model startup finishes after the source and model assets are prepared, one
warmup chunk completes, and the attention backend passes its numerical check.
Use the selected port to inspect readiness and the generated model contract:

```sh
curl -fsS http://localhost:18091/health
curl -fsS http://localhost:18091/schema
```

## Demo frontend

The [`demo`](./demo) directory contains the AlayaWorld-style interactive
frontend adapted to Echo-WM's image, prompt, four-axis camera, field-of-view,
video, and audio contract. With the model serving on its default port:

```sh
cd demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), connect, then choose a
built-in scene or upload an image. See the [demo guide](./demo/README.md) for
alternate ports, keyboard controls, and hosted connection settings.

The manifest stores persistent model data under
`~/.cache/reactor_registry/echo-wm-flash`. Change `runtime.weights_path` to use
another high-capacity host volume; the source, checkpoints, Hugging Face cache,
uploaded images, and compiled kernel cache are all resolved beneath that mount.

## Public source and model assets

Startup prepares missing assets automatically and validates every pinned
revision:

- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) supplies the
  original Echo-WM model, camera conditioning, KV-cache, and codec code.
- [Echo-Team/Echo-WM](https://huggingface.co/Echo-Team/Echo-WM) supplies the
  four-step distilled checkpoint.
- [Gemma 3 12B QAT](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
  supplies the gated text encoder.

`source.revision`, checkpoint revisions, and local paths are declared in
`echo_wm.yaml`. Existing valid files are reused on later starts.

## Interaction contract

A session starts in continuous mode and waits for an image. `set_image` accepts
a JPEG, PNG, WebP, or BMP upload plus an optional prompt and seed. A blank prompt
uses the image-neutral `inference.default_upload_prompt` from `echo_wm.yaml`,
independent of every previously selected image. `random_image` selects one of
the pinned upstream causal examples with its paired prompt. Selecting an image
starts a fresh rollout without reloading model weights.

Image selection begins continuous generation from the first native chunk.

`set_camera_motion` updates Echo-WM's four native pure-camera inputs atomically:

- `forward`: backward (-1) to forward (1)
- `strafe`: left (-1) to right (1)
- `pitch`: look down (-1) to up (1)
- `yaw`: turn left (-1) to right (1)

The values are held and sampled at the next chunk boundary. `release_camera`
returns all four axes to neutral, and `set_fov` controls the horizontal field of
view from 30 through 120 degrees. A frontend can map these values to keyboard,
pointer, touch, or gamepad controls.

Echo-WM fixes its text cross-attention KV for one rollout. `set_prompt` therefore
starts a fresh rollout from the selected image so chunk one is conditioned by
the acknowledged prompt. `reset` also starts from the selected image and can
retain or replace the seed.

Every successful mutation emits a specific model message and a full
`state_update`. `chunk_completed` reports the sampled prompt and camera state,
media counts, wall latency, and CUDA timings for denoising, cache commit, video
decode, and audio decode. The rollout automatically starts fresh from the same
image after 512 chunks while retaining the prompt, seed, field of view, camera
state, and continuous playback.

## Inference performance

Unmasked transformer attention uses FlashAttention 4 on Hopper and Blackwell
GPUs. Masked text attention and UCPE camera attention use the upstream PyTorch
SDPA path. Startup compares both backends on a representative attention shape
and rejects non-finite or numerically inconsistent output.

`inference.warmup_chunks` controls full throwaway audio-video turns during model
load. The default single turn initializes the first-chunk path without adding a
long multi-chunk startup. `inference.profile_cuda` controls per-stage CUDA
timings, and `inference.attention_benchmark` controls the startup backend check.
The B200 configuration decodes each visible video window directly to avoid
spatial tile overhead. Set `inference.video_decode_tiling` to `true` when serving
on a GPU that cannot hold the full VAE decode.

The `main_video` track advertises a 24-frame buffer so Runtime applies
backpressure at the model's native chunk boundary. The video track is paced at
24 FPS alongside the 48 kHz `main_audio` track. Recording combines both tracks
into synchronized H.264/AAC clips.
