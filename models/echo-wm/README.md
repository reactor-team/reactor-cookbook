# Echo-WM Flash

Explore a prompt-, image-, and camera-controlled audiovisual world with
[Echo-WM](https://github.com/jd-opensource/JoyAI-Echo/tree/main/echo_wm) and
Reactor.

## Runtime boundary

### Finite camera paths

After `set_image`, upload a JSON file through `set_camera_path(trajectory)`.
The `poses` array contains camera-to-world matrices at 24 FPS with shape
`[24 * chunks + 1, 4, 4]`. The first pose anchors the selected image.
The reply reports the chunk count and total frames: 25 frames in the first
chunk, then 24 per chunk. Generation stops at the path end; `reset` replays it.
Selecting another image restores held camera controls. `set_fov` before the
path's first chunk sets its field of view, which then stays fixed until `reset`.

A camera path is the same input as upstream `inference_wm_causal.py` with an
Action DSL: the model anchors and calibrates the per-frame poses with
upstream's own CUDA arithmetic, and decodes the video exactly as upstream
decodes a finished clip (one tiled VAE decode with `TilingConfig.default()`).
Each tile is decoded once every latent it reads exists, so a step generates up
to about five native blocks ahead of the frames it returns; later steps return
frames already generated. Held camera controls cannot know their future poses
and keep decoding each visible window directly.

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

The adapter uses Reactor Runtime 3.10's native step loop. `echo_wm.py` handles
client commands, plans camera poses, and publishes each chunk. `echo_wm_model.py`
owns model loading and generation through typed inputs and results, with no
Reactor Runtime dependency. A world identifier associates each output with its
selected image and prompt; the anchor travels only until that world is confirmed.

This directory is a Reactor workspace. Its `reactor.yaml` manifest defines the
Python 3.12 serving image, Reactor Runtime 3.11.3, CUDA and system dependencies,
the pinned upstream source, the model entry point, GPU resource, recording
tracks, and weights folder. See the [Reactor CLI installation
guide](https://docs.reactor.inc/deploy/platform/installation) and [build
configuration guide](https://docs.reactor.inc/deploy/platform/build) for host
setup and supported manifest fields.

The default configuration targets Linux and one NVIDIA B200 with the NVIDIA
Container Toolkit. Gemma 3 is gated: accept its terms on the
[model card](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
with your Hugging Face account. Then build the weights folder once, with a
token that can read it. It downloads about 72 GB at the pinned revisions:

```sh
cd models/echo-wm
pip install huggingface-hub pyyaml
HF_TOKEN=<your token> python prepare_weights.py --output ./weights
```

Then build, run, and deploy. `runtime.weights_path` points `reactor run` at the
same folder, and the deploy uploads it, so a new instance downloads nothing:

```sh
reactor build
reactor run --gpus device=0
reactor deploy production --weights ./weights
```

`--gpus device=3` selects host GPU 3 and presents it as device 0 in the serving
container. `reactor run` serves on `http://localhost:8080` by default. Choose a
different port when needed:

```sh
reactor build && reactor run --gpus device=0 --port 18091
```

Nothing downloads at runtime (`HF_HUB_OFFLINE=1`). A missing checkpoint stops
`load()` with an error that names the file and the command that builds the
folder. Model startup finishes after the source revision and weights are
checked, one warmup chunk completes, and the attention backend passes its
numerical check.
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

The weights folder is read-only at runtime. Uploaded images and the compiled
kernel caches go to `~/.cache/echo-wm` (`assets.runtime_dir` in
`echo_wm.yaml`). To keep the weights on another high-capacity volume, pass that
folder to `prepare_weights.py --output` and to `reactor run --weights` and
`reactor deploy production --weights`.

## Public source and model assets

Every source is pinned by full commit in `echo_wm.yaml`, and the image build
and `prepare_weights.py` fetch those commits:

- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) at
  `c7963031c4425a154a791a46606d3247b2201c2b` supplies the original Echo-WM
  model, camera conditioning, KV-cache, and codec code. The image build clones
  it into `/opt/JoyAI-Echo`, and startup checks its revision.
- [Echo-Team/Echo-WM](https://huggingface.co/Echo-Team/Echo-WM) at
  `5f18c27bcc6e9395c120c055c92e294674e7ea91` supplies the four-step distilled
  checkpoint, `echo-wm/echo-wm-flash.safetensors` in the weights folder
  (48 GB). The folder skips the repository's base checkpoint.
- [Gemma 3 12B QAT](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
  at `68f7ee4fbd59087436ada77ed2d62f373fdd4482` supplies the gated text
  encoder, tokenizer, and processor, `gemma-3-12b-it-qat-q4_0-unquantized/` in
  the weights folder (24 GB).

`prepare_weights.py` is safe to run again; it downloads only the missing files.

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
decode, and audio decode. The rollout stops after 512 chunks and preserves its
final state. Use `reset` or select another image to start another rollout.

## Inference performance

Transformer attention uses upstream's PyTorch SDPA path by default, which
reproduces upstream inference in an environment without xformers. Set
`inference.attention_backend` to `flash_attention_4` for faster unmasked
attention on Hopper and Blackwell GPUs; output then differs numerically from
upstream. Masked text attention and UCPE camera attention always use SDPA.
With FlashAttention 4, startup compares both backends on a representative
attention shape and rejects non-finite or numerically inconsistent output.

`inference.warmup_chunks` controls full throwaway audio-video turns during model
load. The default single turn initializes the first-chunk path without adding a
long multi-chunk startup. `inference.profile_cuda` controls per-stage CUDA
timings, and `inference.attention_benchmark` controls the startup backend check.
For held controls, the B200 configuration decodes each visible video window
directly to avoid spatial tile overhead. Set `inference.video_decode_tiling` to `true` when serving
on a GPU that cannot hold the full VAE decode.

The `main_video` track advertises a 24-frame buffer so Runtime applies
backpressure at the model's native chunk boundary. The video track is paced at
24 FPS alongside the 48 kHz `main_audio` track. Recording combines both tracks
into synchronized H.264/AAC clips.
