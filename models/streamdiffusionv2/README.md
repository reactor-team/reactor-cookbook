# Run StreamDiffusionV2 through Reactor Runtime

Restyle a live video from a text prompt in real time with
[StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2)
([paper](https://arxiv.org/abs/2511.07399)), the 1.3B video-to-video model: a causal Wan2.1 DiT
distilled with DMD, two denoising steps batched across chunks ("stream batch"), and a rolling KV
cache with sink tokens. Send video on `video_in` (a webcam, or a file played in a page), set a
prompt, and every step rewrites the next 4 frames at 480x832. The prompt can change while it plays.

The adapter is a `ReactorApp` in two halves: `streamdiffusionv2.py` (application) and
`streamdiffusionv2_model.py` (model, no runtime import). The model half calls the upstream's own
single-GPU stream pipeline (`streamv2v/inference.py`), one iteration of its per-chunk loop per step.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and Docker.
- An NVIDIA GPU with the NVIDIA Container Toolkit; the manifest declares one B200. The model holds
  about 27 GB of GPU memory while serving.
- About 24 GB in the weights directory: the Wan2.1-T2V-1.3B snapshot (text encoder, VAE, base DiT)
  and the 5.7 GB StreamDiffusionV2 checkpoint.

## Run

```sh
cd models/streamdiffusionv2
reactor validate
reactor build
reactor run --gpus device=0 --shm-size 8g
```

On first start, `streamdiffusionv2_assets.py` fills `runtime.weights_path` once, at pinned
revisions (`streamdiffusionv2.yaml`):

| What | From | Lands at |
|---|---|---|
| upstream source | `github.com/chenfengxu714/StreamDiffusionV2` @ `6961a5c` | `StreamDiffusionV2/` |
| text encoder, VAE, base DiT | `Wan-AI/Wan2.1-T2V-1.3B` | `wan_models/Wan2.1-T2V-1.3B/` |
| the student checkpoint | `jerryfeng/StreamDiffusionV2`, `wan_causal_dmd_v2v/` | `ckpts/wan_causal_dmd_v2v/` |

Startup refuses a `model.pt` of any other size than the tested 5,676,282,858 bytes. After load it
runs 9 throwaway chunks so the first session pays no warm-up. Startup measured on a B200: 82 s with
the weights present, 124 s from an empty weights directory (downloads included).

## Inputs

| Input | Modality | Kind | Command |
|---|---|---|---|
| video to rewrite | video | live control | `video_in` track; resized (stretched) to 832x480; resampled to 16 fps by capture time |
| prompt | text | live control: applies at the next chunk, the motion cache is kept | `set_prompt` |
| seed | number | session setup: set while idle, `-1` for random | `set_seed` |
| starting noise strength, 0..1 (upstream example: 0.8) | number | session setup; the model adapts it every chunk to motion, as upstream does | `set_noise_scale` |

Serving config, the same for every session: 2 denoising steps (timesteps 700, 500), 480x832,
bf16, the full Wan2.1 VAE.

## Commands and messages

`set_prompt`, `set_seed`, `set_noise_scale`, `start`, `pause`, `resume`, `reset`. Every command
answers with `state_update`. A refused command raises with one of `missing_prompt`, `missing_seed`,
`missing_noise_scale`, `already_running`, `not_running`, `not_paused`, `locked_while_running`,
`prompt_too_long`, `invalid_seed`, `invalid_noise_scale`.

`chunk_complete` precedes every chunk (`index`, `frames_out`, `generate_seconds`, `noise_scale`);
`generation_started` follows a run's first chunk; `generation_complete` follows a `reset` that ended
a run. Chunk 0 emits 5 frames, chunk 1 none (the stream batch fills), every later chunk 4: the
output trails the input by one chunk.

## Measured

On one B200, against the upstream's own README example (`examples/original.mp4`, seed 0, noise 0.8):

- the port reproduces the upstream bit for bit: `max|diff| = 0` on all 19 chunks with real input,
  through a served stepped session of this folder started from an empty weights directory, in two
  back-to-back sessions;
- 0.199 s median per 4-frame chunk (about 20 fps of capacity) against the trained 16 fps; output is
  paced at 16 fps, delivered 16.0 fps with a longest gap of 93 ms;
- GPU memory flat at 27.2 GB over 20 back-to-back sessions.

## Tests

```sh
python -m reactor_runtime.schema --path . --out /tmp/schema.json
PYTHONPATH=. python -m pytest tests/ -q
```

## Licence

StreamDiffusionV2's code and the `jerryfeng/StreamDiffusionV2` and `Wan-AI/Wan2.1-T2V-1.3B`
weights are Apache-2.0. This adapter is Apache-2.0, as the rest of this repository.
