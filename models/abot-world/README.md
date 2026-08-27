# ABot-World

This adapter turns the distilled `ABot-World-0-5B-LF` causal video model into a
playable Reactor session. Start from an uploaded image or a built-in example,
change the scene prompt between chunks, and use the model's native W/A/S/D
movement and I/J/K/L view controls. Each inference turn preserves the upstream
rolling KV cache and emits one decoded autoregressive chunk.

ABot-World is an image-to-video world model. It generates three latent frames
per chunk, decoded to 9 RGB frames for the first chunk and 12 RGB frames for
each later chunk at 1280×704. Playback adapts to measured inference throughput,
and the output queue holds one complete 12-frame chunk. The adapter retains the
upstream 21-latent local-attention window.

## Run locally

Install the [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation)
and Docker with the NVIDIA Container Toolkit, then run the model from this
directory. At native resolution the distilled model uses about 23 GB of VRAM;
the deployment manifest requests a B200 and allows up to 32 GiB of host memory.

```sh
reactor build
reactor run --gpus device=0 --port 8080
```

The `build:` block in `reactor.yaml` configures CUDA 12.8, Python 3.12,
FlashAttention 2.8.1, Git, and FFmpeg. See Reactor's
[build configuration](https://docs.reactor.inc/deploy/platform/build) for the
supported fields. Run `reactor build` again after changing model code,
requirements, or build configuration.

The first container start clones the pinned public
[ABot-World](https://github.com/amap-cvlab/ABot-World) revision and downloads
the pinned
[ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF)
snapshot below `runtime.weights_path`. Point that field at a directory on
high-capacity storage before running when the default cache is too small. The
CLI bind-mounts the directory into the container, so source and checkpoint
downloads persist across image rebuilds. A liveness check is available at
`http://localhost:8080/health`.

## Controls

- `set_image` uploads a JPEG, PNG, WebP, or BMP first frame and starts a fresh continuous world.
- `random_image` selects one of the bundled upstream example scenes and starts continuous generation.
- `set_prompt` applies new text conditioning at the next chunk without resetting KV cache.
- `set_key_state` holds or releases W/A/S/D movement and I/J/K/L view keys. Combined keys support diagonal movement and simultaneous view changes; short taps survive until the next chunk sample.
- `release_controls` returns every action channel to neutral.
- `reset` restarts the selected image with the current prompt and an optional new seed.

Selecting an uploaded or built-in image starts continuous generation from
chunk 1.

The public messages report the accepted action, the chunk it will affect, the
last sampled keys, prompt application, rollout progress, and reset or limit
state. The adapter stops after 512 chunks so abandoned sessions cannot consume
the GPU indefinitely; reset or select an image to begin a fresh rollout.

## Upstream fidelity

The adapter uses a pinned upstream checkout and calls its prompt encoder,
first-frame encoder, eight-channel action encoder, denoising loop, rolling KV
cache update, and cached VAE decoder directly. Inference uses the released
distilled checkpoint with bfloat16 execution and FlashAttention 2.8.1.
