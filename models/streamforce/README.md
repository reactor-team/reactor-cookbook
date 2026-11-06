# Play StreamForce through Reactor Runtime

Run [StreamForce](https://github.com/neu-3dvi/StreamForce) ("Streaming Video Generation with
Streaming Force Control", Wang et al., 2026, [arXiv:2606.07508](https://arxiv.org/abs/2606.07508))
as an interactive Reactor backend. Start from an image and a prompt, push an object at a point or
blow a wind across the frame, and change the force while the video plays. The video streams at
480×832, 12 frames per chunk, up to 4089 frames per run.

The adapter runs the authors' distilled 4-step causal student with rolling forcing. It loads an
exact tested StreamForce revision into the weights cache and applies `streamforce.patch`, which
exposes the source's rolling window one step at a time. Driven through the same inputs and seed,
the served model produces the same frames, bit for bit, as the authors'
`inference_causal_rolling_forcing.py`. That was checked on the six sample cases the repository
ships.

## Prerequisites

- The [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation) and Docker.
- An NVIDIA GPU with the NVIDIA driver and the NVIDIA Container Toolkit. CPU inference is
  unsupported. The recipe requests one B200: serving holds about 36 GB of GPU memory, and load
  needs about 64 GB of host memory. The build compiles FlashAttention 2 for Hopper and Blackwell.
- About 100 GB of persistent storage: the 61 GB StreamForce checkpoint, the 34 GB Wan2.2-TI2V-5B
  base, and the source checkout.

## Download the checkpoint

The StreamForce v1 checkpoint (one 61,486,967,303-byte `.pt` file covering point and wind forces)
is distributed by its authors on
[Google Drive](https://drive.google.com/file/d/1ZaOEVMzOxAtcadX2c8hBF6wA7BrtUTOi/view?usp=sharing)
only, so it is the one manual step. Save it as `model.pt` in the weights directory,
`runtime.weights_path` in `reactor.yaml`, `~/.cache/reactor_registry/streamforce` by default:

```sh
mkdir -p ~/.cache/reactor_registry/streamforce
pip install gdown
gdown 1ZaOEVMzOxAtcadX2c8hBF6wA7BrtUTOi -O ~/.cache/reactor_registry/streamforce/model.pt
```

Downloading in the browser and moving the file there works too. Startup checks the file's size
and refuses another file, naming the path and the link.

Everything else is public. On first start the adapter clones the pinned source and downloads the
pinned [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) snapshot (its text
encoder, VAE and base model) into the same directory. Later starts reuse them.

## Run

```sh
cd models/streamforce
reactor validate
reactor build
reactor run --gpus device=0 --shm-size 16g
```

`--gpus device=3` selects host GPU 3. To keep the weights elsewhere, pass
`--weights /path/to/dir`, or set `runtime.weights_path`. A liveness check is at
`http://localhost:8080/health`; the model is ready when it reports `available` (about 80 s once
the assets are on disk).

Drive it from the [Reactor Sandbox](https://reactor-sandbox.vercel.app/) pointed at
`http://localhost:8080`, or from a client SDK in local mode.

## Controls

| Command | Kind | What it does |
| --- | --- | --- |
| `set_first_frame(image)` | session setup | the picture the video starts on, resized to cover 832×480 and centre-cropped |
| `set_seed(seed)` | session setup | the run's seed; `-1` draws one at `start` |
| `set_prompt(prompt)` | live control | the scene description; an empty prompt is a value; at most 512 tokens |
| `set_force(mode, strength, angle, x, y)` | live control | `point` pushes at (`x`, `y`), 0..1 over the frame from the top left; `wind` is a field over the whole frame; `strength` 0..1; `angle` in degrees counter-clockwise from +x |
| `start`, `pause`, `resume`, `reset` | lifecycle | `start` needs a first frame, a prompt and a force; `reset` clears every input |

Session setup is refused with `locked_while_running` during a run. A live control applies from
the next chunk the model starts and reaches the screen 3 chunks later: each chunk is denoised
over four steps while three later ones are already in progress. Every command answers with
`state_update`, which gives each input's value as sent and as applied, with `from_chunk`, the
first chunk generated with the applied value. `chunk_complete` precedes every chunk.
`generation_started` and `generation_complete` mark a run. A run ends on its own after 4089
frames, the length of the model's temporal position table.

The repository's six sample stills and their forces are in the downloaded source under
`assets/samples/` (see its README for the coordinate conventions). In `set_force`, `x` and `y`
are where the push lands in the output frame.

## Performance

On one B200 a step takes 0.50 s (median) for 12 frames, about 24 fps. The model was trained at
16 fps, so motion plays about 1.5× faster than trained unless you pin `fps = 16` on the
application. The first frames arrive about 1.4 s after `start`.

## Tests

```sh
python -m reactor_runtime.schema --path . --out /tmp/schema.json
PYTHONPATH=. python -m pytest tests/ -q
```

## License and attribution

StreamForce is released by its authors under the Apache License 2.0; its `wan/` directory derives
from Alibaba's [Wan2.2](https://github.com/Wan-Video/Wan2.2), also Apache-2.0. The upstream
[NOTICE](https://github.com/neu-3dvi/StreamForce/blob/57a15e95eca25bbbb32d6ac6322ba3e12fd45083/NOTICE)
lists the modifications and prior work (CausVid, Self-Forcing, Force Prompting); the source
checkout this adapter downloads carries its LICENSE and NOTICE. `streamforce.patch` adds
`pipeline/rolling_forcing_step.py` and `pipeline/force_conditions.py`, and makes
`inference_causal_rolling_forcing.py`'s pipeline a loop over them. It draws each chunk's starting
noise when the chunk starts rather than for the whole clip at once, and lets the Wan2.2 base
directory be configured. The checkpoint's terms are the authors'.
