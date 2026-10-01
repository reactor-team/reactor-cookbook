# Play Zing 0.5 through Reactor

Explore a generated world with [Zing 0.5](https://github.com/seedleap/zing-world-model).
Start from text, an uploaded image or the upstream example, then move and look
around using W/A/S/D and I/J/K/L controls.

## Prerequisites

- [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
  NVIDIA driver and NVIDIA Container Toolkit.
- One NVIDIA B200 GPU, as requested by this recipe.
- Persistent storage for the [checkpoint](https://huggingface.co/seedleap/Zing-0.5)
  and serving image. Review the upstream source and model usage terms.

## Run

The YAML manifest declares the image build, Runtime and recording configuration.

```sh
cd models/zing-0-5
reactor build
reactor run --gpus device=0
```

The default endpoint is `http://localhost:8080`. Use `--port` to select
another port and `--weights` to select persistent model storage. Rebuild after
changing code or dependencies. Connect a client after the service is available.

## Inputs and controls

A new session waits for an explicit input selection.

| Command | Effect |
| --- | --- |
| `set_prompt(prompt)` | Start a text-conditioned world, or change the description for the next segment of the current world. |
| `set_image(image, prompt, seed)` | Start a new world from JPEG, PNG, WebP or BMP, up to 25 MiB and 100 million pixels. Images are fitted to 1248×704 by aspect-preserving resize and center crop. Blank text uses a neutral prompt; seed -1 keeps the current seed. |
| `example_image()` | Select the upstream pixel-art example and its paired prompt. Missing sample assets return a command error. |
| `set_key(key, pressed)` | Hold or release one of W/A/S/D (forward/left/back/right) or I/J/K/L (look up/left/down/right). |
| `release_controls()` | Release all held keys. |
| `reset(seed)` | Restart the selected world, optionally changing its seed. Requires a prior text or image selection. |

Controls remain held until released. Changes apply to the next video segment.
Disconnecting releases controls and preserves the shared world; ending the
session clears its selections and progress.

## Output and messages

`main_video` emits 1248×704 RGB video without audio. A text-only world begins
with one frame; subsequent segments and all image-conditioned segments contain
16 frames. Playback follows generation throughput.

Command replies are `prompt_queued`, `image_selected`, `action_changed`,
`controls_released` and `rollout_reset`. Read replies from the awaited command.
`chunk_completed` reports each completed segment; `state_update` broadcasts
the selected input, prompt, held controls, seed and progress.

A world stops after 32 segments by default. Its final `chunk_completed` is
followed by `rollout_limit_reached` and the final state. It remains stopped
until an explicit reset or new image selection. The manifest records
`main_video` by default.

## Tests

```sh
cd models/zing-0-5  # From the repository root.
PYTHONPATH=. python -m pytest tests/ -q
```

Local contract tests require no model weights or GPU. The optional upstream
camera-geometry test uses the pinned source checkout; set
`ZING_TEST_SOURCE_PATH` when that checkout is outside the image.
