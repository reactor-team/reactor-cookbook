# Play Lyra 2.0 through Reactor

`set_camera_path(trajectory)` accepts a finite JSON `poses[N,4,4]` camera-to-world
path after image selection, with optional `intrinsics` (one `[3,3]` matrix or
`[N,3,3]`, in pixels at 832×480). Include the image anchor and 80 new poses per
chunk (`N = 80 × chunks + 1`), up to 2 MiB. Lyra first calibrates the selected
image without emitting video; the path then aligns once to that calibrated
camera. As in Lyra's custom-trajectory script, the path's intrinsics are used
for the seed and every frame (without them, DA3's predicted intrinsics are
used), and world-to-camera translations are scaled by 1.1 (`--pose_scale`).
Future poses retain the authored trajectory while native spatial memory
continues inside the model.
Generation stops after the final chunk. `reset` recalibrates and rewinds;
image selection restores held controls. `camera_path_selected` reports path
length and `state_update` includes `camera_path_chunks` and
`camera_path_complete`.

Explore an image-conditioned video world with
[Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2).
Change the scene description and control six-axis camera motion while the
world continues. This recipe uses the released four-step video model.

## Prerequisites

- [Reactor CLI](https://docs.reactor.inc/deploy/platform/installation), Docker,
  NVIDIA driver and NVIDIA Container Toolkit.
- One NVIDIA B200 GPU and persistent checkpoint storage.
- The public Lyra 2.0 checkpoints, arranged under
  `source/Lyra-2/checkpoints/` in the weights directory according to the
  upstream instructions. Review the upstream source and model usage terms.
- The Hugging Face cache under `hf-cache/` in the weights directory needs
  `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `google/umt5-xxl` and
  `Ruicheng/moge-vitl` (1.3 GB, for the depth rescale), or network access to
  download them at startup.
- The image's upstream source directory must allow creation of its checkpoint
  symlink at startup. With a read-only filesystem, provision that symlink to
  the selected checkpoint mount before starting the service.

## Run

The YAML manifest declares the serving-image build and Runtime configuration.

```sh
cd models/lyra-2
reactor build
reactor run --gpus device=0 --weights /path/to/lyra-weights
```

The default endpoint is `http://localhost:8080`. Select another endpoint port
with `--port`. Connect after the service is available; rebuild after code or
dependency changes.

## Inputs and controls

A new session waits for an image selection. The initial image preparation
precedes video playback.

| Command | Effect |
| --- | --- |
| `set_image(image, prompt, seed)` | Start a new world from JPEG, PNG, WebP or BMP, up to 25 MiB and 100 million pixels. OpenCV decodes the image and resizes it bilinearly to 832×480, as Lyra's custom-trajectory script does. Blank text uses the default continuation prompt; seed -1 keeps the current seed. |
| `random_image()` | Select an upstream sample image and its paired description. |
| `set_prompt(prompt)` | Change the description for the next video segment. Requires a selected image. |
| `set_camera_motion(forward, strafe, vertical, pitch, yaw, roll)` | Set all six held camera axes together, after selecting an image. |
| `set_forward`, `set_strafe`, `set_vertical`, `set_pitch`, `set_yaw`, `set_roll` | Set an individual held axis, including before image selection. |
| `release_camera()` | Stop all camera movement. Requires a selected image. |
| `reset(seed)` | Restart from the selected image and current prompt; progress immediately returns to zero. |

Axis values range from -1 to 1; zero stops movement. Forward/strafe/vertical
control translation, and pitch/yaw/roll control rotation. Changes take effect
on the next video segment. Selecting an image or resetting clears all axes.
A viewer disconnect releases camera controls and preserves the shared world.

The recipe serves the defaults of Lyra's official custom-trajectory script
(`lyra_2._src.inference.lyra2_custom_traj_inference --use_dmd`), not the GUI's:
832×480, DA3 seed depth rescaled to MoGe, pose scale 1.1, a DA3 cache update
every 8 frames over at most 10 history frames, without chunk-end frames or
predicted-pose correction. Each new world seeds every random draw, including
the DMD re-noising between the four steps, as the script does per sample. The
same image, prompt, seed and camera path (with intrinsics) therefore reproduce
the script's video for the same inputs.

## Output and messages

`main_video` contains 832×480 RGB video, 80 frames per segment, without audio.
Playback follows generation throughput. `chunk_completed` reports generation
progress; delivery and playback can finish later.

Commands return `image_selected`, `prompt_queued`, `camera_changed` or
`reset_queued`. Read command replies from the awaited call.
`state_update` broadcasts the selected image, queued and most recently used
prompt, seed, completed segments and camera controls.

Ending the session clears its selections and progress. The manifest records
`main_video` by default.

## Tests

```sh
cd models/lyra-2  # From the repository root.
PYTHONPATH=. python -m pytest tests/ -q
```

CPU contract tests cover input preparation, resets, session isolation, camera
continuity, rejected steps and failed generation without advancing progress.
