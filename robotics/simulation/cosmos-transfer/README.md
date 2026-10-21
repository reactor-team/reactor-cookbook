# Render simulator observations with Cosmos3 Nano Sim Transfer

Turn prepared control video into generated RGB observations. Transfer changes
appearance; your simulator still owns motion, physics, and task scoring. These
Python examples call hosted Reactor models and need no local GPU.

## Setup

Requires Python 3.11+, uv, and a Reactor API key with access to
`reactor/cosmos3-nano-sim-transfer`. The SDK uses the unqualified model name.

```bash
cd robotics/simulation/cosmos-transfer
export REACTOR_API_KEY='your-key'
uv sync --locked
uv run make_inputs.py
```

The generator creates four 121-frame MP4s, RGB depth PNGs, and a caption under
`inputs/`. These are synthetic tabletop shapes for exercising the API, not a
physics simulator or validated robot dataset. No external captures are required.
The hosted slug follows the active release; these clients target model 0.3.0,
runtime 3.8.3, and Python SDK 1.10.0.

## Render a recorded trajectory

```bash
uv run render.py --control inputs/depth.mp4 --kind depth \
  --prompt inputs/prompt.txt --output output/depth
```

The client checks the local video, uploads it, waits for fresh model readiness,
sends `generate`, and saves `preview.png` plus `report.json`. `done` reports the
number of frames queued by the model; the video preview can drop/repeat frames.
The snapshot is a sample, not a complete lossless render or training artifact.

### Choose the representation you already have

| `--kind` | Input you prepare | Useful when |
| --- | --- | --- |
| `depth` | RGB8 visualization of depth | Your simulator exposes depth and you want geometry-conditioned appearance |
| `edge` | Edge-map video | You want image outlines to constrain the generated scene |
| `blur` | Blurred RGB video | You want coarse color/layout conditioning; output may remain soft |
| `seg` | Stable, color-coded region maps | You want region-layout conditioning; there is no automatic label-ID or palette mapping |

`--kind` declares what the uploaded pixels mean. It does **not** extract depth,
edges, blur, or segmentation from ordinary video. The fixture generator prepares
each representation separately. To try another control, change both the file and
kind, for example `--control inputs/edge.mp4 --kind edge`.

These learned constraints do not guarantee exact geometry, color fidelity, label
preservation, or useful training data. Depth values here are image intensities,
not metric float arrays. Prepare consistent depth visualization across a sequence.

Use `--frames` 1–600 (default 121), `--seed` 0–4294967295, and `--aspect-ratio`
`16,9`, `4,3`, `1,1`, `3,4`, or `9,16`. Output sizes are 832×480, 736×544,
640×640, 544×736, and 480×832. Source aspect must match the actual output canvas
within 3%; input is resized without cropping. Prepare 30 or 29.97 FPS video with
at least the requested frames and no more than 128 MiB. 30 FPS is the source/playback
cadence, not promised generation speed.

### Generate appearance variants

Keep the same depth trajectory and edit the caption to describe different
lighting/materials. Run the script with a separate output directory for each
caption. This illustrates appearance exploration for synthetic-data workflows;
it is a session-based render, not a durable batch-job API. For dataset export,
use the reliable PNG path below and validate task-relevant content yourself.

## Advance depth rendering step by step

```bash
uv run step.py --depth-dir inputs/depth --prompt inputs/prompt.txt \
  --frames 121 --output output/steps
```

`initialize` sends one depth frame. Each subsequent `advance` sends four.
The script matches episode/step identifiers, verifies every decoded RGB frame's
SHA256, writes ordered PNGs, and records timings. It uses reliable results only;
add `--preview` to also enable the video track. Input PNGs must be RGB8 at the
exact selected canvas size. Episode length is 1 + 4N, at most 597 frames. This path
accepts depth only, not the other three controls.

This example **replays prepared depth**, not a closed-loop policy evaluation.
To integrate a real simulator, replace the next four fixture files with depth
rendered after applying your policy's actions. Consume the prior `step_result`
before advancing simulation time. At 30 FPS, that means feedback once per four
render frames (about 133 ms simulated time), not after every frame. Keep simulator
state and task scoring outside this model.

The sample fails on timeout and disconnects rather than silently skipping a step.
For a persistent integration, inspect `get_episode_state` and retrieve the latest
cached result with `get_step_result`, or retry identical input for the same step.
Do not advance the simulator again on a network timeout. Session loss requires a
new episode; model state cannot be resumed from these PNGs.

## Failure handling and limits

- `upload_file` returning a reference confirms transport, not model acceptance.
  The Python SDK does not surface every upload-hook refusal. This renderer starts
  a fresh source state and waits up to 30 seconds for readiness. A rejected
  replacement upload can otherwise leave an old source active.
- Command refusals raise SDK exceptions. Decode/FPS/aspect failures can instead
  arrive later as `failed`; `render.py` handles both. Episode failures arrive as
  `step_failed` and require a fresh episode.
- Scripts close sessions on errors. Finite rendering uses a five-minute session
  cap and bounded waits. They do not retry blindly or change deployment capacity.
- Visual examples do not establish physical fidelity, policy success, or a
  guaranteed sim-to-real improvement. Short/long clips and non-depth controls need
  application-specific visual acceptance.

## Verification

Both clients were run against PROD 0.3.0 with the generated fixtures. Finite depth
rendering completed 121 frames and delivered a nonblank preview; incremental
rendering delivered all 121 frames across 31 correlated results with matching RGB
hashes. These are integration checks, not a throughput SLA or policy evaluation.
Raw SDK previews are BGRA; the renderer converts them to RGB before saving.
