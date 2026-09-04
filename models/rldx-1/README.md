# RLDX-1 with RTC

This recipe deploys RLDX-1 on one NVIDIA Blackwell GPU. It contains the
Reactor adapter and the vendored RLDX-1 inference source used by the model image.

The default configuration enables guided Real-Time Chunking (RTC):

- action horizon: 16 control steps;
- RTC delay: 5 control steps; and
- execution horizon: 8 control steps.

The released RoboCasa checkpoint supports guided RTC. The `trained` RTC mode
requires a checkpoint trained with RTC.

## Prerequisites

- Reactor CLI authentication
- The `RLWRLD/RLDX-1-FT-ROBOCASA` checkpoint

The source is Apache-2.0. The checkpoint uses the separate
[RLWRLD Model License](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA/blob/main/LICENSE.md),
which includes use restrictions. Review that license before downloading or
deploying the weights.

## Download the checkpoint

Run this command from this directory:

```bash
uvx --from 'huggingface_hub[cli]' hf download RLWRLD/RLDX-1-FT-ROBOCASA \
  --local-dir ./weights
```

The `weights/` directory is excluded from the image build. `reactor.yaml`
names it in `runtime.weights_path`, so the publish command uploads it as the
release's weight bundle.

## Publish and deploy

Run these commands from this directory. The publish command builds the image
from the `build:` block of `reactor.yaml`, pushes it, and uploads the weights.
The build compiles FlashAttention 2 for Blackwell from source, so the first
build takes a while; later builds reuse the cached layer until the
dependencies change.

```bash
reactor model publish
reactor model deploy
```

To build the image without publishing, run `reactor build`.

### Build on a remote machine

The build and the publish both use the Docker daemon the CLI is pointed at,
and the publish exports the image from that daemon before it pushes. For an
image this size, run both on the build machine rather than through a laptop.
On that machine, clone this folder, download the weights as above, then build
and publish from the built image:

```bash
reactor build
reactor model publish --source reactor-local/rldx-1:dev
reactor model deploy
```

`reactor build` tags the image `reactor-local/<folder name>:dev`. With
`--source`, publish skips the build, pushes that image, and still uploads
`./weights`.

The deployment target is part of `reactor.yaml`:

```yaml
deployment:
  instances:
    - region: us-west
      count: 1
```

`reactor model deploy` reads this section directly.

## Run the test client

The matching client publishes one synthetic frame on each of the three camera
tracks every control tick. Each tick uses one capture timestamp for all frames
and the proprioceptive state.

```bash
cd client
uv sync
export REACTOR_API_KEY=rk_your_key_here
uv run python main.py --model <account-slug>/rldx-1 --duration 60
```

Replace `<account-slug>` with the account slug printed by Reactor when the
model is published.

The summary reports RTC request-to-response latency, observation age, and
cross-view capture skew at p50 and p99. The returned actions are model outputs
for synthetic inputs and must not be sent to a robot.

## Source provenance

The `rldx/` directory is the inference subset of
[`RLWRLD/RLDX-1`](https://github.com/RLWRLD/RLDX-1) at commit
`ecbfaf80cd031dcc892186ed30465de3591047e6`. Reactor-specific RTC changes are
documented in `rldx/PROVENANCE.md`.
