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

The `weights/` directory is excluded from the image build. Reactor uploads it
as the release's weight bundle.

## Publish and deploy

A new release must be published once before it can be deployed:

```bash
reactor model publish --weights ./weights
reactor model deploy
```

After publication, run `reactor model deploy` from this directory whenever you
want to activate the release named in `reactor.yaml`.

## Run the test client

The matching client publishes one synthetic frame on each of the three camera
tracks every control tick. Each tick uses one capture timestamp for all frames
and the proprioceptive state.

```bash
cd ../../robotics/rldx-1/client-python
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
