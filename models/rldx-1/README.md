# RLDX-1

This recipe serves [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1) as a
three-camera vision-language-action model. It accepts synchronized left,
right, and wrist views plus robot state and a task instruction, then streams
action chunks over the data channel.

The adapter also demonstrates the server half of the frame-metadata pattern in
the matching [`robotics/rldx-1`](../../robotics/rldx-1) client: camera views
are aligned by their declared capture time, robot state stays attached to the
selected frames, and each action echoes the client observation that produced
it.

## Prerequisites

- Docker with the NVIDIA Container Toolkit
- An NVIDIA RTX PRO 6000 (Blackwell, `sm_120`)
- The [Reactor CLI](https://docs.reactor.inc/deploy/overview) if you prefer the
  CLI workflow over the included Docker script
- The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

The source code is Apache-2.0. The checkpoint uses the separate
[RLWRLD Model License](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA/blob/main/LICENSE.md),
which includes non-commercial restrictions. Review it before downloading or
deploying the weights.

## Download the checkpoint

For the direct-Docker flow, download the RoboCasa checkpoint anywhere on the
host and pass that directory to the run script:

```bash
hf download RLWRLD/RLDX-1-FT-ROBOCASA \
  --local-dir ~/rldx-robocasa
```

The adapter reads the checkpoint's own modality configuration, so the schema
it announces reflects the loaded views, state dimensions, action dimensions,
and temporal window.

## Run with Docker

The included script builds the image, mounts the checkpoint read-only, starts
the server on port 8080, and manages its logs and container lifecycle:

```bash
WEIGHTS_DIR=~/rldx-robocasa ./scripts/run_reactor.sh build
WEIGHTS_DIR=~/rldx-robocasa ./scripts/run_reactor.sh start
./scripts/run_reactor.sh status
./scripts/run_reactor.sh logs
```

Stop it when finished:

```bash
./scripts/run_reactor.sh stop
```

The first build compiles FlashAttention for `sm_120`. Reduce build parallelism
on a smaller host with `FLASH_ATTN_MAX_JOBS=<n>`.

## Run with the Reactor CLI

The CLI reads weights from the path declared in `reactor.yaml`. Download the
checkpoint there, then build and run the same Dockerfile:

```bash
hf download RLWRLD/RLDX-1-FT-ROBOCASA \
  --local-dir ~/.cache/reactor_registry/rldx-1
reactor build
reactor run --gpus device=0
```

In another terminal, run the local client:

```bash
cd ../../robotics/rldx-1/client-python
uv sync
uv run python main.py --local
```

The bundled client publishes synthetic images and state to exercise the wire
contract. Replace both sources before controlling a robot.

## Upstream pin

The Dockerfile fetches `RLWRLD/RLDX-1` at commit
`ecbfaf80cd031dcc892186ed30465de3591047e6`. Update the pin deliberately and
revalidate the adapter whenever upstream changes.
