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

- The [Reactor CLI](https://docs.reactor.inc/deploy/overview)
- Docker with an NVIDIA Blackwell GPU; the included build targets a B200
- The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

The source code is Apache-2.0. The checkpoint uses the separate
[RLWRLD Model License](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA/blob/main/LICENSE.md),
which includes non-commercial restrictions. Review it before downloading or
deploying the weights.

## Download the checkpoint

From this folder, download the RoboCasa checkpoint into the weights path named
by `reactor.yaml`:

```bash
hf download RLWRLD/RLDX-1-FT-ROBOCASA \
  --local-dir ~/.cache/reactor_registry/rldx-1
```

The adapter reads the checkpoint's own modality configuration, so the schema
it announces reflects the loaded views, state dimensions, action dimensions,
and temporal window.

## Run locally

```bash
reactor build
reactor run --gpus device=0
```

The first image build compiles FlashAttention from source. If the Docker
builder has limited memory, reduce parallel compilation with
`--build-arg FLASH_ATTN_MAX_JOBS=<n>` using the equivalent Docker build flow.

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
