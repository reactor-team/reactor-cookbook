# cosmos-droid-sim

Drives NVIDIA's RoboLab DROID manipulation benchmark from a Reactor-served
`cosmos-nano-policy-droid`, with the simulator completely unmodified.

## Why a gateway

`libero` wraps its simulator as a Python library and owns the rollout
loop. RoboLab is not that kind of sim: Isaac Sim owns its process, its
python, and its episode loop, and the seam it exposes for a remote policy is
an openpi WebSocket port (`run.py --remote-host --remote-port`). So this
example *is* that port:

```
main thread:    openpi WebsocketPolicyServer  ◄── RoboLab (Isaac container,
                     │ infer()                     unmodified policies/cosmos3/run.py)
                     ▼
                GatewayState  ── thread-safe hand-off ──┐
bridge thread:  Bridge (reactor-sdk) ── api.reactor.inc ── cosmos-nano-policy-droid
```

RoboLab stays completely unmodified, so any success rate you measure is
comparable to RoboLab's own published evals.

```
contract.py   BOTH wire schemas: the openpi obs dict RoboLab sends (incl. the
              composite-image split back into three views) and the Reactor
              commands / action_prediction chunk format
gateway.py    the hand-off: this example's RolloutState, and also the
              openpi Policy (infer()) the server calls
tracks.py     three video tracks (wrist + 2 exterior), one frame per request,
              heartbeat between requests
bridge.py     the reactor-sdk integration: publish tracks, task/proprio/echo
              commands, resolve each pending request with the next chunk
main.py       wires the above together and serves on :8000
```

## One request cycle

RoboLab requests a chunk, executes all 32 absolute joint-position actions at
15 Hz open-loop (~2.1 s), then requests again. Per request the gateway:

1. splits the composite frame into `wrist_view` / `exterior_view_1` /
   `exterior_view_2` and lets the tracks publish them;
2. sends `set_task_description` (on change) and `set_proprio_json`;
3. echoes the previous chunk via `set_executed_step_json`; this is the
   model's flow control: it will not predict chunk N+1 until step N is
   echoed;
4. awaits the next `action_prediction` and returns it to RoboLab.

Two properties keep the protocol simple:

- The model is stateless per request: no KV cache, no reset event on the
  wire at all. A new episode or task change needs no ceremony; the prompt
  and proprio are sent with every prediction.
- Whole-chunk execution is the measured optimum for this policy. A
  replan-horizon sweep found success strictly *increases* with open-loop
  horizon (mid-chunk replanning collapses it), and that serving latency
  below ~300 ms costs nothing. RoboLab's own request-execute-request
  cadence is exactly that.

## Install

The gateway needs no simulator, no GPU and no model weights. With
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12
```

## Run

Requirements that don't fit in `pyproject.toml`: an RTX-class GPU for
Isaac's renderer (L40S validated; datacenter A100/H100/B200 have no graphics
engines and cannot render), the `robolab` Docker image built with Isaac 5.1,
NVIDIA driver 580.x (595.x segfaults Isaac at boot), and the model reachable
at your API URL (`REACTOR_API_URL`, default `https://api.reactor.inc`).

```
# 1. the gateway (this repo, host side or any machine RoboLab can reach)
export REACTOR_API_KEY=...   # create one at https://reactor.inc/account/api-keys
uv run python -m cosmos_droid_sim.main --port 8000

# 2. RoboLab, unmodified, in its own container
docker run --rm --entrypoint /isaac-sim/python.sh --net host --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $HOME/RoboLab:/workspace/robolab -w /workspace/robolab \
  robolab:<tag> policies/cosmos3/run.py --task BananaInBowlTask \
  --headless --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000
```

RoboLab writes the scored episode (success boolean + videos) to
`output/<ts>_cosmos3/<task>/`.

## Check the wiring

No GPU, no network, numpy only:

```bash
uv run python check_wiring.py
```

## Timing caveat

Frames are sent over WebRTC video; commands are sent over the data channel;
the engine pairs the *newest* frame with a prediction. `--settle` (default
0.1 s) pauses between pushing frames and sending the echo that unblocks the
next prediction, so the fresh frame lands first. This reduces the risk but
does not remove it. If predictions look one-observation stale, raise it.

## Provenance

<!-- Maintainers: the harness and the latency study referenced in this section
     are tracked internally as REA-4355 and REA-4616. Tracker ids stay in
     comments, never in reader-facing text. -->

This wiring is a cleaned-up port of the harness Reactor used for its own
benchmark runs of this policy. The same contract sustained a 120-task ×
3-rollout benchmark, scoring 40.0% served in-process, a Reactor measurement,
against NVIDIA's published 39.7%, and made 8/10 solves over the production
wire (p50 745 ms think+wire, 0 stalls in 150 chunks). If your rollouts
misbehave, suspect your wiring or serving before the contract.
