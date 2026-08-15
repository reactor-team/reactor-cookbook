# Quickstart scripts and guides

Six runnable scripts, each with a guide, that drive a hosted Reactor robotics
policy from Python, replaying recorded observations. They need no simulator,
no GPU and no model weights: `uv sync`, an API key, and a few minutes.

Open the guide for the model you care about and run its script; there is no
reading order. `xwam` is the reference implementation of
the generic [robot policy client contract](./robot-policy-client-contract.md),
so it is the one to read if you want the contract itself rather than a model.

## Choose a model

Every guide starts with the same lightweight recorded-observation replay. Its
last two sections cover the optional closed-loop simulator and the remaining
work for physical deployment. Published figures link to upstream sources;
figures labeled Reactor-measured come from the committed harness or fixture
provenance.

| Guide | Protocol | Chunk | Closed-loop harness |
|---|---|---|---|
| [`lingbot-va`](./lingbot_va_quickstart.md) | executed-action echo | `(16, 7)` eef deltas | [`LIBERO`](../libero), CPU |
| [`cosmos-nano-policy-droid`](./cosmos_droid_quickstart.md) | stateless executed-step report | `(32, 8)` absolute joints | [`RoboLab`](../cosmos-droid), RTX GPU |
| [`xwam`](./xwam_quickstart.md) | request/reply with `chunk_id` echo | `(32, 14)` delta joints | [`RoboTwin 2.0`](../robotwin), CUDA GPU |
| [`groot-n17`](./groot_n17_quickstart.md) | free-running | `(40, 17)` named fields | not wired in this repo |
| [`dreamzero`](./dreamzero_quickstart.md) | free-running with `obs_seq` gate | `(24, 8)` absolute joints | [`RoboLab`](../dreamzero), RTX GPU |
| [`xr1-robocasa365`](./xr1_robocasa365_quickstart.md) | echo-gated from first chunk | `(16, 60)` packed, 12 live | [`RoboCasa365`](../robocasa365), CUDA GPU |

`xwam` is the generic contract's reference implementation. The other five each
depart from it (different state fields, no `chunk_id` echo, free-running
instead of request/reply) and each guide states how for its own model. A
client written against the generic contract will not drive them unchanged.

## Setup

With [uv](https://docs.astral.sh/uv/):

```sh
cd robotics/sim/notebooks        # from the cookbook repo root
uv sync --locked --python 3.12

export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
uv run python xwam_quickstart.py      # or any other *_quickstart.py
```

For an explicit package install instead of the project environment:

```sh
uv venv --python 3.12
uv pip install "reactor-sdk==0.8.0" "aiortc>=1.9" "av>=12.0" "numpy>=1.26"
```

Set the key in your shell, not in a script: a key pasted into a script is
committed. Nothing in these quickstarts prints it; the key check reports only
its length.

At session close the SDK may log `WARNING Control channel not open; dropping
'unpublish_track' notification` a few times. That is benign teardown noise —
the session is already closing — not a failure.

`REACTOR_API_URL` defaults to `https://api.reactor.inc` (PROD, where all six
models are served). It exists as an escape hatch for pointing at another
deployment.

## Contents

```
<model>_quickstart.py            the runnable script, one per model
<model>_quickstart.md            its guide
reactor_robotics/
  session.py       connect + READY + tracks + keepalive; the shared plumbing
  track.py         a video track that repeats one frame until you replace it
  xwam.py          XwamClient.predict():        lock-step, chunk_id echoed
  lingbot_va.py    LingbotVaClient.predict():   lock-step, executed-action echo
  cosmos_droid.py  CosmosDroidClient.predict(): stateless, executed-step report
  groot_n17.py     GrootN17Client.predict():    free-running, paired by ordering
  dreamzero.py     DreamZeroClient.predict():   free-running, keyed on obs_seq
  xr1_robocasa365.py  Xr1Robocasa365Client.predict(): lock-step, echo-gated from the first chunk
examples/
  xwam_examples.npz              5 observations from a recorded RoboTwin 2.0 eval
  dreamzero_examples.npz         5 observations recorded against the live model
  lingbot_va_examples.npz        5 observations recorded against the live model
  groot_n17_examples.npz         5 real FR3 rig captures + live-recorded chunks
  cosmos_droid_examples.npz      5 observations recorded against the live model
  xr1_robocasa365_examples.npz   5 synthetic observations + live-recorded chunks
  *_examples.manifest.json       provenance manifests for live-recorded .npz files
  record_*_examples.py           the scripts that recorded (and calibrated) them
  PROVENANCE.md                  where every example and tolerance came from
robot-policy-client-contract.md  generic contract plus each model's departures
```

`cosmos_droid_quickstart.py` uses `cosmos_droid_examples.npz` and its
calibrated tolerances (both ship in `examples/`); if the file is missing —
say, when running the script standalone outside a clone — it falls back to
the recording script's own deterministic observations, so it runs either way.

"Recorded examples" means real observations captured from an evaluation run
or recorded against the live model, stored together with the actions the model
returned at the time. Each script sends the same observations to the hosted
model and checks the answers against what was recorded.

`reactor_robotics/` is the same client an evaluation harness or a controller
for a physical robot uses. The wire protocol does not change when frames start
coming from a camera instead of an `.npz`, which is the point of the contract:
an eval client and a robot client are the same client. `session.py` is the
same code path that produced the published X-WAM evaluation numbers.

## Session lifecycle

`reactor_robotics/session.py` manages the connection lifecycle (handler
registration, readiness, keepalive) so no script has to:

1. Register handlers before `connect()`. `READY` can arrive before your
   first `await` after `connect()` returns.
2. Publish tracks only after `READY`. An earlier `publish_track` creates no
   track, and the model then waits for frames that never arrive.
3. Ping every 10 s, for the whole session. The runtime disconnects a client
   that stays quiet for 20 s, and 0.8.0 leaves keepalive to the client. This
   matters most when you are not sending anything else, for example while a
   robot executes a chunk. Three scripts demonstrate it by sitting idle for
   25 s.

Keep `logging.basicConfig(level=INFO)` on: dropped commands are logged, not
raised.

## What gets checked

Each script claims only what its model's determinism supports. A numeric
replay comparison needs the model to be seeded and reference outputs to
compare against; only `xwam` has both. Each guide's "Check the results"
section states its own checks in full, and every tolerance is documented
with the measurement behind it in
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md).

| Model | Seeded? | The check |
|---|---|---|
| `xwam` | Yes | exact replay, `max\|Δ\| ≤ 5e-2` on executed actions |
| `lingbot-va` | Partly: 4 pinned seed rows | structure, plus the pinned rows exactly |
| `groot-n17` | No | structure, plus two calibrated physical tolerances |
| `cosmos-nano-policy-droid` | No | structure, plus two calibrated physical tolerances |
| `dreamzero` | No | structure, plus a reported-only spread band |
| `xr1-robocasa365` | No | structure, plus reported-only per-column and run-to-run bands |

`xwam`'s replay gate is headroom over nine measured runs: the worst delta
seen was `3.7e-2` and the gate sits at `5e-2`. Every other tolerance is the
mean plus three standard deviations, measured over 3 repeat passes of the
same observations at recording time. A three-standard-deviation bound from 15
samples puts a single value outside it occasionally by construction, which is
why the bands are reported and the deterministic checks carry the pass/fail.
Every tolerance in these checks comes from a measurement.

## Latency

The number a script prints is not always the model's compute; each guide
has a Latency section with its own breakdown.

| Model | Printed latency |
|---|---|
| `xwam` | ~163 ms model inference; ~290 ms script path |
| `lingbot-va` | p50 208 ms, echo → chunk |
| `groot-n17` | the pairing cost, ~1.5 × the ~100 ms chunk period |
| `cosmos-nano-policy-droid` | ~745 ms p50 think + wire |
| `dreamzero` | median 258 ms against a 267 ms operating point |

## Nothing to strip before committing

A script keeps its results in your terminal, so there is no stored output to
clear before committing. The reason still applies to anything you paste back
into the repository: output can carry key fragments, internal hostnames,
session ids and ICE candidate IPs, none of which belong in a repository
shared outside the team.

```sh
cd robotics/sim/notebooks
uv run python xwam_quickstart.py     # results stay in your terminal
```

One maintenance note: each guide links its model's `predict()` in the
matching `reactor_robotics/*.py` rather than copying it, so there is no
second copy to keep in step when you change `predict()`.

Run the offline consistency checks after changing a guide, fixture, or model
list:

```sh
uv run python -m unittest discover -s tests
```

## Sessions hold GPUs

Every connection is a live GPU worker: Always `await client.close()`.
Every script does, and the patterns are copyable.

HTTP 429 `no available capacity` on session creation means no capacity is
free right now. Wait and retry, or ask Reactor for additional capacity. All
six single-session models are affected; the recording scripts retry
automatically.

## Going further

- The paper's own simulator. See the matrix at the top. Five of the six
  are in this repo: [`../libero/`](../libero) (LIBERO, no GPU),
  [`../cosmos-droid/`](../cosmos-droid) and
  [`../dreamzero/`](../dreamzero) (RoboLab/Isaac, RTX GPU required),
  [`../robotwin/`](../robotwin) (RoboTwin 2.0, CUDA GPU), and
  [`../robocasa365/`](../robocasa365) (RoboCasa365, CUDA GPU). Only
  `groot-n17` has none. Expect a multi-hour install for Isaac, RoboTwin, or
  RoboCasa365; LIBERO is 20-40 minutes. None of it is needed for the
  quickstarts.
- Physical deployment. Same client, frames from your cameras. See
  [robot-policy-client-contract.md](./robot-policy-client-contract.md) and the
  Physical deployment section of the guide for your model, which states what
  exists today and what you would have to build. Reactor support for physical
  deployment is coming soon; the contract is published now so integration can
  start against a stable target.
