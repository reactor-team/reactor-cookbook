# robotwin-sim

Runs the RoboTwin 2.0 authors' own evaluation client, unmodified, against an
`xwam` policy served on Reactor. Their client keeps the episode loop, the
seeds, the expert feasibility checks, the instruction sampling and the
success predicates. This package substitutes the transport and who serves
the policy, nothing else, so a success rate measured through it is
comparable to the authors' published numbers.

## Why a gateway

`libero` wraps its simulator as a Python library and owns the rollout
loop. RoboTwin 2.0 is not that kind of sim: the authors ship an evaluation
client that owns the loop, and the seam it exposes for a policy is a broker
port speaking pickled dicts over zmq. So this example *is* that port:

```
RoboTwin 2.0 env (upstream, unmodified)              api.reactor.inc
  robotwin_client.py ──pickle/zmq:10086──▶  Gateway ──▶ Bridge ──▶ xwam
  (via the authors' deploy_policy.py)          │   3 video tracks (each repeats
                                               │   its frame between requests)
                                               └── set_state_json / action_prediction
```

```
contract.py   BOTH wire schemas: the authors' pickled request/reply, and the
              Reactor commands + action_prediction chunk format
tracks.py     three video tracks, one per camera view, repeating + paced
bridge.py     the reactor-sdk integration: connect, publish, keepalive, and
              one lock-step predict() per request
gateway.py    the zmq ROUTER that binds the authors' port
main.py       wires the above together
```

There is no `env.py` and no rollout loop, because the authors' client
already has both. There is also no threading: their client is lock-step (one
request out, block for the reply) and the model takes one inference at a
time, so a single asyncio loop runs the zmq socket and the WebRTC session
together.

## One request cycle

Per request the gateway:

1. **recovers the rendered frames.** The client normalises its uint8 frames
   into `[-1, 1]` before sending them. That mapping is bijective from uint8,
   so inverting it returns the *exact* pixels the simulator rendered, not an
   approximation. `check_wiring.py` proves it for all 256 levels.
2. **publishes one frame per view**, then waits out a short settle. Frames
   are sent over WebRTC video while commands are sent over the data channel,
   and the model pairs a request with the next frames to *arrive*. So the new
   observation has to clear the encoder before the request goes out, or the
   model answers from the previous one and the reply looks entirely
   plausible.
3. **sends `set_task_description`** (only when the instruction changes) and
   `set_state_json`, carrying the proprioception, a `chunk_id`, and the
   client's `(env_rank, rollout_id, step_id)` triple.
4. **awaits the `action_prediction`** whose `step` echoes that `chunk_id`,
   and pickles `{"actions": (32, 14), "proprios": (9, 16)}` back.

The seed triple is what makes a rollout reproducible: the model derives its
sampling noise from it, so the gateway relays it verbatim.

## Two virtualenvs

Run the simulator and this gateway in separate virtualenvs:

| | Contains | Constraint |
|---|---|---|
| **Sim env** | RoboTwin 2.0, SAPIEN, curobo, the authors' client | `numpy==1.23.5` |
| **Gateway env** | this package | `numpy>=1.26` (this package's own floor) |

The numpy pins are directly incompatible. The split costs nothing: the two
processes only ever exchange pickled dicts over a local socket, and pickle
does not care which numpy wrote them. The gateway needs no simulator, no GPU
and no model weights.

## Install the simulator

Follow RoboTwin 2.0's own installation guide, and the X-WAM authors'
evaluation setup on top of it. Expect a multi-hour install on a CUDA GPU
machine: SAPIEN, a renderer, curobo's compiled kernels and the task assets.
Nothing about it is specific to Reactor, and none of it is vendored here.

Upstream: **[github.com/sharinka0715/X-WAM](https://github.com/sharinka0715/X-WAM)**
(Apache-2.0), whose `evaluation/` directory holds `robotwin_client.py` and
`deploy_policy.py`. This path's evaluation numbers are pinned to upstream
commit `72cfb86b`.

You do not need the authors' `policy_broker.py` / `policy_server.py`, and
you do not need model weights: this gateway replaces both.

## Install the gateway

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12
export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
```

Set the key in your shell, not in a file you might commit. Nothing here prints
or logs it. `reactor-sdk` exchanges it for a session JWT through the Reactor
API's `/tokens` endpoint over HTTPS.

`REACTOR_API_URL` defaults to `https://api.reactor.inc`, where `xwam` is
served. It exists as an escape hatch for another deployment.

## Run the client

```
# 1. the gateway, in its own venv (binds the port their client expects)
uv run python -m robotwin_sim.main --port 10086

# 2. the authors' client, unmodified, in the sim env, from their checkout
cd <X-WAM checkout>/evaluation
python robotwin_client.py --task_name <task> --task_config demo_randomized \
    --num_evals_per_worker 50 --server_port 10086 --save_root_dir <out>
```

The client scores the episodes itself and writes them under
`--save_root_dir`. Run it with the task list and seed count your comparison
needs; this package deliberately ships no sweep driver or scorer, because
upstream owns both.

Wait for the gateway's `tracks published` line before starting the client. A
cold deployment has to schedule a GPU and stage weights, which is why
`--ready-timeout` defaults to 300 s.

## Check the wiring

```bash
uv run python check_wiring.py
```

Needs numpy and pyzmq only: no simulator, no GPU, no network, no API key. It
builds a request exactly as the authors' client pickles one, runs it through
the real gateway with a stub predictor, and asserts the frame inversion is
exact, that malformed requests are rejected, and that a retry changes
the request string without changing a single seed.

## Silent failure modes

Four classic bugs worth checking on every run, because each one silently
produces a *number* rather than an error:

1. **`done` must not mean `success`.** A failing episode has to terminate at
   the step limit and be counted as a failure. This is upstream's code, and
   this gateway does not touch it.
2. **Correlated initial states.** Log the accepted (expert-check-passing)
   seed list per task and reuse it across arms, so A/B comparisons pair.
3. **A stale chunk crossing a reset.** Every reply echoes its `chunk_id`;
   the bridge discards a reply whose echo does not match and logs it, and
   the count appears in the shutdown summary.
4. **A lost message stalling the episode forever.** On timeout the bridge
   re-sends with a bumped `retry` counter. That changes the request *string*
   (re-sending the exact same request is deduplicated as unchanged state and
   never answered) while holding every seed, so the retried answer is
   identical to the lost one when frames are fed directly; over the video
   transport the re-encoded frames make it equal within tolerance instead.

## One caveat

The authors' transport carried raw arrays; WebRTC carries H.264 video. That
hop is lossy, and it is the one difference this path cannot hide. Both arms
of an A/B comparison share it, so a comparison *between* two Reactor-served
configurations is unaffected. A comparison against the authors' own
published numbers is not, and any report should say so.

## Provenance and licensing

<!-- Maintainers: ported from Reactor's internal model repository, commit
     f8e1c767 (2026-08-10). Internal identifiers stay in comments, never in
     reader-facing text. -->

The code here is Reactor's own, ported from the harness it runs internally.
Through this exact path that evaluation reproduced the reference outputs
recorded from the authors' own serving stack 30/30 exactly, and scored
79.3% (138/174, Wilson 95% [72.7, 84.7]) on RoboTwin 2.0's ten hardest
tasks.

Two pieces were deliberately left behind: a checker that compares served
outputs against those recorded references for an exact match, which only runs
inside the model's own serving container and so cannot live in this repo;
and the sweep driver and success scorer, because upstream's client already
scores episodes.

Nothing from the X-WAM authors' repository is copied or vendored here; you
clone it yourself, as with the other examples' simulators. It is Apache-2.0,
its `LICENSE` travels with your checkout, and the protocol this gateway
speaks is theirs.
