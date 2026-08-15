# dreamzero-sim

Drives NVIDIA's RoboLab DROID manipulation benchmark from a Reactor-served
`dreamzero`, with the simulator completely unmodified. This example is the
same shape as [`../cosmos-droid/`](../cosmos-droid): RoboLab is not
a library you wrap, so this example *is* the openpi WebSocket port it
connects to. That README explains the shape in more detail.

What makes this one different is the model. `cosmos-nano-policy-droid` is
stateless per request; `dreamzero` free-runs. It does not wait to be asked:
once a prompt is set and every camera has a frame, it infers whenever the
cameras go fresh and broadcasts the chunk. RoboLab asks one question and
expects one answer. Reconciling those two is most of this package.

```
contract.py       BOTH wire schemas: RoboLab's openpi observation keys (incl.
                  the camera-index mapping) and the Reactor commands +
                  action_chunk format
tracks.py         three queue-fed video tracks: one frame per camera per
                  request, no filler, because repeats would fill the model's
                  4-push context window with copies of one observation
bridge.py         the reactor-sdk integration: lazy connect, publish, keepalive,
                  and the obs_seq gate
gateway.py        the openpi Policy RoboLab calls, plus episode lifecycle
policy_server.py  the openpi msgpack-WebSocket protocol server
msgpack_numpy.py  the numpy-aware msgpack codec RoboLab's client speaks
main.py           wires the above together and serves on :5000
```

## Matching chunks to observations

The obvious implementation is wrong:

```python
push_frames(obs); chunk = await next_chunk()      # WRONG
```

The chunk that arrives next was **already in flight** when you pushed, and was
computed from the *previous* observation. It has the right shape, finite values
and a plausible trajectory; nothing about it looks wrong. Every action RoboLab
executes would lag one observation behind, and the only symptom would be a
mediocre success rate.

So the bridge keys everything on `obs_seq`, the field naming the observations
a chunk was computed from: note the highest `obs_seq` seen, push, then discard
arriving chunks until one carries a strictly larger `obs_seq`. Because the model
waits for *every* camera to deliver a fresh frame before inferring, the first
chunk past that mark necessarily saw the pushed frames on all three cameras.

A chunk that arrives without `obs_seq` is refused outright rather than guessed
at.

## Settings that matter

**`--open-loop-horizon 24`** (RoboLab's own default). The checkpoint emits a
`(24, 8)` chunk, so horizon 24 executes exactly one full chunk per inference,
the deployment shape it was trained for. Measured on one task, 3 episodes per
cell: horizon 24 cut dropped-object events by 6-7× and produced every
success observed; horizon 8 never solved it, because it executes the first
third of each chunk and then replans from a partially-executed state. Horizon 8
does match the checkpoint's training-time *frame* stride, which is why it was
tried; matching the frame stride at the cost of truncating the action chunk is
the worse trade. Per-query latency is horizon-independent; what changes is how
many queries an episode makes.

**`--num-envs 1`.** The model holds a *single* causal KV cache. Parallel
RoboLab envs would interleave their observation histories into one episode and
both would degrade. Scale by running more model sessions, not more envs.

**`--cam2-source black`** (the default). That setting sends an all-black
second exterior view, matching the checkpoint's training-time camera
dropout. This was once suspected of stalling the model's every-camera-fresh
check; tested head to head, it does not. Keep it black.

## The camera index trap

RoboLab numbers its exterior cameras from 0; the checkpoint numbers its
video keys from 1. So RoboLab's `exterior_image_0_left`, its real left
camera, the one with the scene in it, is Reactor's `exterior_1`, and
RoboLab's `exterior_image_1_left` (black, per above) is `exterior_2`.

Getting this backwards feeds the model a black primary view and does not
error. `contract.py` holds the mapping and `check_wiring.py` asserts it.

## No heartbeat, late connect

The tracks are queue-fed: each request pushes exactly one frame per camera,
once. The model consumes the 4 newest frames per camera as its temporal
context, so its window holds the last 4 requests. A track that repeated its
frame at a steady rate to keep video flowing (which is what
`cosmos-droid` does, harmlessly, because its model keeps only the newest
frame per view) would fill this model's window with four copies of the newest
observation and quietly delete its temporal context.

The cost is that no video flows between requests. A peer connection brought up
and then left idle for tens of seconds has been observed to leave a serving
runtime mapping inbound video to the wrong track names, after which the model
silently never satisfies its every-camera-fresh check. RoboLab takes minutes to
boot Isaac, so:

- the Reactor session connects on RoboLab's **first request**, not at startup
  (`--connect-eagerly` exists, defaults to off, and should normally stay off);
- the first observation's frames go out one camera at a time,
  `--prime-stagger` seconds apart, so the order the three streams first appear
  on the wire is deterministic rather than a race. The priming frames are that
  real observation's own frames; nothing synthetic reaches the model.

If the model reports no `episode_started` within 45 s of the first observation,
the bridge raises immediately rather than burning the chunk timeout and then
handing RoboLab a stale plan.

## Install

The gateway needs no simulator, no GPU and no model weights. With
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12
export REACTOR_API_KEY='<your key>'   # create one at https://reactor.inc/account/api-keys
```

Set the key in your shell, not in a file you might commit. Nothing here prints
it; `reactor-sdk` mints the session JWT from it in-process, so the key never
leaves the machine. `REACTOR_API_URL` defaults to `https://api.reactor.inc`,
where `dreamzero` is served.

The simulator side is the expensive part, and none of it is Reactor-specific:

- **RoboLab and its Isaac Sim 5.1 image.** Expect a multi-hour install, and
  Isaac's renderer needs an RTX-class GPU: datacenter A100/H100/B200 parts
  have no graphics engines and cannot render.
- **NVIDIA driver on the 580.x branch.** 595.x segfaults Isaac 5.x's renderer
  at startup on Blackwell GPUs, before any asset is touched. This is a known
  driver issue, not a RoboLab or gateway problem. Isaac 5.1 declares a minimum
  of 570.169, so 580.x is inside its window.

## Run RoboLab

```
# 1. the gateway, anywhere RoboLab can reach: host side or another machine
uv run python -m dreamzero_sim.main --port 5000

# 2. RoboLab, unmodified, in its own container
docker run --rm --entrypoint /isaac-sim/python.sh --net host --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $HOME/RoboLab:/workspace/robolab -w /workspace/robolab \
  robolab:<tag> policies/dreamzero/run.py \
    --task BananaInBowlTask --headless \
    --num-envs 1 --num-runs 10 --open-loop-horizon 24 \
    --remote-host localhost --remote-port 5000
```

RoboLab scores the episodes itself and writes them under its `output/`
directory. The gateway binds `0.0.0.0`, so it does not have to share a host
with the simulator; pass the gateway machine's address as `--remote-host`.

Two operational notes. A `dreamzero` session holds two GPU workers, so a
cold start is minutes and `--ready-timeout` defaults to 900 s; and a busy
cluster answers session creation with HTTP 429 `no available capacity`
rather than queueing, which is not a client error. Retry after a short wait.

## Check the wiring

```bash
uv run python check_wiring.py
```

Needs numpy, msgpack and websockets only: no simulator, no GPU, no API key, no
network beyond loopback. It drives the real protocol server with a synthetic
RoboLab request, and asserts the handshake, the camera mapping, the state
clamping, that a chunk without `obs_seq` is refused, and that both ways RoboLab
ends an episode (its `reset` endpoint, and a changed `session_id`) reset the
model. A green run means the gateway is protocol-correct; it says nothing about
task success.

## The bundled protocol server

`cosmos-droid` takes the packaged `openpi-server` as a dependency, which is
the tidier choice where it works. This package includes the ~200 lines of
protocol instead, for two reasons: that dependency pulls in `openpi-client` and
its own dependency tree, which does not build on current Python versions; and
carrying the codec directly means one definition of the wire format, which is
what the evaluation this was ported from relied on.

Both are derived from the openpi project (Physical Intelligence),
Apache-2.0. RoboLab itself is not vendored here; you install it yourself,
as with the other examples' simulators.

## Provenance

<!-- Maintainers: ported from Reactor's internal model repository, commit
     50ccb918 (2026-07-30). Internal identifiers stay in comments, never in
     reader-facing text. -->

Ported from the evaluation harness Reactor runs internally. Through this path
that evaluation ran RoboLab end to end against the production wire contract;
the `--open-loop-horizon` and `--cam2-source` findings above are its
measurements.

Two of its configurations are not ported. The in-process one loads the model
into the bridge process to isolate the observation mapping from the transport,
so it needs the model and its weights locally and cannot run against a served
deployment. The reference-server one is upstream's own server rather than
anything Reactor ships.
