# XR-1 RoboCasa365 quickstart

Get robot actions from the hosted XR-1 RoboCasa365 model in a few minutes.

[`Xiaomi-Robotics-1`](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)
is a vision-language-action policy: a Qwen3-VL-4B
backbone with a 604M DiT action expert, about 5B parameters, generating actions
by flow matching. This checkpoint is its fine-tune for the RoboCasa365 kitchen
benchmark on the single-arm PandaOmron embodiment. You stream three camera
views plus a short robot-state history, and it returns a 16-step action chunk
in the vendor's packed 60-column layout, of which the first 12 columns drive
this embodiment.

The script runs live checks against the hosted model. Once the recorded-example
fixture exists (observations recorded against this deployment, stored together
with the chunks it returned at the time) it compares against those recordings
instead. The script detects which mode it is in.

## Setup

Two things: a Reactor API key (create one at [reactor.inc/account/api-keys](https://reactor.inc/account/api-keys)) and the
client package.

```bash
cd robotics/sim/notebooks && uv sync --python 3.12
export REACTOR_API_KEY='<your key>'   # in your shell
uv run python xr1_robocasa365_quickstart.py   # then run it
```

For an explicit package install instead of `uv sync`, see the
[shared setup](./README.md#setup).

The replay needs no simulator, GPU, or model weights. Keep the key in your
shell; [`xr1_robocasa365_quickstart.py`](./xr1_robocasa365_quickstart.py)
reports only its length and the API endpoint.

The script falls back to the recording script's own deterministic observations
when `examples/xr1_robocasa365_examples.npz` is not present, so it runs either
way. Without calibrated bands it runs only the checks that need none and
reports the rest.

## Connect

[`reactor_robotics/session.py`](./reactor_robotics/session.py) handles the
shared, order-sensitive connection lifecycle: register handlers, await
`READY`, publish tracks, and keep the session alive. See the
[shared lifecycle notes](./README.md#session-lifecycle) for the failure modes.

```python
client = Xr1Robocasa365Client()   # model="xr1-robocasa365", 15 fps, 256x256
await client.connect()
```

Capacity: the deployment serves one session at a time. HTTP 429
`no available capacity` on session creation means no capacity is free right
now. Wait and retry, or ask Reactor for additional capacity; the script
retries the connect a few times before giving up.

This model's wire: three tracks, three commands, one reply type, and a `reset`
event.

| Direction | Message | Notes |
|---|---|---|
| → | publish tracks `left_agentview`, `right_agentview`, `wrist_view` | the checkpoint's declaration order, which is also the prompt order |
| → | `set_task_description {task_description}` | ≤300 chars, on change |
| → | `set_state_history_json {state_history_json}` | ≤24000 chars, per chunk |
| → | `set_executed_step_json {executed_step_json}` | ≤8000 chars, per chunk |
| ← | `action_prediction {action: [16,60], step}` | first 12 columns live for PandaOmron |
| → | `reset` | episode boundary; clears the history and the flow counter |

The three cameras are separate named tracks, and the model pairs them frame for
frame in arrival order into complete observations before it samples its
history. Publish them as a set. A camera that skips a step shifts its whole
history against the other two, which reads as a bad policy rather than as a
client bug.

## Get actions

Every prediction, including the first, needs a task, a state history, complete
frame sets, and an echo:

```
state_history_json   {"state_history": [[<14 floats>], ...]}   exactly 4 rows
executed_step_json   {"step": <int>}
```

The state history is exactly 4 rows of 14 floats, oldest first, sampled 2
environment steps apart so the rows pair with the 4 video frames the model
holds. Each row is `[0:3]` left EE position xyz, `[3:6]` left EE axis-angle,
`[6]` left gripper, then the same for the right arm. This embodiment is
single-arm, so the right block stays at rest. While an episode is younger than
the window, repeat the earliest observation to fill the missing rows;
`encode_state_history()` does that for you. Rows are zero-padded to the model's
60-dim layout server-side.

The model answers only when the echoed `step` is strictly greater than the last
value it advanced on. A repeat, a lower value, or malformed JSON produces no
chunk and no error. This stops the model running ahead of a client that is
still executing, and stops a stalled control loop's retry from causing a
spurious re-prediction. The echo carries the step alone; this model does not
want the executed rows back.

The actions are the vendor's packed 60-column layout, already decoded
(denormalized) with the checkpoint's own `robocasa365` statistics, so they are
the same numbers upstream's eval client works with after its `decode_action`
call. The benchmark embodiment consumes `action[:, :12]`; the remaining 48
columns are the packed-layout padding.

Three ways this differs from the generic
[contract](./robot-policy-client-contract.md):

| Generic contract | XR-1 RoboCasa365 |
|---|---|
| `state_json` is `{proprio: [N floats], chunk_id}` | `state_history_json` (4 rows) + `executed_step_json`, no `chunk_id` |
| `step` echoes your `chunk_id` | `step` is the model's own prediction counter; you echo execution progress |
| the first request needs no echo | the first request needs one too |

That last row is the one to keep in mind. Requiring the echo on the first
chunk as well is deliberate: an
asymmetric first prediction races the first echo, and losing that race leaves
the echo unconsumed, which reopens the gate on the same observation and makes
every later chunk answer the previous one. That shows up as a fixed one-step
lag that never corrects itself, and it looks exactly like a bad policy.

The model also refuses to predict until it has received 4 complete observations
per echo it has consumed. A client publishing frames continuously, as
`RepeatingFrameTrack` does, satisfies that without doing anything special.

### What `predict()` does

The client is
[`reactor_robotics/xr1_robocasa365.py`](./reactor_robotics/xr1_robocasa365.py).
Three orderings matter:

- Frames and state before the echo. The model predicts only when it holds a
  complete observation set and a state history it can parse, and it snapshots
  both on the turn it predicts. If the echo lands first, the prediction comes
  from the previous observation.
- All three cameras together. The pairing is by arrival order, so a partial
  set is held, not consumed. Publishing one camera late shifts that view's
  whole history against the others.
- Drain before echoing. An answer to an earlier, timed-out request can still
  arrive late. Draining first keeps that stale chunk from being taken as the
  answer to this request.

`predict()` also keeps the echo counter, advancing it by the replan window, so
the caller never has to. Any strictly increasing sequence works.

## Check the results

The policy samples, so replaying an observation does not reproduce its chunk
and the returned numbers cannot be compared one-for-one. The check verifies:
correct shape, finite values, `step` always increasing, the first chunk of a
session at step 0, exactly one chunk per echo, and no stale chunk served as an
answer.

There is no anchor check here, and the reason is worth stating. This
checkpoint emits the vendor's packed layout decoded with per-step
relative-action statistics, so there is no absolute pose to anchor a chunk's
first row against. The substitutes are a per-step magnitude band over the live columns and the
run-to-run L2 band, both calibrated at recording time (the mean plus three
standard deviations over 3 passes of the same observations, see
[`examples/PROVENANCE.md`](./examples/PROVENANCE.md)).

The magnitude band is kept **per column**, one number for each of the 12 live
columns, rather than one number for the chunk. The gripper column swings the
full `[-1, 1]` range, so a single global band would be set by that column alone
and would pass anything the other eleven did. Per column, a smoothly moving
column gets a tight band and starts failing if a later deployment makes it
thrash.

Both bands are **reported, never gated**. This policy varies enough between
sessions that a
healthy run can graze a three-sigma bound, so gating it would fail runs where
nothing is wrong. A single column slightly over its band is normal; the same
column over on run after run is the signal, and a gross excursion shows up
plainly either way.

Without the recorded fixture the bands are unknown, so the same numbers are
printed with no band beside them: every tolerance here comes from a
measurement.

The script prints a row per check and then its verdict, which is the line to
read:

```
RESULT: PASS
```

After the verdict it runs the failure modes you are most likely to hit: a
non-increasing echoed `step` produces no chunk, a 25 s idle does not drop the
session, and `reset` starts a clean episode with the step counter back at 0.

## Latency

One chunk is 16 steps. The benchmark executes either the whole chunk or its
first 8 before asking again, so the number that decides whether this policy can
drive a robot in real time is how much of that window the round trip consumes.

The reference figure for this deployment is about 171 ms of model compute per
prediction on the hosted deployment. A gated lossless optimization brings that to about 71 ms
when `XR1_COMPILE_DIT` is enabled on the deployment, which is off by default;
it is `torch.compile` on the DiT flow loop, which is launch-bound rather than
compute-bound, and it was gated on a paired success-rate evaluation rather than
on the speedup alone. Both figures are reference points, not guarantees.

## Run the simulator

Everything above replayed synthetic frames, so none of it measures task
competence. That number comes from Xiaomi's own RoboCasa365 benchmark
(robosuite/MuJoCo), and the harness that measures it is in this repository:
[`../robocasa365/`](../robocasa365).

It runs the vendor's evaluation stack unmodified (rollout loop, task
registry, seeding, success criteria) and swaps only the client, so the
benchmark measures the same thing it measures over the vendor's own socket;
its README has setup, commands and failure modes.

Served over the production wire, this model matched the vendor's own transport
on the benchmark itself. Over 1000 paired episodes on the target50 task set,
seed-matched:

| leg | replan 16 | replan 8 |
|---|---|---|
| vendor TCP socket | 56.4% (282/500) | 59.2% (296/500) |
| **Reactor runtime (WebRTC)** | **56.8% (284/500)**, McNemar p=0.92 | **60.2% (301/500)**, p=0.66 |

External anchors vary by evaluation protocol: the
[paper](https://arxiv.org/abs/2607.15330) reports 57.6% on RoboCasa365, while
the [RoboCasa leaderboard](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/Xiaomi-Robotics_2026_07_08.md)
reports 57.1% Composite-Seen. The Reactor measurements above used the previous
runtime version.

### Commands

```
# 1. the simulator: a Xiaomi-Robotics-1 checkout with its RoboCasa365
#    install (robosuite/MuJoCo), per upstream's instructions

# 2. this client
cd ../robocasa365
uv run python check_wiring.py   # offline: no GPU, no network, no simulator

pip install -e .                # into the simulator's environment

# 3. the served model: any xr1-robocasa365 runtime you can reach, e.g. a
#    local `reactor run` from the model workspace

# 4. the evaluation, from the simulator's environment
MUJOCO_GL=egl python -m robocasa365_sim.entry \
    --vendor-dir ~/Xiaomi-Robotics-1 \
    --api-url http://127.0.0.1:8080 \
    --task-name CloseBlenderLid --num-trials 3 \
    --save-root-dir eval_results/reactor
```

All vendor knobs (`--task-set`, `--num-trials`, `--replan-steps`, `--seed`,
...) are accepted and forwarded; `--task-set target50` runs the full
benchmark. The summary lands in `<save-root-dir>/<run-id>/summary.json` with
a `transport` block recording the wire and per-inference latency.

### Expected output

```
INFO Writing evaluation outputs to eval_results/reactor/<run-id>
INFO Transport: Reactor SDK -> http://127.0.0.1:8080 (xr1-robocasa365)
INFO [reactor] connected; tracks published: left_agentview, right_agentview, wrist_view
CloseBlenderLid: 100%|##########| 3/3 [00:52<00:00, 17.4s/it]
INFO Success rate: 33.33% (1/3)  |  infer {'n': 171, 'p50_ms': 773.4, ...}
```

One line per task with a progress bar, one prediction per replan window
inside it, and the success summary at the end. A session that connects and
publishes the three tracks but never logs a prediction usually means the
echo never opened the gate; the [harness README](../robocasa365/README.md)
lists the failure modes.

### Requirements

| | |
|---|---|
| GPU | a CUDA GPU for MuJoCo EGL rendering (`MUJOCO_GL=egl`) |
| Simulator | a Xiaomi-Robotics-1 checkout with its RoboCasa365 install |
| Environment | the simulator's own environment; the client installs into it |
| Install time | dominated by RoboCasa365 and its assets |
| Wall-clock | ~15 to 25 s per episode plus an environment build per task, so target50 at 10 trials is a long run |

These requirements do not fit in a `pyproject.toml`, which is why they are
listed here rather than installed.


## Physical deployment

Two properties of this model shape what a hardware bring-up has to handle. Its
chunks are the vendor's packed layout, so a real arm needs the mapping from the
12 live columns to its own command space before anything moves. And each
session carries an observation history, so an episode boundary is a real event:
call `reset()` rather than relying on the window to flush.

To drive an arm:

1. Publish `left_agentview`, `right_agentview` and `wrist_view` from your three
   cameras, at the same track names and order used above, and publish them as a
   set so the views stay paired.
2. Send the arm's measured state as `state_history_json` (4 rows, oldest
   first) and report execution progress as `executed_step_json`, exactly as the
   script does. The wire does not change when the frames come from a camera
   instead of an `.npz`.
3. Confirm the action semantics against your embodiment before commanding
   anything. The checkpoint's statistics are per-step relative-action
   statistics for the data it was post-trained on, so a different robot or a
   different post-train changes what the 12 live columns mean.
4. Add the safety layer `predict()` does not have: joint clamps, a joint-limit
   margin, an action-semantics guard, and a starvation e-stop. Do not command
   an arm from the bare `predict()` above; it has none of these.
5. Follow [robot-policy-client-contract.md](./robot-policy-client-contract.md)
   for the wire rules while you build the rig.

Reactor support for physical deployment is coming soon.
