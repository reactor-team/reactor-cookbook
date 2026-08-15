# Where the recorded examples came from

> **Note for external readers.** This file is the maintainers' provenance
> record, kept so the fixtures are auditable rather than magic. Parts of it
> refer to Reactor-internal machines and to Reactor's private model
> repository; those parts are labelled **(internal)** and are not reachable
> from outside Reactor. Nothing in the notebooks requires them — the facts
> that matter (hashes, sizes, measurements, calibrations) are all recorded
> here in full.

The fixtures have different origins, which is why the notebooks check different
things. Each notebook claims only what its model's determinism supports.

| | Origin | Frames | Notebook claim | Why |
|---|---|---|---|---|
| `xwam_examples.npz` | **Subset** of a recorded upstream evaluation | Real RoboTwin 2.0 renders | Numeric replay gate, `5e-2` | Pipeline is seeded, so replay is meaningful |
| `dreamzero_examples.npz` | **Recorded** against the live deployment | Synthetic | Invariants + a calibrated loose band | Pipeline is **unseeded**, so no parity claim is possible |
| `lingbot_va_examples.npz` | **Recorded** against the live deployment | Synthetic | Invariants + **one exact numeric gate** (the pinned seed rows) + a calibrated band, reported only | Unseeded, but part of the seed chunk is *pinned* and therefore bit-reproducible |
| `groot_n17_examples.npz` | **Recorded** against the live deployment | **Real** FR3 rig captures | Invariants + two calibrated physical bands (anchor, per-step) + an L2 band, reported only | Unseeded; the absolute-anchoring gives a tight, meaningful gate |
| `cosmos_droid_examples.npz` | **Recorded** against the live deployment | Synthetic | Invariants + two calibrated physical bands (anchor, per-step) + an L2 band, reported only | Unseeded; same policy as `groot-n17` |
| `xr1_robocasa365_examples.npz` | **Recorded** against the live deployment | Synthetic | Invariants + per-column motion bands + an L2 band, reported only | Unseeded; packed relative-action layout has no absolute anchor |

Only `xwam` gets a numeric parity gate, because only `xwam` is seeded and only
`xwam` has reference outputs from another stack. Everywhere else a numeric gate
would be a tolerance nobody measured.

---

## `xwam_examples.npz` — 2.42 MB, 5 examples

### Where it came from **(internal — Reactor maintainers)**

```
lambda:/home/ubuntu/kavya/xwam/golden/goldens.jsonl   (Reactor dev machine)
  1 374 295 387 bytes, 372 rows
  sha256 1a12e168f255db8709d4b95a057e574d3d61614618c4b3cede2b4c151748059f
  lines used: 0, 1, 2, 3, 4   (the first five rows)
```

That file is the Phase-1 tee: one JSONL row per upstream request/response pair
captured while the **authors' unmodified** RoboTwin 2.0 eval client drove
their own serving stack, pickles base64-encoded. `expected_actions` is
therefore what the reference implementation produced for that exact request,
not a previous Reactor response, which makes replaying it a cross-stack check.

The tee is 1.37 GB and stays on that machine — it is not distributable, and
the subsetting step below is recorded for Reactor maintainers, not as
something an external reader can re-run:

```sh
# (internal) on the Reactor dev machine, in /home/ubuntu/kavya/xwam/
python3 make_notebook_goldens.py \
    --goldens golden/goldens.jsonl --out golden/xwam_goldens_5.npz --n 5
```

(The on-box script and its output keep their original `goldens` names; the file
copied into this repo is `examples/xwam_examples.npz`.)

Result: `sha256 ae55c5c498d572d413002256c92b191df5770dbe9f31b74d94767847065878f5`
(2 415 988 bytes), produced under numpy 1.26.4.

### The five rows

| example | line | env_rank | rollout_id | step_id | cfg | instruction | upstream latency |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0.0 | Pick up the brown bottle from the table head-up. | 67 424.6 ms |
| 1 | 1 | 0 | 0 | 32 | 0.0 | Pick up the brown bottle from the table head-up. | 242.0 ms |
| 2 | 2 | 0 | 0 | 64 | 0.0 | Pick up the brown bottle from the table head-up. | 244.1 ms |
| 3 | 3 | 0 | 0 | 96 | 0.0 | Pick up the brown bottle from the table head-up. | 234.9 ms |
| 4 | 4 | 0 | 1 | 0 | 0.0 | Grab the green plastic bottle with ridged bottom from the table with the left arm | 234.9 ms |

These five rows were chosen for their structure. Rows 0–3 are four consecutive
chunks of one rollout, and row 4 crosses an **episode boundary** into a
different rollout with a **different instruction**. That exercises
`set_task_description` changing mid-session and the `chunk_id` echo across a
reset-shaped transition, which a run of five same-episode chunks would not.

`(env_rank, rollout_id, step_id)` is the **seed triple**. Passing it to
`predict(..., seed=...)` pins the sampling noise to the recorded request; that
is what makes this a replay rather than a fresh sample. Robot clients omit it.

Row 0's 67 s upstream latency is the authors' cold start, not a steady-state
number. The other four (235–244 ms) are their warm serving latency. That is
not the origin of the "229 ms reference serving" figure, which comes from the
separate full measurement over n=371 requests.

### How the frames are stored

The tee's request pickle carries `video (3, 240, 320, 3) float32` in `[-1, 1]`,
ordered **positionally**. The served model takes three **named** tracks. The
mapping is fixed in the model's serving code and re-asserted by its parity
checker (both internal, in Reactor's private model repository):

```
video[0] -> head_view
video[1] -> left_wrist_view
video[2] -> right_wrist_view
```

The fixture therefore stores frames **keyed by track name**. A positional
fixture would feed a wrist view to the head track, which the model accepts
without complaint.

Frames are stored as the `uint8` the simulator actually rendered, recovered
with the inversion the gateway and parity checker both use:

```python
np.clip(np.round((video + 1.0) * 127.5), 0, 255).astype(np.uint8)
```

The client's normalization (`uint8 -> f32/127.5 - 1`) is bijective from
`uint8`, so this loses nothing. The subsetting script **asserts** it:
re-normalizing the recovered `uint8` reproduced the recorded float video
bit-for-bit for all five rows, or it would have refused to write the file.

### How the tolerance was chosen

The notebook gates executed actions at **`ACTION_TOL = 5e-2`** (max-abs). The
relevant floors:

- **`4.2e-3`** — direct-fed parity: recorded observations handed to the model
  as tensors, no video transport (max |Δ| 4.197e-3 over all 372 recorded
  requests, from the parity report in Reactor's private model repository —
  internal).
  **Not achievable through this notebook's path.**
- **`~4.4e-2`** — worst previously observed through the H.264 WebRTC path,
  which is what this notebook uses.

Measured on *these* five examples against PROD while building the notebook,
per example, max-abs action delta:

| run | fps | e0 | e1 | e2 | e3 | e4 | worst |
|---|---|---|---|---|---|---|---|
| driver 1 | 15 | 2.757e-2 | 2.447e-2 | 1.248e-3 | 7.135e-3 | 1.501e-3 | 2.757e-2 |
| driver 2 | 15 | 2.757e-2 | 2.893e-2 | 1.425e-3 | 5.850e-3 | 2.348e-3 | 2.893e-2 |
| fps probe | 15 | — | — | — | — | — | 2.757e-2 |
| fps probe | 30 | — | — | — | — | — | 3.160e-2 |
| notebook | 15 | 3.111e-2 | 2.400e-2 | 2.250e-3 | 9.307e-3 | 1.261e-3 | 3.111e-2 |
| notebook (re-run) | 15 | 3.111e-2 | 2.400e-2 | 2.250e-3 | 9.307e-3 | 1.225e-3 | 3.111e-2 |
| notebook (latency-note run) | 15 | 3.111e-2 | 2.400e-2 | 2.286e-3 | 9.813e-3 | 1.301e-3 | 3.111e-2 |
| notebook (pre-rename run) | 15 | 3.111e-2 | 2.400e-2 | 2.398e-3 | 9.217e-3 | 1.295e-3 | 3.111e-2 |
| notebook (post-rename run) | 15 | 2.319e-2 | **3.748e-2** | 2.183e-3 | 5.950e-3 | 3.342e-3 | **3.748e-2** |

**Worst observed across every run: `3.748e-2`**, so the `5e-2` gate carries
about 1.33× headroom.

**The gate was not tightened further.** Examples 0 and 1 sit at 2.3e-2–3.7e-2
while 2–4 sit at 1e-3–1e-2, and the larger two move run to run because the
H.264 encoding of an abruptly swapped observation is the only stochastic
element; the model itself is seeded by the triple. The last run illustrates the
spread well: example 0 fell to 2.319e-2 while example 1 rose to 3.748e-2, the
largest value seen in nine runs. A `3e-2` gate would already have failed that
run, and even `4e-2` would leave under 7% margin, so `5e-2` is the right level.
It is still ~30× tighter than the action scale a controller absorbs.

If a future run exceeds `5e-2`, check whether it is one of examples 0–1 before
concluding the model changed: those two carry the transport transient, and a
single-example value outside the band there is more likely a bad encode than
drift.

**Proprios are recorded but never gated.** Worst observed max |Δ| was
`1.50e+00`, which is the same transport transient without the action head's
smoothing. Nothing executes the predicted future states, so the notebook
reports them without checking them. (The internal parity report's direct-fed
proprio delta is already 5.6e-2 for the same reason.)

---

## `dreamzero_examples.npz` — 16.0 KB, 5 examples

Recorded **2026-08-11T07:38:12Z** against `dreamzero` on PROD
(`https://api.reactor.inc`), a 2× B200 deployment, by the committed script:

```sh
REACTOR_API_KEY=... uv run python examples/record_dreamzero_examples.py
```

No recorded corpus for DreamZero exists anywhere, so unlike X-WAM these
examples had to be created by driving the live model. The script is committed so the
artifact can be reproduced.

### The frames are synthetic

`synthetic_observation(i)` builds a deterministic five-step reaching sequence
(a tabletop, a marker travelling left-to-right, a gripper descending, and a
wrist view closing in). It is a pure function of `i`, using no RNG and no
clock, so re-running the script regenerates byte-identical inputs.

Real DROID frames were looked for and **none were appropriate**:

- The local DROID reference checkout carries only hardware-setup
  *photographs*: portrait framing, no wrist camera, no manipulation scene.
- The other local episode captures are RoboCasa kitchen renders of a
  different embodiment (Panda-Omron, 256×256).

Either would look like a real observation while being far outside the
checkpoint's distribution, so labelled synthetic frames are the more accurate
option.

**The recorded actions are therefore not a behavioural demonstration.** They
are a protocol and invariant fixture. Every check the notebook runs against
them (shape, finiteness, monotonic `obs_seq`, monotonic `chunk_index`, Franka
joint limits, gripper range, chunk-boundary continuity, the L2 band) is a
statement about the wire contract and the model's self-consistency, not about
task competence. Task competence is the **25.7% RoboLab success rate**, which
is NVIDIA's own leaderboard submission and was never re-run on Reactor. The
in-repo harness runs the same benchmark but did not produce that figure.

Two details of the fixture do reflect real deployment conditions. First,
**`exterior_2` is all black**, verified in the fixture, which is what the
evaluation harness sends: RoboLab's default `--cam2-source black` leaves the
second exterior slot black to match the checkpoint's training-time camera
dropout, so black is the *in-distribution* input for that view (implemented
in the evaluation harness's observation adapter — internal, in Reactor's
private model repository).

Second, the streamed joint state. Because `set_joint_position` carried a
plausible Franka ready pose, the returned joint targets are **absolute**
(j2 ≈ −0.35→−0.20, j4 ≈ −2.10→−1.90, j6 ≈ 1.64→1.81, j7 ≈ 0.75→0.89, all
clustered around the streamed pose and well inside the joint limits). With
zeros streamed instead, the model emits *relative deltas*, which changes the
meaning of the output without raising an error.

### What was recorded

Five observation → chunk pairs inside **one** episode:

| example | chunk_index | obs_seq | inference_seconds | client latency | discarded |
|---|---|---|---|---|---|
| 0 | 0 | 15 | 0.289 s | 624.8 ms | 0 |
| 1 | 1 | 29 | 0.232 s | 232.6 ms | 0 |
| 2 | 2 | 39 | 0.252 s | 254.0 ms | 0 |
| 3 | 3 | 51 | 0.263 s | 267.5 ms | 0 |
| 4 | 4 | 63 | 0.259 s | 256.2 ms | 0 |

`obs_seq` advances by ~10–14 per chunk, which is the model consuming
`frames_per_chunk: 4` frames on each of three cameras (4 × 3 = 12). It
therefore consumes essentially every frame the 15 fps tracks send.

Example 0's 625 ms client latency includes the client's own episode-start
settle (~267 ms), which the recording script waits out before setting the
prompt so chunk 0's warmup anchors on real frames rather than the black
placeholder the track opens with. That portion is not model latency.

### `inference_seconds` vs the advertised operating point

Across all 15 chunks of the three recording passes:

```
run0  0.289  0.232  0.252  0.263  0.259
run1  0.260  0.227  0.245  0.261  0.262
run2  0.256  0.224  0.242  0.261  0.258

median 0.258 s   mean 0.253 s   min 0.224 s   max 0.289 s
```

**Median 258 ms against a 267 ms advertised operating point**, so the
advertised figure holds. This is recorded explicitly because an earlier smoke
run on this deployment measured ~314 ms median while contended, which reflected
contention rather than the operating point.

### Run-to-run spread, and the band it sets

DreamZero's pipeline is **unseeded**, so replaying an observation does not
reproduce its actions and there is nothing to gate numerically. Instead the
recording script replayed the identical five-observation sequence **twice
more** in fresh episodes and measured the per-chunk L2 distance between runs:

```
runs 0-1   0.1670  0.0708  0.0748  0.0924  0.0929
runs 0-2   0.1506  0.0725  0.0789  0.0973  0.1061
runs 1-2   0.0691  0.0635  0.0681  0.1021  0.1018

n = 15    mean 0.09385    std 0.03004    max observed 0.16697
band = mean + 3 sigma = 0.18397
```

That `0.184` is what the notebook uses, stored in the fixture as `l2_band`
(with `l2_band_mean` / `l2_band_std` alongside). For scale, the chunks
themselves have L2 norms of 13.29–14.27, so the band is ≈1.3% of chunk
magnitude. The model is therefore substantially more reproducible than
"unseeded" suggests, though still not bit-reproducible, and the notebook states
this.

A failure against this band means the deployment's behaviour moved well outside
its own measured run-to-run spread. It works as a drift detector rather than a
parity check.

**The notebook reports the band check but does not fail on it**, for a measured
reason. Across the notebook verification runs the worst per-chunk L2 came out
at:

```
run 1   0.1537   (inside 0.18397)
run 2   0.2108   (OUTSIDE)
```

A bound at the mean plus three standard deviations, estimated from 15 samples,
is exceeded occasionally by construction, and the second run showed that on
live traffic. Making it a hard
gate would give F&F users a spurious FAIL, while leaving it out of the fixture
would discard a useful drift signal. It is therefore checked, printed with an
explicit `OUTSIDE BAND` marker, and excluded from the PASS/FAIL verdict, which
is carried entirely by the deterministic invariants (shape, finiteness,
monotonic `obs_seq`/`chunk_index`, joint limits, gripper range, boundary
continuity). Tightening it would require recalibrating over many more passes;
15 samples supports an order-of-magnitude sanity bound rather than a control
limit.

### Re-recording

Re-running `record_dreamzero_examples.py` re-records the examples **and
re-calibrates the band**, keeping the two consistent. Do that after any
change to the deployment, and update the tables above. Note that DreamZero
holds two B200s for the life of a session, so a busy cluster answers with
`429 no available capacity`; the script retries that automatically (it took
two 60 s retries on the recording run above, then reported READY 126.7 s
after the successful session was created).

---

## `lingbot_va_examples.npz` — 9.09 kB, 5 examples

Recorded **2026-08-11T09:37:04Z** against `lingbot-va` on PROD
(`https://api.reactor.inc`), a 1× B200 deployment (READY in 5.9 s), by the
committed script:

```sh
REACTOR_API_KEY=... uv run python examples/record_lingbot_va_examples.py
```

### The frames are synthetic

`synthetic_observation(i)` builds a deterministic five-step reaching sequence at
**128×128**, LIBERO's native camera size: a tabletop with a bowl and a plate
from a fixed third-person camera (`agentview`) plus a wrist close-up
(`eye_in_hand`). Pure function of `i` — no RNG, no clock — so re-running the
script regenerates byte-identical inputs.

Real LIBERO renders would be better and are **not cheaply obtainable**: LIBERO
is not on PyPI, needs a from-source install plus asset downloads and
`torch<2.6`, and no recorded LIBERO clip is vendored anywhere. Passing off
another embodiment's frames as LIBERO observations would be less honest than
labelled synthetic ones.

**The recorded actions are therefore not a behavioural demonstration.** They are
a protocol and invariant fixture. Task competence is the published **98.5% on
LIBERO-Long**, measured in the real simulator — which
[`../../libero/`](../../libero) in this repo runs against this same
model. To record against real renders instead, run that harness with
`--record out.mp4`.

### What was recorded

Five observation → chunk pairs in one episode:

| example | step | executable rows | latency (gate → chunk) |
|---|---|---|---|
| 0 | 0 | 12 | 190.3 ms (gate = `reset`) |
| 1 | 1 | 16 | 208.4 ms (gate = echo) |
| 2 | 2 | 16 | 203.8 ms |
| 3 | 3 | 16 | 214.5 ms |
| 4 | 4 | 16 | 243.2 ms |

`step` is the model's own inference counter, restarting at 0 per episode — not
an echo. The echo row counts across all three passes were
`[12, 16, 16, 16] × 3`, and `echo_duplicates` was **0**.

### How long each chunk took

The two gates measure different things and are never pooled.

```
steady state (echo -> chunk), n = 12
  run0  208.4  203.8  214.5  243.2
  run1  239.3  211.9  203.8  204.3
  run2  212.2  200.2  207.2  205.6
  p50 207.8   min 200.2   max 243.2

seed chunk (reset -> chunk), n = 3
  190.3   195.2   227.4        p50 195.2
```

**p50 207.8 ms per chunk.** The seed number also contains the server gathering
its first 12 frames, so it is not comparable.

On top of that, **each observation costs a 1.00 s frame-window hold** (20 frame
periods at 20 fps). That is the model's frame budget rather than its latency: the
server commits a window of video frames per chunk — nominally 16
(`frame_chunk_size` 4 × `action_per_frame` 4, from the model's internal config —
internal), 12 on the seed chunk — and the committed window ranges over
**14–19 frames** in practice because nothing ties the data channel's echo to
the video track. A real client does not
pay the hold: it is executing the previous chunk while its cameras fill the
window. A LIBERO rollout at 20 Hz renders exactly 16 frames while executing 16
actions.

### The pinned seed rows are the one exact check

An episode's first chunk has its leading action latent **pinned to normalized
zero as a conditioning slot** rather than predicted. Measured: its first
`action_per_frame` = **4 rows are bit-identical to each other and bit-identical
across all three passes and across separate sessions**, at

```
[0.120536, 0.005358, 0.0, 0.025179, 0.012322, 0.039108, 0.0]
```

That is stored as `seed_pinned_row` and the notebook gates on exact equality.
It is the only part of this model's output a replay can gate numerically, and it
is also the evidence for `SEED_SKIP_STEPS = 4`: executing those rows commands
the quantile midpoint, which is real motion (`dx` +0.12 four times over).

### Run-to-run spread — reported, never gated

The pipeline is unseeded. The script replayed the identical five-observation
sequence twice more in fresh episodes and measured per-chunk L2 between runs:

```
runs 0-1   0.7595  1.0518  1.3325  1.1094  0.8830
runs 0-2   0.5357  0.4703  1.2130  2.1663  1.3688
runs 1-2   0.7627  1.1682  1.9895  2.3832  1.7468

n = 15    mean 1.07951    std 0.36496    max observed 1.70370
band = mean + 3 sigma = 2.17441
```

Chunk L2 norms are 3.58–4.17, so **the band is ~55% of chunk magnitude**. At
that width it detects gross drift and nothing finer, so the notebook prints it
and **excludes it from the PASS/FAIL verdict**. The verdict is carried by the
deterministic checks: the pinned seed rows, shape, finiteness, `step`
monotonicity, the 12-then-16 executable row pattern, membership of the training
quantile box, and zero echo duplicates.

The quantile-box check is the action-semantics guard: rows arrive as **deltas in
raw LIBERO action units** (the server already un-normalized them through the
training quantiles), and deltas sit inside the box while absolute poses would
not. Measured worst excursion outside the box was **0.008** on the gripper
channel, so the notebook allows `±0.05` slack.

### Verification runs

The notebook was executed top to bottom against PROD on 2026-08-11, twice.

| | run 1 | run 2 |
|---|---|---|
| Verdict (11 invariants) | **PASS** | **PASS** |
| Steady-state p50 | 210.5 ms | 227.6 ms |
| Seed chunk | 188.4 ms | 189.1 ms |
| Pinned seed rows bit-exact | yes | yes |
| Worst band L2 | 1.0581 (inside 2.174) | **3.9263 (OUTSIDE)** on chunk 4 |

Both runs also passed the negative tests: a byte-identical echo produced no
chunk in 12 s, and 25 s of client silence did not drop the session (the 10 s
keepalive in `session.py` carried it).

**Run 2 landing outside the band is the reason the band is not a gate**, and it is worth
recording rather than smoothing over. 3.93 is ~97% of chunk magnitude — the
run-to-run spread on this model can be as large as a chunk itself. A bound at the
mean plus three standard deviations, estimated from 15 samples of a
distribution this wide, is exceeded routinely, not rarely. Had it been a gate, the second run of this notebook would
have handed an F&F user a spurious FAIL. Tightening it is not the fix either;
the honest statement is that **this model's chunk-level output is not
reproducible enough for any numeric band to be meaningful**, which is why the
verdict rests on the pinned seed rows, the quantile box, and the structural
invariants.

### Re-recording

Re-running the script re-records the examples **and re-calibrates the band and
the pinned row**, keeping them consistent. Do that after any change to the
deployment, and update the tables above.

---

## `groot_n17_examples.npz` — 735 016 bytes (0.74 MB), 5 examples

`sha256 7007b81b05275baac0dcf16b1401e3cc4cbd51d6a487ca15d43dd89ca1100008`

Recorded **2026-08-11T09:43:17Z** against `groot-n17` on PROD
(`https://api.reactor.inc`), a 1× B200 deployment (READY in 6.0 s):

```sh
REACTOR_API_KEY=... uv run python examples/record_groot_n17_examples.py
```

### The frames are real

`exterior_view` and `wrist_view` are **real captures from a Franka Research 3
rig** running the DROID camera layout at 180×320 — the embodiment this
checkpoint targets, so they are in distribution. Scene: a tabletop with a plate,
toy fruit, a blue block and a green bowl; task string
`"Put the blue block in the green bowl"`.

The source capture is a **Reactor-internal artifact (internal)**. The frames
themselves are committed inside the fixture, so the recording script **re-runs
from the fixture alone** and needs nothing internal:

```sh
uv run python examples/record_groot_n17_examples.py                 # from the fixture
uv run python examples/record_groot_n17_examples.py --from-capture PATH   # maintainers
```

### Where the streamed state comes from

The capture stores frames and chunks but no `state_json`. Each observation's
state is recovered from the **reference chunk's first row**: this model's row 0
is its absolute-anchored prediction for the state it was handed, so row 0 is
that observation's proprioception to within the anchoring error — measured below
at ≤0.055 rad at recording time. The recovered `joint_position` (7),
`eef_9d` (9) and `gripper_position` (1) vectors are committed in the fixture,
so nothing is derived at run time.

### What was recorded

| example | step | anchor max abs (rad) | latency | discarded |
|---|---|---|---|---|
| 0 | 13 | 0.02261 | 157.9 ms | 13 |
| 1 | 28 | 0.02799 | 102.6 ms | 14 |
| 2 | 44 | 0.03801 | 199.9 ms | 15 |
| 3 | 59 | 0.03157 | 106.1 ms | 14 |
| 4 | 74 | 0.05483 | 109.7 ms | 14 |

`step` climbs by ~15 per observation because the model **free-runs**: it emits a
chunk every engine tick regardless of whether anyone asked, so the gate discards
the ~14 produced during the 1.40 s frame-window hold. `step` is an inference
counter, restarting at 0 on `reset`.

### The p50 is the gate's cost, not the model's compute

```
end of frame-window hold -> chunk (one chunk discarded first), n = 15
  run0  157.9  102.6  199.9  106.1  109.7
  run1  127.7  102.6  199.8  200.0  106.7
  run2  197.0  200.0  199.7  199.8  198.1
  p50 197.0   min 102.6   max 200.0

free-running chunk period, measured separately: 98.4 ms
```

The p50 is **the gate's cost, not the model's compute**: one discarded chunk
plus the next is ~1.5 chunk periods. The model's own cadence is the 98 ms
period. For comparison, a reference run directly against the model on the GPU
box (not the served path) measured **125–130 ms warm** per chunk, with a
1589 ms cold first chunk (`capture_reference_latency_ms` in the manifest, from
the same capture). Its parity check against the served path has not passed.
On top of the p50 each observation costs the **1.40 s frame-window hold**; a
robot never pays it, because its cameras fill the window while it executes the
previous chunk.

### How the bands were measured

**Anchor tolerance — the check that catches broken wiring.** The server converts
the checkpoint's relative prediction to absolute using the `state_json` of the
same tick, so `joint_position[0]` must sit near the state that was sent. Over 15
samples (3 passes × 5 observations), `max|joint_position[0] − streamed joints|`:

```
0.02261 0.02799 0.03801 0.03157 0.05483
0.02546 0.02120 0.03582 0.05407 0.04212
0.02141 0.02516 0.04083 0.04696 0.03699

mean 0.03500   std 0.01126   max observed 0.05483
band = mean + 3 sigma = 0.06878 rad
```

The notebook **gates** on this. It is tight, physically meaningful, and it is
what fails if the streamed state is lost — in which case the rows silently
become relative deltas rather than poses.

**Per-step joint delta.** `max|joint_position[k+1] − joint_position[k]|` per
chunk, over the same 15 samples:

```
mean 0.04056   std 0.00721   max observed 0.05936
band = mean + 3 sigma = 0.06219 rad
```

The notebook **gates** on this too. Reference: Reactor's FR3 client clamps at
**0.05 rad/tick** (0.75 rad/s at 15 Hz — internal). The measured max of 0.05936
is above that, so a chunk is *mostly* commandable at 15 Hz unclamped but the
clamp does engage sometimes — keep it.

**Run-to-run L2 on `joint_position` — reported, not gated.**

```
n = 15   mean 1.21950   std 0.44320   max observed 1.91920
band = mean + 3 sigma = 2.54918
```

Chunk L2 norms are 16.03–17.98, so the band is **~15% of chunk magnitude** —
much more informative than a full-chunk band on a delta-space model, but still a
three-standard-deviation bound from 15 samples, so an occasional value outside
the band is expected by construction
and it stays outside the verdict.

### Verification runs

The notebook was executed top to bottom against PROD on 2026-08-11, twice.

| | run 1 | run 2 |
|---|---|---|
| Verdict (8 invariants) | **PASS** | **PASS** |
| Worst anchor | 0.0560 rad | 0.0451 rad (band 0.0688) |
| Worst per-step \|Δq\| | 0.0482 rad | 0.0480 rad (band 0.0622) |
| Worst band L2 | 2.2350 | 1.8108 (band 2.5492) |
| Measured chunk period | 100 ms | 102 ms |
| p50 (gate cost) | 110.9 ms | 137.4 ms |

Both bands held on both runs, unlike `lingbot-va`'s — the anchor band in
particular is tight (0.069 rad) and never came close to failing, which is what a
useful calibrated gate looks like.

Negative tests behaved as documented on both runs: a malformed `state_json` did
**not** stop the model — it kept predicting, confirming the bad key degrades to
zeros rather than being rejected — 25 s of client silence did not drop the
session, and `reset` restarted the inference counter.

### Re-recording

Re-running the script re-records the chunks **and re-calibrates all three
bands**. Do that after any change to the deployment, and update the tables
above.

---

## `cosmos_droid_examples.npz` — 22 050 bytes, 5 examples

`sha256 f4332fbb1d5e681eaeada42e87a3f7acfcf8995f6a94ab199e5c641b426cdc90`

Recorded **2026-08-12T23:45:59Z** against `cosmos-nano-policy-droid` on PROD
(`https://api.reactor.inc`), a 1× B200 deployment (READY in 9.9 s), serving
release **0.2.1**:

```sh
REACTOR_API_KEY=... uv run python examples/record_cosmos_droid_examples.py
```

The deployment context matters for this one **(internal)**: release 0.3.0
(the runtime-3.1.2 port) had been rolled back to 0.2.1 roughly 40 minutes
before this recording, after shipping a serving regression in which sessions
connected but no prediction was ever answered (REA-5103 / REA-5115). These
bands therefore describe **0.2.1's** behaviour, and the run-to-run L2 signal
is exactly the drift check to watch when a fixed 3.x port redeploys.

### What was recorded

| example | step | anchor max abs (rad) | latency | discarded |
|---|---|---|---|---|
| 0 | 0 | 0.0510 | 659.0 ms | 0 |
| 1 | 1 | 0.0228 | 783.8 ms | 0 |
| 2 | 2 | 0.0452 | 830.2 ms | 0 |
| 3 | 3 | 0.0816 | 809.5 ms | 0 |
| 4 | 4 | 0.0691 | 782.5 ms | 0 |

Plus two more passes of the same five observations for calibration: 15
samples total, latency p50 783.8 ms, min 659.0, max 835.7 — against the
2133 ms chunk budget and the ~745 ms reference p50.

### Calibrated bands (mean + 3σ over 3 passes, n = 15)

| Band | mean | std | band | max observed |
|---|---|---|---|---|
| row-0 anchor (rad) | 0.0626 | 0.0219 | **0.1282** | 0.0938 |
| per-step joint delta (rad) | 0.0570 | 0.0109 | **0.0896** | 0.0844 |
| run-to-run L2 | 1.698 | 0.592 | **3.4731**, reported only | 2.944 |

The pipeline is unseeded, so the L2 band is the model's own run-to-run
spread and is a drift signal, reported without gating — same policy as
`groot-n17`. The per-step band is tight: a 3σ bound from 15 samples will
occasionally be grazed by construction (a later green run measured worst
0.0878 against the 0.0896 band).

### Re-recording

Re-running the script re-records the chunks **and re-calibrates all three
bands**. Do that after any change to the deployment, and update the tables
above.

### What was checked offline, before any live session **(internal — Reactor maintainers)**

Established on 2026-08-11, before the model had capacity to serve a session
(session creation answered HTTP 429 for ~15 minutes of attempts that day).
Recorded here rather than in the notebook, which states wire facts and not
the evidence for them:

- The wire contract in the notebook — three track names, the three command
  names, the `action_prediction` envelope — checked against the model's own
  served definition.
- Every encode/decode path exercised offline, including that a worst-case 32×8
  `executed_step_json` echo serialises to 5102 characters against the field's
  8000-char limit. That bound is a wire fact and is stated in the notebook where
  the echo is explained.
- The `synthetic_observation()` fallback above, so every observation cell runs
  without a committed fixture.

### The frames are synthetic

This checkpoint takes **three** views (wrist + two exterior). The real DROID
captures available locally come from a two-camera rig, and duplicating one
exterior view into the second slot would misrepresent a second camera. So
`synthetic_observation(i)` builds a deterministic five-step reach with three
genuinely distinct views at 180×320: a left-side exterior, a right-side exterior
with mirrored geometry and dimmer light, and a wrist close-up. Verified offline
to be distinct from each other and byte-identical across runs.

Task competence comes from the real simulator instead, and
[`../../cosmos-droid/`](../../cosmos-droid) in this repo runs it:
**40.0% over 120 tasks × 3 rollouts served in-process** (a Reactor
measurement) **against NVIDIA's published 39.7%**, and **8/10 solves over the
production wire** with a p50 of 745 ms
think+wire and 0 stalls in 150 chunks (internal port report — internal; also
quoted in that example's README).

### What was verified offline on 2026-08-11

| Check | Result |
|---|---|
| `encode_proprio` round trip | 125 chars for one timestep |
| `encode_proprio` rejects a wrong joint width | raises |
| `encode_proprio` rejects non-finite | raises |
| `encode_executed_step` rejects 1-D rows | raises |
| Worst-case 32×8 float64 echo | **5102 chars** against the field's 8000 limit |
| `_decode` accepts `[32,8]` | ok |
| `_decode` rejects a wrong width / non-finite | discards both |
| Three synthetic views distinct + deterministic | yes |
| Chunk budget | 2133.3 ms (32 rows at 15 Hz) |

### Reference timings to compare against

| | Value | Source |
|---|---|---|
| chunk budget | 2133 ms | 32 rows at 15 Hz |
| model compute per chunk | ~568 ms | internal port report (internal) |
| p50 think + wire | ~745 ms | internal port report (internal); also in `../../cosmos-droid/README.md` |

## `xr1_robocasa365_examples.npz` - 19,246 bytes, 5 examples

`sha256 0d28d4e701f4d02ea507da124ce6140cda0783d6a0a74cf0e66dd14ce1f56528`

Recorded **2026-08-13T21:04:37Z** against `reactor/xr1-robocasa365` on DEV
(`https://api.rea.live`), READY in 10.1 s, serving
release **0.2.0**:

```sh
REACTOR_API_KEY=... uv run python examples/record_xr1_robocasa365_examples.py
```

### The frames are synthetic

A deterministic five-step kitchen approach at the benchmark's native 256x256,
with three genuinely different views: two agentview cameras from opposite sides
and a wrist close-up. `synthetic_observation(i)` is a pure function of `i`, so
re-running regenerates identical inputs.

The recorded actions are a protocol fixture, not a behavioural demonstration.
Task competence comes from the RoboCasa365 benchmark itself, where this
contract scored 56.8% and 60.2% at replan 16 and 8 against 56.4% and 59.2% for the
vendor's own TCP socket over 1000 paired episodes; the vendor's published
anchor is 57.28%.

### What was recorded

| example | step | max abs step delta | worst column | latency | discarded |
|---|---|---|---|---|---|
| 0 | 0 | 2.0000 | 11 | 356 ms | 0 |
| 1 | 1 | 0.3750 | 0 | 267 ms | 0 |
| 2 | 2 | 0.2168 | 0 | 272 ms | 0 |
| 3 | 3 | 0.1719 | 2 | 262 ms | 0 |
| 4 | 4 | 0.0879 | 0 | 281 ms | 0 |

Plus 4 more passes of the same five observations, **each in its own session**:
25 samples per column, latency p50 281 ms, p95 343, max 356.

### Calibrated bands (reported, never gated)

Per column, `min(mean + 3σ, observed span)` over 25 samples:

| col | mean + 3σ | observed span | band | from |
|---|---|---|---|---|
| 0 | 0.8952 | 1.6289 | **0.8952** | 3σ |
| 1 | 0.3640 | 0.6406 | **0.3640** | 3σ |
| 2 | 0.4042 | 0.9727 | **0.4042** | 3σ |
| 3 | 0.0431 | 0.1514 | **0.0431** | 3σ |
| 4 | 0.4268 | 0.7617 | **0.4268** | 3σ |
| 5 | 0.0473 | 0.0894 | **0.0473** | 3σ |
| 6 | 3.4344 | 2.0391 | **2.0391** | span |
| 7 | 0.5751 | 1.3008 | **0.5751** | 3σ |
| 8 | 0.4690 | 1.0312 | **0.4690** | 3σ |
| 9 | 0.1026 | 0.3145 | **0.1026** | 3σ |
| 10 | 0.0159 | 0.0244 | **0.0159** | 3σ |
| 11 | 3.7119 | 2.0156 | **2.0156** | span |

Run-to-run L2: mean 3.982, σ 2.028, band **10.068**.

Three properties of this band, and why each is needed:

1. **Per column, not one number.** Columns 6 and 11 swing the full `[-1, 1]`
   while others creep at 0.02. One global band would sit at ~2.0 and be blind
   to the rest.
2. **Clamped to the span.** A plain mean + 3σ bound can exceed a column's
   entire observed range, which makes the check unfailable. Several columns
   are bimodal, where a Gaussian bound is meaningless.
3. **One session per pass.** This model varies more between sessions than
   within one, and the check always runs in a fresh session, so the
   calibration has to sample across sessions too.

The bands are **reported, not gated**: healthy runs graze them, and gating
would fail runs where nothing is wrong.

### What is gated

Shape `(16, 60)`, all values finite, `step` strictly increasing, first chunk at
step 0, exactly one chunk per echo, no stale chunk served. Then three probes: a
non-increasing echo yields no chunk, a 25 s idle does not drop the session, and
`reset` restarts the episode at step 0.

### Re-recording

Re-running the script re-records the chunks and re-calibrates both bands. It
opens 5 sessions, so it needs the dev deployment free for a few minutes; a busy
deployment answers `429 no available capacity` and the script retries.
