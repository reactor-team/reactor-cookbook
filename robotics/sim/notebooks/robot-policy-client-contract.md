<!--
  ─────────────────────────────────────────────────────────────────────────
  MAINTAINERS ONLY — none of this block renders on GitHub.

  This file is the reader-facing contract for driving Reactor-served robot
  policies. Nothing in the rendered body refers to Reactor-internal
  documents, repositories, or tracker ids, and nothing should: a reader with
  this repository and an API key must be able to treat the body as complete
  and authoritative. Keep it that way.

  SOURCE. Adapted for this repository from Reactor's internal robot-policy
  contract doc — the engineering source of truth for the wire facts — at
  docs/robot-policy-client-contract.md in the private model repository. When
  re-syncing, diff against upstream commit 46c859e9c389 (2026-08-11, the
  commit that introduced the contract), then re-apply the deltas below,
  dropping any the upstream doc has since absorbed. Fix contract *facts*
  upstream first and bring them here; wording, headings and reader-facing
  framing are this file's own.

  DELTAS vs that upstream commit:
    1. Reader-facing framing: plain-English headings, prose tightened section
       by section (no fact added or dropped), and the "If a model disagrees
       with this document" section in place of upstream's precedence note
       ("where this text and X-WAM disagree, X-WAM is right and this text is
       a bug"). Upstream keeps that note, which is correct for an audience
       that can read xwam/ and diff it against the text.
    2. Internal references removed rather than labelled. Upstream's
       private-repo paths and bare tracker ids are gone from the body: the
       keepalive example points at this repository's
       reactor_robotics/session.py, and the conformance checklist asks for
       published behaviour documentation without naming model_behaviour.md.
    3. Three extra rows in the per-model table: dreamzero,
       cosmos-nano-policy-droid, and lingbot-va filled in. Upstream has no row
       for the first two and reads TBD in every column for the third; getting
       them there is tracked separately. Each records observed wire behaviour,
       not a conformance commitment.
    4. Extra detail in the groot-n17 Notes column (free-running; a malformed
       state key degrades to zeros), observed on the live deployment. Its
       numbers — tracks, proprio 17, action 17, K 40 — are upstream's own,
       re-checked against the model's source and unchanged.

  TRACKER IDS, kept here and never in the body: REA-5036 this contract;
  REA-5037 the SDK/runtime gaps the body states behaviourally (silent
  publish_track before READY, client-side keepalive pings, frame
  watermarking); REA-5038 the generic client notebook.
  ─────────────────────────────────────────────────────────────────────────
-->

# Generic robot policy client contract

## Same client for a simulator or a physical robot

Every Reactor robotics policy uses the same session transport: named video
tracks in and action chunks back on the data channel. This document defines
the generic **one request at a time** policy contract implemented by X-WAM.
The other hosted models use the same transport but expose model-specific state,
request, and reply messages; the table below records each departure.

Simulator and physical-robot controllers can share the wire client. Hardware
still needs action mapping, motion limits, stale-chunk rejection, and a
watchdog/e-stop; those controls are outside this transport contract.

## If a model disagrees with this document

Use this document directly for X-WAM and any future model that passes the
contract-only checklist below. The clients in
[`reactor_robotics/`](./reactor_robotics) are verified against the live models;
X-WAM demonstrates every generic rule, while the other clients demonstrate
their documented model-specific variants.

If served behaviour ever disagrees with this text, **trust the served
behaviour** and report the mismatch to your Reactor contact, so the text gets
fixed.

## Running a session without silent hangs

1. **Register your status and message handlers, then connect** with the
   Reactor SDK (`reactor-sdk`).
2. **Wait for `READY`.** Status goes `CONNECTING` → `WAITING` → `READY`
   asynchronously after `connect()`. `publish_track` before `READY` does
   nothing at all (no error, no track) and the model waits forever for frames
   that never arrive. Await the status event; do not sleep and hope.
3. **Publish one track per view the model declares**, using its exact track
   names.
4. **Keep the session alive.** The runtime kills a connection after 20 s of
   client silence, and some SDK builds do not ping for you. Send a
   runtime-scope `ping` at least every 10 s for the whole session, including
   while your robot is executing a chunk and sending nothing else.
   [`reactor_robotics/session.py`](./reactor_robotics/session.py) is a working
   keepalive loop to copy.
5. **Set the task** once per episode, then loop over requests.
6. **Reset or reconnect freely when the model implements this stateless
   contract.** A prediction then depends only on the frames and state that came
   with its request. Stateful model variants document their own reset rules.

## What you send

**Video tracks.** Each model declares its own named views (X-WAM:
`head_view`, `left_wrist_view`, `right_wrist_view`). Names and order come from
the checkpoint's training-time camera order, so a wrist frame published on the
head track produces a wrong prediction, with no error raised. Any resolution
works, because the model does its own preprocessing. Send frames at a steady
10-30 fps and
**keep repeating the current observation between requests**, so every view has
a frame at least as new as your next request.

**`task_description`** is the episode's language instruction, sent once per
episode with `set_task_description`. Models answer no requests until it is set.

**`state_json`** carries one prediction request per update, sent with
`set_state_json`:

```
{"proprio": [/* N floats */], "chunk_id": 3, "cfg": 0.0}
```

| Field | Required | Meaning |
|---|---|---|
| `proprio` | yes | Current robot state, N floats in the model's documented layout. Must be finite; a malformed request is dropped rather than zero-filled, because a fabricated state would command a real arm. |
| `chunk_id` | yes | Your request id. The reply echoes it as `step`. |
| `cfg` | no (default `0.0`) | Guidance scale. |
| `env_rank`, `rollout_id`, `step_id` | no (defaults `0`, `0`, `chunk_id`) | Fix the sampling noise seed exactly. Send them **only** to replay a recorded evaluation request exactly; robot clients omit them. |

Those defaults make the seed a pure function of the request's own content, so a
robot client still gets distinct noise per chunk and identical answers for a
re-sent request.

## One request at a time, and how to recover a lost reply

- **One prediction per distinct `state_json` value.** An identical re-send
  is deduplicated: the model cannot tell it from the continuous re-delivery of
  unchanged state, so it produces **no** second reply.
- **To recover a lost reply, re-send with the same `chunk_id` and any byte
  changed**, by convention a bumped `retry` counter, which the parser ignores
  like any unknown field. The noise seed is a pure function of the request's
  seed fields (the explicit triple, or the `chunk_id`-derived default), so the
  retried reply is identical to the one you lost, and the model holds no
  state that retrying could corrupt.
- **A reply comes only once every view has delivered a frame that arrived
  after the request.** So push your fresh frames first, give them a few track
  periods to clear the encoder, *then* send the request. Nothing tags a chunk
  with the frame it came from, so this ordering is the only thing pairing an
  observation with its reply.
- **One request outstanding at a time.** Do not pipeline.

## What you get back

Replies arrive on the data channel as:

```
{"type": "action_prediction",
 "data": {"actions": [/* K × A */], "proprios": [/* … */], "step": 3}}
```

- `actions`: `[K, A]`, one row per control step, in the model's documented
  action layout.
- `proprios`: the model's predicted future robot states, same layout as the
  request's `proprio`.
- `step`: echo of your `chunk_id`. **Discard any reply whose `step` does not
  match your outstanding request**; a stale reply crossing an episode reset is
  the classic harness bug.

Execute the chunk, then request again with the next `chunk_id`.

## Track names, shapes, and where each model departs from this contract

| Model | Video tracks | proprio N | action dim A | chunk K | Notes |
|---|---|---|---|---|---|
| `xwam` | `head_view`, `left_wrist_view`, `right_wrist_view` | 16 | 14 | 32 | Delta joint actions, bimanual; RoboTwin 2.0 SFT. The reference implementation. [Guide](./xwam_quickstart.md). |
| `groot-n17` | `exterior_view`, `wrist_view` | 17 | 17 | 40 | Free-running: it predicts every engine tick rather than once per request. `state_json` is a dict of named vectors (`eef_9d` 9 + `gripper_position` 1 + `joint_position` 7), `actions` split across those same three fields, and `step` is an inference counter. A malformed `state_json` key becomes **zeros** instead of dropping the request. Absolute joint targets, converted server-side. [Guide](./groot_n17_quickstart.md). |
| `lingbot-va` | `agentview`, `eye_in_hand` | **none**; the observation is the video | 7 | 16 | Lock-step, driven by an **executed-action echo** (`set_executed_action_json`) rather than `state_json`, and the echo must change value to signal. Replies carry a single `action` `[16,7]` field and a `step` that is an inference counter. Actions are 6 end-effector deltas + gripper in raw LIBERO units. `reset` takes an **empty** payload. An episode's first chunk pins its leading 4 rows: execute 12 of 16. [Guide](./lingbot_va_quickstart.md). |
| `cosmos-nano-policy-droid` | `wrist_view`, `exterior_view_1`, `exterior_view_2` | 8: 7 joints + gripper, sent as `set_proprio_json` row lists (**not** `state_json`), last row = current | 8 | 32 | Lock-step, one chunk per request, and **stateless per prediction**: no KV cache and no `reset` event on the wire. The gate is `set_executed_step_json` (`{"step": int, "action": [[...]]}`), whose `step` must **strictly increase**. Reply `step` is the model's own prediction counter from 0. Absolute joint targets + gripper. [Guide](./cosmos_droid_quickstart.md). |
| `dreamzero` | `exterior_1`, `exterior_2`, `wrist` | 7 joints + gripper, sent as `set_joint_position` + `set_gripper_position` (**not** `state_json`) | 8 | 24 | Free-running broadcast whenever the cameras deliver fresh frames, so there is no request to echo and no `chunk_id`. Task is set with `set_prompt`, replies are `action_chunk`, paired to an observation by an `obs_seq` high-water-mark gate. Absolute joint targets + gripper. [Guide](./dreamzero_quickstart.md). |
| `xr1-robocasa365` | `left_agentview`, `right_agentview`, `wrist_view` | 14 per row, **4 rows**, sent as `set_state_history_json` (**not** `state_json`), oldest first, 2 env steps apart | 60 packed, first 12 live | 16 | Lock-step, but unlike the others the **first** request needs an echo too: `set_executed_step_json` (`{"step": int}`, step alone, no rows) whose `step` must strictly increase. An asymmetric first prediction races the first echo and yields a permanent one-step lag. The model waits for 4 complete observations per consumed echo, and pairs the three camera tracks frame-for-frame before sampling its history. Reply `step` is the model's own prediction counter from 0. Actions are the vendor's packed layout, decoded server-side. [Guide](./xr1_robocasa365_quickstart.md). |

Every row records the model's **actual observed wire behaviour**. This
document describes the contract; each model's guide describes that model's own
wire in full, which is why the notes stop at the essentials and link onward.
Where a row says a model departs from this contract, the departure is what you
code against.

## What makes a model work with a contract-only client

A client that implements only this document, with no per-model special cases,
works with any model where all of these hold:

- [ ] Declares its camera views as named video tracks; names documented.
- [ ] Accepts `task_description` as a per-episode string.
- [ ] Accepts `state_json` with **`proprio` (flat float list) and `chunk_id`
      required, everything else optional with documented defaults**.
- [ ] Rejects a malformed `state_json` by dropping the request. It never
      crashes the session and never substitutes fabricated state.
- [ ] Answers exactly once per distinct `chunk_id`, and only after every view
      has delivered a post-request frame.
- [ ] Emits `action_prediction` with `actions` `[K, A]`, `proprios`, and
      `step` echoing the request's `chunk_id`.
- [ ] Is stateless across chunks, so a retried request (same `chunk_id`,
      bumped `retry` byte) reproduces the identical answer and a reset is
      cheap.
- [ ] Deduplicates identical `state_json` (continuous re-delivery of
      unchanged state must not re-trigger prediction).
- [ ] Publishes N, A, K and both layouts in its behaviour documentation, and
      has a row in the table above.

## What the tooling does not do for you

Three of the rules above are there because the tooling does not yet cover
them, not because the protocol requires them:

- `publish_track` before `READY` fails silently, so your client has to wait
  for the status event rather than assume the track landed.
- The SDK does not keep the session alive. Your client sends the pings.
- Nothing tags a chunk with the frame it came from, so push-then-request
  ordering is the only thing pairing an observation with its reply.

If a model behaves differently from this document, or you hit something these
rules do not cover, tell your Reactor contact.
