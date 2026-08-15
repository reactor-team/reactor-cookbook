#!/usr/bin/env python3
"""GR00T N1.7 quickstart: drive the hosted GR00T N1.7 model from Python.

Run: python groot_n17_quickstart.py, with REACTOR_API_KEY set.
"""
import asyncio


async def main():
    import logging

    # Keep INFO on: dropped commands and status transitions are logged, not raised.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from reactor_robotics import describe_api_key

    # Confirms the key is present and reports the endpoint. Prints the key's
    # LENGTH, never any part of its value.
    print(describe_api_key())

    from reactor_robotics.groot_n17 import (
        ACTION_DIMS, ACTION_HORIZON, FR3_JOINT_LIMITS, STRIDE, VIEWS,
        GrootN17Client, encode_state,
    )

    # Capacity: one B200 serves one session. A busy cluster returns HTTP 429
    # "no available capacity" at session creation. Wait and retry, or ask
    # Reactor for more.
    client = GrootN17Client()      # model="groot-n17", 15 fps tracks
    await client.connect()         # handlers -> connect -> await READY -> tracks -> ping

    print("status transitions :", " -> ".join(client.session.status_log))
    print("tracks published   :", ", ".join(client.session.tracks))
    print("endpoint           :", client.session.api_url)
    print("action horizon     :", ACTION_HORIZON, "rows;  fields", dict(ACTION_DIMS),
          "= 17 dims")
    print(f"frame window       : stride {STRIDE} ticks, hold {client.window_s:.2f}s "
          f"at {client.fps} fps")
    print("pairing            : drain, discard", client.discard_chunks, "chunk, take the next")
    print("keepalive          : ping every 10s (runtime kills at 20s of silence)")

    import numpy as np

    EXAMPLES_PATH = "examples/groot_n17_examples.npz"
    examples = np.load(EXAMPLES_PATH)
    N = len(examples["prompt"])

    # Five recorded examples. The frames are real: Franka Research 3 rig captures
    # in the DROID camera layout at 180x320, the embodiment this checkpoint
    # targets, so they are in distribution. `expected_*` are the chunks this
    # deployment returned for them at recording time.
    #
    # The streamed state is recovered from the recorded chunk's first row, this
    # model's absolute-anchored prediction for the state it was handed, so it is
    # that observation's proprioception to within the anchoring error measured
    # below. Full details: examples/PROVENANCE.md
    print(f"{N} recorded examples, frames keyed by track name (never positionally):")
    for v in VIEWS:
        print(f"  {v:<15} {examples[v].shape} {examples[v].dtype}")
    print(f"\ntask: {str(examples['prompt'][0])!r}")
    print(f"\nstreamed state per observation ({len(ACTION_DIMS)} keys, 17 floats):")
    print(f"  {'joint_position':<18} {examples['joint_position'].shape}")
    print(f"  {'eef_9d':<18} {examples['eef_9d'].shape}")
    print(f"  {'gripper_position':<18} {examples['gripper_position'].shape}")
    print(f"\ncalibrated at recording time:")
    print(f"  anchor band          {float(examples['anchor_band_rad']):.4f} rad")
    print(f"  per-step |dq| band   {float(examples['dq_band_rad']):.4f} rad")
    print(f"  run-to-run L2 band   {float(examples['l2_band']):.4f}")
    print(f"  chunk period         {float(examples['chunk_period_ms']):.0f} ms")

    import time

    live = []
    t_start = time.perf_counter()

    for i in range(N):
        frames = {v: examples[v][i] for v in VIEWS}      # keyed by track name

        # The three named state vectors. encode_state validates locally, because
        # the server zeroes anything it cannot parse.
        state_json = encode_state(
            examples["joint_position"][i],
            examples["eef_9d"][i],
            float(examples["gripper_position"][i]),
        )

        pred = await client.predict(frames, state_json, task=str(examples["prompt"][i]))
        live.append(pred)

        anchor = float(np.max(np.abs(
            pred.joint_position[0] - examples["joint_position"][i]
        )))
        print(
            f"example {i}: step={pred.step:>3}  joints={pred.joint_position.shape}  "
            f"anchor|d|={anchor:.4f} rad  {pred.latency_ms:6.1f} ms  "
            f"discarded={len(pred.discarded):>2}"
        )

    print(f"\n{N} chunks in {time.perf_counter() - t_start:.1f}s")
    print("'discarded' counts free-running chunks predict() dropped during the")
    print("hold. A large count is normal: at ~100 ms per chunk,")
    print(f"a {client.window_s:.2f}s hold produces about that many.")

    # The chunk period is a property of the deployment, not a constant. Measure it.
    period_ms = 1e3 * await client.observe_period_s()
    print(f"\nfree-running chunk period, measured now: {period_ms:.0f} ms")

    # ---------------------------------------------------------------------------
    # Deterministic checks plus the two calibrated bands, from
    # examples/PROVENANCE.md. L2 spread at recording time: mean 1.22, standard
    # deviation 0.44 over 3 passes.
    # ---------------------------------------------------------------------------
    # Both bands are the mean plus three standard deviations at recording time.
    ANCHOR_BAND = float(examples["anchor_band_rad"])   # 0.0688 rad
    DQ_BAND = float(examples["dq_band_rad"])           # 0.0622 rad
    L2_BAND = float(examples["l2_band"])               # 2.55, reported only
    # Worst per-step |dq| measured at recording time, not the band above.
    # Reported only; the gate uses DQ_BAND.
    MEASURED_DQ_MAX = 0.0594

    joints = np.stack([p.joint_position for p in live])       # (N, 40, 7)
    packed = np.stack([p.packed for p in live])               # (N, 40, 17)
    steps = [p.step for p in live]

    checks = []
    checks.append((f"joint_position is ({ACTION_HORIZON}, 7) on every chunk",
                   all(p.joint_position.shape == (ACTION_HORIZON, 7) for p in live)))
    checks.append((f"eef_9d ({ACTION_HORIZON}, 9) and gripper ({ACTION_HORIZON}, 1)",
                   all(p.eef_9d.shape == (ACTION_HORIZON, 9)
                       and p.gripper_position.shape == (ACTION_HORIZON, 1)
                       for p in live)))
    checks.append(("all 17 action dims finite", bool(np.isfinite(packed).all())))
    checks.append((f"step strictly increasing {steps}",
                   all(b > a for a, b in zip(steps, steps[1:]))))

    # THE anchor check: absolute targets must start at the state you sent. This is
    # what a mis-wired client fails.
    anchors = np.array([
        float(np.max(np.abs(live[i].joint_position[0] - examples["joint_position"][i])))
        for i in range(N)
    ])
    checks.append((f"row 0 anchored to the streamed state <= {ANCHOR_BAND:.4f} rad "
                   f"(worst {anchors.max():.4f})", bool(anchors.max() <= ANCHOR_BAND)))

    lo, hi = FR3_JOINT_LIMITS[:, 0], FR3_JOINT_LIMITS[:, 1]
    checks.append(("joint targets within FR3 joint limits",
                   bool((joints >= lo).all() and (joints <= hi).all())))

    dq = float(np.max(np.abs(np.diff(joints, axis=1))))
    checks.append((f"per-step |dq| <= {DQ_BAND:.4f} rad (worst {dq:.4f})",
                   dq <= DQ_BAND))

    grip = np.stack([p.gripper_position for p in live])
    checks.append((f"gripper in [0, 1] (observed {grip.min():.3f}..{grip.max():.3f})",
                   bool((grip >= 0).all() and (grip <= 1).all())))

    print(f"{'check':<64} result")
    print("-" * 74)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"{label:<64} {'PASS' if passed else 'FAIL'}")
    print("-" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")

    # --- reported only ---------------------------------------------------------
    lat = [p.latency_ms for p in live]
    print()
    print("Latency:")
    print(f"  measured here          : p50 {np.median(lat):.1f} ms   "
          f"min {min(lat):.1f}   max {max(lat):.1f}")
    print(f"  chunk period           : {period_ms:.0f} ms  (the model's own cadence)")
    print(f"  what predict() measures: one discarded chunk plus the next, so ~1.5")
    print(f"                           periods ~ {1.5 * period_ms:.0f} ms, not per-chunk compute.")
    print(f"  frame-window hold      : {client.window_s:.2f}s per observation, on top. A robot")
    print("                           never pays it: its cameras fill the window")
    print("                           while it executes the previous chunk.")
    print()
    print(f"PER-STEP |dq|: worst {dq:.4f} rad. At 15 Hz that is "
          f"{dq * 15:.2f} rad/s.")
    print("  Reactor's FR3 client clamps at 0.05 rad/tick, so a chunk this size")
    print("  mostly needs no clamping, but the measured spread reaches")
    print(f"  {MEASURED_DQ_MAX:.4f}, so the clamp does engage sometimes. Keep it.")
    print()
    l2 = np.array([
        float(np.linalg.norm(live[i].joint_position
                             - examples["expected_joint_position"][i]))
        for i in range(N)
    ])
    mag = float(np.mean([np.linalg.norm(j) for j in joints]))
    print(f"Run-to-run L2 vs the recorded chunks (band {L2_BAND:.3f}; "
          f"{100 * L2_BAND / mag:.0f}% of magnitude {mag:.1f}):")
    for i, d in enumerate(l2):
        print(f"  chunk {i}: {d:.4f}   {'ok' if d <= L2_BAND else 'OUTSIDE BAND'}")
    print(f"  worst {l2.max():.4f}")
    print()
    print("A three-standard-deviation bound from 15 samples puts a value outside")
    print("the band occasionally by construction, so this is a drift signal,")
    print("reported only. Repeated values across re-runs suggest the deployment")
    print("has changed.")

    # ---------------------------------------------------------------------------
    # Negative tests: the failure modes you are most likely to hit. ~45 s.
    # ---------------------------------------------------------------------------
    import asyncio

    # (1) Malformed state_json is not rejected -- the affected key becomes zeros
    #     and the model keeps predicting. See it happen.
    await client.session.send("set_state_json", {"state_json": "{not json"})
    try:
        data = await client.session.next_message("action_prediction", timeout_s=30.0)
        print(f"still predicting after malformed state_json (step={data.get('step')})")
        print("=> the bad key became zeros. No error, no dropped request: the policy")
        print("   is acting on a fabricated state. Validate client-side.")
    except asyncio.TimeoutError:
        print("no chunk after malformed state_json -- the model stopped instead")

    # Put a good state back before anything else.
    await client.session.send(
        "set_state_json",
        {"state_json": encode_state(examples["joint_position"][0],
                                    examples["eef_9d"][0],
                                    float(examples["gripper_position"][0]))},
    )

    # (2) The keepalive matters. The runtime disconnects a client quiet for 20 s,
    #     and a robot sends nothing while it executes a chunk. Sit idle longer
    #     than that; session.py has been pinging every 10 s.
    print("\nsitting idle for 25 s to exercise the keepalive...")
    await asyncio.sleep(25.0)
    print("status after 25s idle:", " -> ".join(client.session.status_log))
    pred = await client.predict(
        {v: examples[v][0] for v in VIEWS},
        encode_state(examples["joint_position"][0], examples["eef_9d"][0],
                     float(examples["gripper_position"][0])),
        task=str(examples["prompt"][0]),
    )
    print(f"still serving after 25s idle: step={pred.step} {pred.latency_ms:.1f} ms")
    print("=> 25 s of silence did not drop the session. Without the 10 s ping it "
          "would have.")

    # (3) reset restarts the inference counter at 0.
    before = pred.step
    await client.reset()
    pred = await client.predict(
        {v: examples[v][0] for v in VIEWS},
        encode_state(examples["joint_position"][0], examples["eef_9d"][0],
                     float(examples["gripper_position"][0])),
        task=str(examples["prompt"][0]),
    )
    print(f"\nstep before reset: {before}  ->  after reset: {pred.step}")
    print("=> the counter restarted, so `step` is per-episode state, not an echo of")
    print("   anything you sent. It is not 0 here because the model kept")
    print("   free-running through the frame-window hold, and predict() discarded")
    print(f"   those {len(pred.discarded)} chunks; step 0 was one of them.")

    # Always close. A live session holds a real GPU worker.
    await client.close()
    print("closed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
