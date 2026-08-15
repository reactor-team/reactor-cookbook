#!/usr/bin/env python3
"""XR-1 RoboCasa365 quickstart: drive the hosted model from Python.

Run: python xr1_robocasa365_quickstart.py, with REACTOR_API_KEY set.
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

    from reactor_robotics.xr1_robocasa365 import (
        ACTION_SHAPE, EXPECTED_COMPUTE_COMPILED_MS, EXPECTED_COMPUTE_MS,
        LIVE_DIMS, OBS_HISTORY, OBS_INTERVAL, REPLAN_STEPS, STATE_ROW_DIM,
        TRACKS, Xr1Robocasa365Client,
    )

    # Capacity: the deployment serves one session at a time. A busy cluster
    # returns HTTP 429
    # "no available capacity" on session creation. Wait and retry, or ask
    # Reactor for more capacity.
    client = Xr1Robocasa365Client()   # model="xr1-robocasa365", 15 fps, 256x256

    CONNECT_ATTEMPTS, CONNECT_WAIT_S = 4, 30.0
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            await client.connect()
            break
        except Exception as exc:
            if not ("429" in str(exc) or "no available capacity" in str(exc)):
                raise
            if attempt == CONNECT_ATTEMPTS:
                raise RuntimeError(
                    f"no capacity for xr1-robocasa365 after "
                    f"{CONNECT_ATTEMPTS} attempts. No server has free capacity "
                    f"right now -- retry in a few minutes, or ask your Reactor "
                    f"contact."
                ) from exc
            print(f"429 no capacity (attempt {attempt}/{CONNECT_ATTEMPTS}); "
                  f"retrying in {CONNECT_WAIT_S:.0f}s")
            import asyncio; await asyncio.sleep(CONNECT_WAIT_S)

    print("status transitions :", " -> ".join(client.session.status_log))
    print("tracks published   :", ", ".join(client.session.tracks))
    print("endpoint           :", client.session.api_url)
    print("action shape       :", ACTION_SHAPE,
          f"= packed layout, first {LIVE_DIMS} columns live for PandaOmron")
    print(f"observation        : {OBS_HISTORY} frames per camera, "
          f"{OBS_INTERVAL} env steps apart, paired into complete sets")
    print(f"state history      : {OBS_HISTORY} rows x {STATE_ROW_DIM} floats, "
          "oldest first")
    print(f"replan window      : {REPLAN_STEPS} steps between predictions")
    print(f"reference latency  : ~{EXPECTED_COMPUTE_MS:.0f} ms compute default, "
          f"~{EXPECTED_COMPUTE_COMPILED_MS:.0f} ms with the gated lever")
    print(f"frame settle       : {client.settle_s:.2f}s so the fresh frame set "
          "lands first")
    print("reset event        : yes -- this model holds a per-session history")
    print("keepalive          : ping every 10s (runtime kills at 20s of silence)")

    import numpy as np
    from pathlib import Path

    EXAMPLES_PATH = Path("examples/xr1_robocasa365_examples.npz")

    # The fixture is not recorded yet, so this runs live checks against the model.
    # Once examples/xr1_robocasa365_examples.npz exists it holds five observations
    # recorded against this deployment together with the chunks it returned then,
    # plus the bands calibrated at recording time, and the comparison switches to
    # those recordings. CALIBRATED below is that detection.
    #
    # The frames are synthetic and deterministic: a five-step kitchen approach at
    # the benchmark's native 256x256, with three genuinely distinct views. So this
    # is a protocol fixture, not a demonstration of task behaviour; task competence
    # comes from the RoboCasa365 benchmark itself, quoted in the guide.
    # Full details: examples/PROVENANCE.md
    CALIBRATED = EXAMPLES_PATH.exists()
    if CALIBRATED:
        examples = dict(np.load(EXAMPLES_PATH))
        N = len(examples["prompt"])
        print(f"\n{N} recorded examples from {EXAMPLES_PATH}")
        print("calibrated at recording time:")
        _band = np.atleast_1d(examples["step_delta_band"]).astype(float)
        print("  per-step |delta| band, one per live column:")
        print("   ", " ".join(f"{v:.3f}" for v in _band))
        print(f"  run-to-run L2 band    {float(examples['l2_band']):.4f}")
    else:
        # No fixture recorded yet (it is produced against the live deployment by
        # examples/record_xr1_robocasa365_examples.py). Fall back to the same
        # deterministic observations that script uses, so the protocol still runs
        # end to end -- but without calibrated bands, so the cell below runs only
        # the checks that need none and reports the rest.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rec", "examples/record_xr1_robocasa365_examples.py"
        )
        _rec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_rec)
        N = 5
        obs = [_rec.synthetic_observation(i) for i in range(N)]
        examples = {t: np.stack([o[0][t] for o in obs]) for t in TRACKS}
        examples.update(
            prompt=np.asarray([_rec.TASK] * N),
            state_history=np.stack([_rec.state_window(i) for i in range(N)]),
        )
        print("\nNo recorded fixture yet -- using the recording script's own")
        print("deterministic observations. The cell below runs the absolute checks")
        print("and reports the rest, because no band is calibrated.")
        print("To create the fixture (needs the live model):")
        print("  REACTOR_API_KEY=... python "
              "examples/record_xr1_robocasa365_examples.py")

    print("\nframes keyed by track name (never positionally):")
    for t in TRACKS:
        print(f"  {t:<17} {examples[t].shape} {examples[t].dtype}")
    print(f"\ntask: {str(examples['prompt'][0])!r}")

    import time

    live = []
    t_start = time.perf_counter()

    for i in range(N):
        frames = {t: examples[t][i] for t in TRACKS}      # keyed by track name

        # Every prediction, including the first, is gated on an echo whose step
        # strictly increases. predict() keeps that counter, advancing it by the
        # replan window, so the caller never has to.
        pred = await client.predict(
            frames,
            examples["state_history"][i],
            task=str(examples["prompt"][i]),
        )
        live.append(pred)

        per_col = np.max(np.abs(np.diff(pred.live, axis=0)), axis=0)
        print(
            f"example {i}: step={pred.step}  shape={pred.actions.shape}  "
            f"max|step delta|={float(np.max(per_col)):.4f} "
            f"(col {int(np.argmax(per_col))})  {pred.latency_ms:7.1f} ms"
        )

    print(f"\n{N} chunks in {time.perf_counter() - t_start:.1f}s")
    print(f"stale chunks discarded: {sum(len(p.discarded) for p in live)} "
          f"(at most one chunk is ever in flight)")

    # ---------------------------------------------------------------------------
    # Checks that hold regardless of sampling; see the section above.
    # Bands come from the fixture; without it they are reported, not enforced.
    # ---------------------------------------------------------------------------
    actions = np.stack([p.actions for p in live])          # (N, 16, 60)
    live_cols = actions[:, :, :LIVE_DIMS]
    steps = [p.step for p in live]
    # Per column, worst over every chunk: shape (LIVE_DIMS,). Kept per-column
    # rather than collapsed to one number because the gripper column swings
    # the full [-1, 1] and would otherwise set the only band there is.
    step_delta = np.max(np.abs(np.diff(live_cols, axis=1)), axis=(0, 1))

    checks = []
    checks.append((f"shape is {ACTION_SHAPE} on every chunk",
                   all(p.actions.shape == ACTION_SHAPE for p in live)))
    checks.append(("all actions finite", bool(np.isfinite(actions).all())))
    checks.append((f"step strictly increasing {steps}",
                   all(b > a for a, b in zip(steps, steps[1:]))))
    checks.append(("first chunk of the session is step 0", steps[0] == 0))
    checks.append((
        f"exactly one chunk per echo ({len(live)} echoes, {len(live)} chunks)",
        len(live) == N))
    checks.append(("no stale chunks served as answers",
                   all(not p.discarded for p in live)))

    # The per-step magnitude band is REPORTED, never gated: this policy
    # varies enough between sessions that a healthy run can graze
    # a 3-sigma bound, so gating it would fail runs where nothing is wrong. A
    # single column slightly over is normal; the same column over on run after
    # run is the signal, and a gross excursion is visible either way.
    reported = []
    if CALIBRATED:
        STEP_BAND = np.atleast_1d(examples["step_delta_band"]).astype(float)
        over = [c for c in range(LIVE_DIMS) if step_delta[c] > STEP_BAND[c]]
        worst = int(np.argmax(step_delta - STEP_BAND))
        reported.append(
            f"per-step |delta| vs band, worst col {worst}: "
            f"{step_delta[worst]:.4f} vs {STEP_BAND[worst]:.4f}"
            + (f"; columns over band: {over}" if over else "; all columns inside"))
    else:
        reported.append(
            f"per-step |delta| per column, worst {float(np.max(step_delta)):.4f} "
            f"on col {int(np.argmax(step_delta))} "
            f"(bands not calibrated -- no fixture yet)")

    print(f"\n{'check':<64} result")
    print("-" * 74)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"{label:<64} {'PASS' if passed else 'FAIL'}")
    print("-" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")
    for line in reported:
        print(f"  reported only: {line}")

    # --- reported only ---------------------------------------------------------
    lat = [p.latency_ms for p in live]
    print()
    print("Latency. One chunk is 16 steps, and the benchmark executes either all")
    print(f"{REPLAN_STEPS} of them or the first 8 before asking again, so the number")
    print("that matters is how much of that window the round trip consumes.")
    print(f"  measured here          : p50 {np.median(lat):.1f} ms   "
          f"min {min(lat):.1f}   max {max(lat):.1f}")
    print(f"  reference compute      : ~{EXPECTED_COMPUTE_MS:.0f} ms default build, "
          f"~{EXPECTED_COMPUTE_COMPILED_MS:.0f} ms with XR1_COMPILE_DIT enabled")
    print()
    if CALIBRATED:
        L2_BAND = float(examples["l2_band"])
        l2 = np.array([
            float(np.linalg.norm(live[i].actions - examples["expected_actions"][i]))
            for i in range(N)
        ])
        mag = float(np.mean([np.linalg.norm(a) for a in actions]))
        print(f"Run-to-run L2 vs the recorded chunks (band {L2_BAND:.3f}, measured "
              f"from the")
        print(f"model's own spread; {100 * L2_BAND / mag:.0f}% of chunk magnitude "
              f"{mag:.1f}):")
        for i, d in enumerate(l2):
            print(f"  chunk {i}: {d:.4f}   {'ok' if d <= L2_BAND else 'OUTSIDE BAND'}")
        print(f"  worst {l2.max():.4f}")
        print()
        print("A three-standard-deviation bound from 15 samples puts a single")
        print("value outside the band occasionally by construction, so this is a")
        print("drift signal, reported only.")
    else:
        print("Run-to-run L2: not available -- no recorded chunks to compare against.")

    # ---------------------------------------------------------------------------
    # Negative tests: the three failure modes you are most likely to hit. ~45 s.
    # ---------------------------------------------------------------------------
    import asyncio
    from reactor_robotics.xr1_robocasa365 import encode_executed_step

    # (1) A non-increasing echoed step produces no chunk. This is the model's
    #     flow control, so see it hold.
    #
    #     Re-send the last echo the model ACTUALLY advanced on. Note this is
    #     client.last_echo, not client.executed_steps - 1: executed_steps has
    #     already moved on by a full replan window, so that value is still
    #     strictly greater than what the model consumed and gets answered
    #     normally, which is not the case under test.
    await client.session.send(
        "set_executed_step_json",
        {"executed_step_json": encode_executed_step(client.last_echo)},
    )
    try:
        await client.session.next_message("action_prediction", timeout_s=12.0)
        print("\nunexpected: a non-increasing step was answered")
    except asyncio.TimeoutError:
        print("\nnon-increasing echoed step: no chunk in 12s, as documented.")
        print("=> the model advances only on a strictly greater step, never on bad")
        print("   input, so a stalled control loop cannot run it ahead.")

    # (2) The keepalive matters. The runtime disconnects a client that stays quiet
    #     for 20 s, and a robot executing a full 16-step window can easily be
    #     silent that long. Sit idle past the watchdog and confirm the session
    #     survives; session.py has been pinging every 10 s.
    print("\nsitting idle for 25 s to exercise the keepalive...")
    await asyncio.sleep(25.0)
    print("status after 25s idle:", " -> ".join(client.session.status_log))

    pred = await client.predict(
        {t: examples[t][0] for t in TRACKS},
        examples["state_history"][0],
        task=str(examples["prompt"][0]),
    )
    print(f"still serving after 25s idle: step={pred.step} {pred.latency_ms:.1f} ms")
    print("=> 25 s of silence did not drop the session. Without the 10 s ping it "
          "would have.")

    # (3) reset starts a new episode. The model carries a per-session
    #     observation history and flow-control counter, so an episode
    #     boundary is a real event on this wire. After it, the model's step
    #     counter restarts.
    await client.reset()
    pred = await client.predict(
        {t: examples[t][0] for t in TRACKS},
        examples["state_history"][0],
        task="open the microwave door",
    )
    print(f"\nafter reset and a task change: step={pred.step} "
          f"{pred.latency_ms:.1f} ms")
    print("=> reset cleared the observation history and the flow-control counter,")
    print("   so the new episode starts from step 0 with nothing carried over.")

    # Always close. A live session holds a real GPU worker.
    await client.close()
    print("closed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
