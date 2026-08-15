#!/usr/bin/env python3
"""Cosmos-Nano-Policy-DROID quickstart: drive the hosted model from Python.

Run: python cosmos_droid_quickstart.py, with REACTOR_API_KEY set.
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

    from reactor_robotics.cosmos_droid import (
        ACTION_SHAPE, CHUNK_BUDGET_MS, CONTROL_HZ, EXPECTED_COMPUTE_MS,
        EXPECTED_WIRE_P50_MS, FR3_JOINT_LIMITS, TRACKS, CosmosDroidClient,
    )

    # Capacity: one B200 serves one session. A busy cluster returns HTTP 429
    # "no available capacity" on session creation. Wait and retry, or ask
    # Reactor for more capacity.
    client = CosmosDroidClient()      # model="cosmos-nano-policy-droid", 15 fps

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
                    f"no capacity for cosmos-nano-policy-droid after "
                    f"{CONNECT_ATTEMPTS} attempts. No server has free capacity "
                    f"right now -- retry in a few minutes, or ask your Reactor "
                    f"contact."
                ) from exc
            print(f"429 no capacity (attempt {attempt}/{CONNECT_ATTEMPTS}); "
                  f"retrying in {CONNECT_WAIT_S:.0f}s")
            import asyncio; await asyncio.sleep(CONNECT_WAIT_S)

    try:

        print("status transitions :", " -> ".join(client.session.status_log))
        print("tracks published   :", ", ".join(client.session.tracks))
        print("endpoint           :", client.session.api_url)
        print("action shape       :", ACTION_SHAPE, "= 7 absolute joints (rad) + gripper")
        print(f"chunk budget       : {CHUNK_BUDGET_MS:.0f} ms "
              f"({ACTION_SHAPE[0]} rows at {CONTROL_HZ:.0f} Hz)")
        print(f"reference latency  : ~{EXPECTED_COMPUTE_MS:.0f} ms compute, "
              f"~{EXPECTED_WIRE_P50_MS:.0f} ms p50 wire (reference)")
        print(f"frame settle       : {client.settle_s:.2f}s so the fresh frame lands first")
        print("reset event        : none -- this model is stateless per prediction")
        print("keepalive          : ping every 10s (runtime kills at 20s of silence)")

        import numpy as np
        from pathlib import Path

        EXAMPLES_PATH = Path("examples/cosmos_droid_examples.npz")

        # examples/cosmos_droid_examples.npz holds five observations recorded against
        # this deployment together with the chunks it returned then, plus the
        # tolerances calibrated at recording time, and the comparison runs against
        # those recordings. Without the fixture this falls back to live checks.
        # CALIBRATED below is that detection.
        #
        # The frames are synthetic and deterministic. This checkpoint takes three views
        # (wrist + two exterior); the available real DROID captures come from a
        # two-camera rig, and duplicating one exterior view into the second slot would
        # misrepresent a second camera. So this is a protocol fixture, not a
        # demonstration of task behaviour; task competence comes from the real
        # simulator, which ../cosmos-droid/ in this repo runs against this same
        # model. Full details: examples/PROVENANCE.md
        CALIBRATED = EXAMPLES_PATH.exists()
        if CALIBRATED:
            examples = dict(np.load(EXAMPLES_PATH))
            N = len(examples["prompt"])
            print(f"{N} recorded examples from {EXAMPLES_PATH}")
            print("calibrated at recording time:")
            print(f"  anchor band        {float(examples['anchor_band_rad']):.4f} rad")
            print(f"  per-step |dq| band {float(examples['dq_band_rad']):.4f} rad")
            print(f"  run-to-run L2 band {float(examples['l2_band']):.4f}")
        else:
            # No fixture recorded yet (it is produced against the live deployment by
            # examples/record_cosmos_droid_examples.py). Fall back to the same
            # deterministic observations that script uses, so the protocol still runs
            # end to end -- but without calibrated tolerances, so the checks below
            # gate only on the ones that need none and report the rest.
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_rec", "examples/record_cosmos_droid_examples.py"
            )
            _rec = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_rec)
            N = 5
            obs = [_rec.synthetic_observation(i) for i in range(N)]
            examples = {t: np.stack([o[0][t] for o in obs]) for t in TRACKS}
            examples.update(
                prompt=np.asarray([_rec.TASK] * N),
                joints=np.stack([o[1] for o in obs]),
                gripper=np.asarray([o[2] for o in obs]),
            )
            print("No recorded fixture yet -- using the recording script's own")
            print("deterministic observations. The checks below gate only on the")
            print("absolute ones and report the rest, because no band is calibrated.")
            print("To create the fixture (needs the live model):")
            print("  REACTOR_API_KEY=... python examples/record_cosmos_droid_examples.py")

        print(f"\nframes keyed by track name (never positionally):")
        for t in TRACKS:
            print(f"  {t:<17} {examples[t].shape} {examples[t].dtype}")
        print(f"\ntask: {str(examples['prompt'][0])!r}")

        import time

        live = []
        t_start = time.perf_counter()

        for i in range(N):
            frames = {t: examples[t][i] for t in TRACKS}      # keyed by track name

            # No reset, no episode ceremony: the model is stateless per prediction, so
            # the task and the proprio are sent with every call. predict() echoes the
            # previous chunk's step; the first call has nothing to echo.
            pred = await client.predict(
                frames,
                examples["joints"][i],
                float(examples["gripper"][i]),
                task=str(examples["prompt"][i]),
            )
            live.append(pred)

            anchor = float(np.max(np.abs(pred.joint_position[0] - examples["joints"][i])))
            print(
                f"example {i}: step={pred.step}  shape={pred.actions.shape}  "
                f"anchor|d|={anchor:.4f} rad  {pred.latency_ms:7.1f} ms  "
                f"({100 * pred.latency_ms / CHUNK_BUDGET_MS:4.1f}% of the "
                f"{CHUNK_BUDGET_MS:.0f} ms budget)"
            )

        print(f"\n{N} chunks in {time.perf_counter() - t_start:.1f}s")
        print(f"stale chunks discarded: {sum(len(p.discarded) for p in live)} "
              f"(at most one chunk is ever in flight)")

        # ---------------------------------------------------------------------------
        # Checks that hold regardless of sampling; see the section above.
        # Bands come from the fixture; without it they are reported, not enforced.
        # ---------------------------------------------------------------------------
        actions = np.stack([p.actions for p in live])          # (N, 32, 8)
        joints = actions[:, :, :7]
        steps = [p.step for p in live]

        anchors = np.array([
            float(np.max(np.abs(live[i].joint_position[0] - examples["joints"][i])))
            for i in range(N)
        ])
        dq = float(np.max(np.abs(np.diff(joints, axis=1))))

        checks = []
        checks.append((f"shape is {ACTION_SHAPE} on every chunk",
                       all(p.actions.shape == ACTION_SHAPE for p in live)))
        checks.append(("all actions finite", bool(np.isfinite(actions).all())))
        checks.append((f"step strictly increasing {steps}",
                       all(b > a for a, b in zip(steps, steps[1:]))))
        checks.append(("first chunk of the session is step 0", steps[0] == 0))
        lo, hi = FR3_JOINT_LIMITS[:, 0], FR3_JOINT_LIMITS[:, 1]
        checks.append(("joint targets within FR3 joint limits",
                       bool((joints >= lo).all() and (joints <= hi).all())))
        grip = actions[:, :, 7]
        # [0, 1] is the gripper's data convention, not a generator guarantee: the
        # policy samples an unbounded continuous vector, and gripper training data
        # sits AT the bounds (open/closed), so raw output lands a hair outside them
        # about as often as a hair inside. The wire relays the model's raw output;
        # clamp before actuating (see the guide's physical-deployment section). A
        # small excursion is healthy sampling; a large one is a real regression,
        # which is what this tolerance still catches.
        GRIPPER_TOL = 0.05
        checks.append((f"gripper in [0, 1] +/- {GRIPPER_TOL} sampling tol "
                       f"(observed {grip.min():.3f}..{grip.max():.3f})",
                       bool((grip >= -GRIPPER_TOL).all()
                            and (grip <= 1 + GRIPPER_TOL).all())))
        checks.append(("no stale chunks served as answers",
                       all(not p.discarded for p in live)))

        reported = []
        if CALIBRATED:
            ANCHOR_BAND = float(examples["anchor_band_rad"])
            DQ_BAND = float(examples["dq_band_rad"])
            checks.append((f"row 0 anchored to the sent proprio <= {ANCHOR_BAND:.4f} rad "
                           f"(worst {anchors.max():.4f})", bool(anchors.max() <= ANCHOR_BAND)))
            checks.append((f"per-step |dq| <= {DQ_BAND:.4f} rad (worst {dq:.4f})",
                           dq <= DQ_BAND))
        else:
            reported.append(f"row 0 anchor, worst {anchors.max():.4f} rad "
                            f"(band not calibrated -- no fixture yet)")
            reported.append(f"per-step |dq|, worst {dq:.4f} rad "
                            f"(band not calibrated -- no fixture yet)")

        print(f"{'check':<64} result")
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
        print("Latency vs the chunk budget: one chunk is 2133 ms of motion, so anything")
        print("under that keeps the arm fed.")
        print(f"  measured here          : p50 {np.median(lat):.1f} ms   "
              f"min {min(lat):.1f}   max {max(lat):.1f}")
        print(f"  chunk budget           : {CHUNK_BUDGET_MS:.0f} ms  -> headroom "
              f"{CHUNK_BUDGET_MS / np.median(lat):.1f}x")
        print(f"  reference              : ~{EXPECTED_COMPUTE_MS:.0f} ms compute, "
              f"~{EXPECTED_WIRE_P50_MS:.0f} ms p50 think+wire")
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
        # Negative tests: the two failure modes you are most likely to hit. ~40 s.
        # ---------------------------------------------------------------------------
        import asyncio
        from reactor_robotics.cosmos_droid import encode_executed_step, encode_proprio

        # (1) A non-increasing echoed step produces no chunk. This is the model's
        #     flow control, so see it hold.
        await client.session.send(
            "set_executed_step_json",
            {"executed_step_json": encode_executed_step(
                client.last_step - 1, live[-1].actions)},
        )
        try:
            await client.session.next_message("action_prediction", timeout_s=12.0)
            print("unexpected: a non-increasing step was answered")
        except asyncio.TimeoutError:
            print("non-increasing echoed step: no chunk in 12s, as documented.")
            print("=> the model advances only on a strictly greater step, never on bad")
            print("   input, so a stalled control loop cannot run it ahead.")

        # (2) The keepalive matters. The runtime disconnects a client that stays quiet
        #     for 20 s, and this protocol sends nothing for the whole 2133 ms a robot
        #     spends executing a chunk -- and longer if it stalls. Sit idle past the
        #     watchdog and confirm the session survives; session.py has been pinging
        #     every 10 s.
        print("\nsitting idle for 25 s to exercise the keepalive...")
        await asyncio.sleep(25.0)
        print("status after 25s idle:", " -> ".join(client.session.status_log))

        pred = await client.predict(
            {t: examples[t][0] for t in TRACKS},
            examples["joints"][0], float(examples["gripper"][0]),
            task=str(examples["prompt"][0]),
        )
        print(f"still serving after 25s idle: step={pred.step} {pred.latency_ms:.1f} ms")
        print("=> 25 s of silence did not drop the session. Without the 10 s ping it "
              "would have.")

        # (3) A task change needs no ceremony, because the model is stateless. No reset,
        #     no cache to clear -- the new prompt applies to the next chunk.
        pred = await client.predict(
            {t: examples[t][0] for t in TRACKS},
            examples["joints"][0], float(examples["gripper"][0]),
            task="put the spoon in the drawer",
        )
        print(f"\nafter a task change: step={pred.step} {pred.latency_ms:.1f} ms")
        print("=> no reset event exists on this wire and none was needed.")

    finally:
        # Always close, even on the error path: a live session holds a real
        # GPU worker, and an unclosed one exits with unclosed-transport noise.
        await client.close()
        print("closed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
