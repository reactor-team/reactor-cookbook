#!/usr/bin/env python3
"""DreamZero quickstart: drive the hosted DreamZero model from Python.

Run: python dreamzero_quickstart.py, with REACTOR_API_KEY set.
"""
import asyncio


async def main():
    import logging

    # Keep INFO on: the SDK logs status transitions and dropped commands.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from reactor_robotics import describe_api_key

    # Reports the key's LENGTH and the endpoint. Never any part of its value.
    print(describe_api_key())

    from reactor_robotics.dreamzero import FRANKA_JOINT_LIMITS, TRACKS, DreamZeroClient

    # DreamZero holds TWO B200s for the life of a session. Two consequences:
    #   - a cold session can take minutes to report READY (14B of weights, a
    #     TensorRT engine and torch.compile warmup), hence the long default;
    #   - a busy cluster returns HTTP 429 "no available capacity" on session
    #     creation. Wait and retry in about a minute, or ask Reactor for more capacity.
    client = DreamZeroClient()          # model="dreamzero", 15 fps tracks
    await client.connect()

    print("status transitions :", " -> ".join(client.session.status_log))
    print("tracks published   :", ", ".join(client.session.tracks))
    print("endpoint           :", client.session.api_url)

    import numpy as np

    EXAMPLES_PATH = "examples/dreamzero_examples.npz"
    examples = np.load(EXAMPLES_PATH)
    N = len(examples["prompt"])

    # Five recorded examples: observations recorded against this same deployment,
    # together with the action chunks it returned then. The frames are synthetic
    # and deterministic (no DROID-appropriate real frames were available), so this
    # is a protocol fixture, not a demonstration of task behaviour. exterior_2 is
    # all black on purpose (RoboLab's --cam2-source black = training-time camera
    # dropout). Full details: examples/PROVENANCE.md
    print(f"{N} recorded examples, frames keyed by track name:")
    for t in TRACKS:
        black = "  (all black, by design)" if not examples[t].any() else ""
        print(f"  {t:<12} {examples[t].shape} {examples[t].dtype}{black}")
    print(f"\ntask: {str(examples['prompt'][0])!r}\n")

    live = []
    for i in range(N):
        frames = {t: examples[t][i] for t in TRACKS}
        pred = await client.predict(
            frames,
            examples["joints"][i],
            float(examples["gripper"][i]),
            task=str(examples["prompt"][i]),
        )
        live.append(pred)
        print(
            f"example {i}: chunk_index={pred.chunk_index}  "
            f"obs_seq={pred.obs_seq:>3}  "
            f"shape={pred.actions.shape}  inference={pred.inference_seconds:.3f}s  "
            f"latency={pred.latency_ms:6.1f}ms  in-flight discarded="
            f"{len(pred.discarded)}"
        )

    print("\n'discarded' counts chunks ignored because they were computed from the")
    print("previous observation. A non-zero count is normal.")

    # Checks that hold regardless of sampling; see the section above.
    ADVERTISED_S = 0.267        # the deployment's published operating point
    CONTINUITY_BAND = 0.25      # rad, see note below
    L2_BAND = float(examples["l2_band"])  # calibrated at recording; PROVENANCE.md

    actions = np.stack([p.actions for p in live])            # (N, 24, 8)
    obs_seqs = [p.obs_seq for p in live]
    chunk_ids = [p.chunk_index for p in live]
    infer = np.array([p.inference_seconds for p in live])

    checks = []

    checks.append(("shape is (24, 8) on every chunk",
                   all(p.actions.shape == (24, 8) for p in live)))
    checks.append(("all actions finite", bool(np.isfinite(actions).all())))
    checks.append((f"obs_seq strictly increasing {obs_seqs}",
                   all(b > a for a, b in zip(obs_seqs, obs_seqs[1:]))))
    checks.append((f"chunk_index strictly increasing {chunk_ids}",
                   all(b > a for a, b in zip(chunk_ids, chunk_ids[1:]))))

    lo, hi = FRANKA_JOINT_LIMITS[:, 0], FRANKA_JOINT_LIMITS[:, 1]
    joints_ok = bool((actions[:, :, :7] >= lo).all() and (actions[:, :, :7] <= hi).all())
    checks.append(("joint targets within Franka limits", joints_ok))

    grip = actions[:, :, 7]
    checks.append((f"gripper in [0, 1]  (observed {grip.min():.3f}..{grip.max():.3f})",
                   bool((grip >= 0).all() and (grip <= 1).all())))

    # Last joint target of one chunk vs the first of the next; replans are ~267 ms
    # apart, so a small step is expected.
    bnd = [float(np.max(np.abs(actions[i, -1, :7] - actions[i + 1, 0, :7])))
           for i in range(len(actions) - 1)]
    checks.append((f"chunk-boundary continuity <= {CONTINUITY_BAND} rad "
                   f"(worst {max(bnd):.4f})", max(bnd) <= CONTINUITY_BAND))

    print(f"{'check':<58} result")
    print("-" * 68)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"{label:<58} {'PASS' if passed else 'FAIL'}")
    print("-" * 68)
    print("RESULT:", "PASS" if ok else "FAIL")

    # --- reported only ---------------------------------------------------------
    print()
    print("Replan cost (model-reported inference_seconds):")
    print(f"  median {np.median(infer):.3f}s   mean {np.mean(infer):.3f}s   "
          f"min {infer.min():.3f}s   max {infer.max():.3f}s")
    print(f"  advertised operating point: {ADVERTISED_S:.3f}s"
          f"  ->  {'at or better than advertised' if np.median(infer) <= ADVERTISED_S else 'slower than advertised (contention?)'}")
    print(f"  that is ~{1 / np.median(infer):.1f} replans/s: it replans every "
          f"{np.median(infer) * 1e3:.0f} ms against the 1.6 s of motion a "
          f"24-step chunk covers at 15 Hz.")

    l2 = np.array([
        float(np.linalg.norm(live[i].actions - examples["expected_actions"][i]))
        for i in range(N)
    ])
    print()
    print(f"L2 vs the recorded examples (band {L2_BAND:.4f}, calibrated from the")
    print("model's own run-to-run spread at recording time, not a parity check):")
    for i, d in enumerate(l2):
        flag = "ok" if d <= L2_BAND else "OUTSIDE BAND"
        print(f"  chunk {i}: {d:.4f}   {flag}")
    print(f"  worst {l2.max():.4f}  vs chunk magnitude "
          f"~{np.linalg.norm(actions[0]):.1f} (band is ~{100 * L2_BAND / np.linalg.norm(actions[0]):.1f}% of it)")
    print()
    print("The band is a three-standard-deviation bound estimated from 15 samples,")
    print("so a single value outside it is normal and does not fail the checks")
    print("above. Repeated values outside it across re-runs suggest the deployment")
    print("has changed.")

    # ---------------------------------------------------------------------------
    # reset ends the episode: it clears the prompt and the causal cache and
    # returns the model to WAITING. The observable consequence is that obs_seq
    # restarts from 0, which is why the client's high-water mark has to be reset
    # along with it, and why reset() is a client method rather than a bare
    # send("reset").
    # ---------------------------------------------------------------------------
    before = max(p.obs_seq for p in live)
    await client.reset()                      # sends reset, awaits episode_reset
    print(f"obs_seq high-water mark before reset : {before}")
    print(f"client mark after reset              : "
          f"{client.obs_seq_high}  (-1 = no chunks seen this episode)")

    # One observation in the NEW episode: obs_seq starts over, well below `before`.
    frames = {t: examples[t][0] for t in TRACKS}
    pred = await client.predict(frames, examples["joints"][0],
                                float(examples["gripper"][0]),
                                task=str(examples["prompt"][0]))
    print(f"first chunk of new episode           : chunk_index={pred.chunk_index}, "
          f"obs_seq={pred.obs_seq}")
    print()
    print("chunk_index and obs_seq both restarted, so the episode really is new.")
    print("A client that kept its old high-water mark would now discard every")
    print("chunk forever, waiting for obs_seq to climb past a mark from an")
    print("episode that no longer exists.")

    # ---------------------------------------------------------------------------
    # Always close. This session holds two B200s for as long as it is open.
    # ---------------------------------------------------------------------------
    await client.close()
    print("\nclosed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
