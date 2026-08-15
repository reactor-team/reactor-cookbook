#!/usr/bin/env python3
"""LingBot-VA quickstart: drive the hosted LingBot-VA model from Python.

Run: python lingbot_va_quickstart.py, with REACTOR_API_KEY set.
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

    from reactor_robotics.lingbot_va import (
        ACTION_CHANNELS, ACTION_SHAPE, SEED_SKIP_STEPS, VIEWS, LingbotVaClient,
    )

    # Capacity: one B200 serves one session. A busy cluster returns HTTP 429
    # "no available capacity" on session creation. Wait and retry, or ask
    # Reactor for more capacity.
    client = LingbotVaClient()      # model="lingbot-va", 20 fps tracks, 128x128
    await client.connect()          # handlers -> connect -> await READY -> tracks -> ping

    print("status transitions :", " -> ".join(client.session.status_log))
    print("tracks published   :", ", ".join(client.session.tracks))
    print("endpoint           :", client.session.api_url)
    print("action shape       :", ACTION_SHAPE, "channels", ACTION_CHANNELS)
    print("seed skip          :", SEED_SKIP_STEPS, "rows on an episode's first chunk")
    print(f"frame window hold  : {client.window_s:.2f}s per chunk "
          f"({client.fps} fps tracks)")
    print("keepalive          : ping every 10s (runtime kills at 20s of silence)")

    import numpy as np

    EXAMPLES_PATH = "examples/lingbot_va_examples.npz"
    examples = np.load(EXAMPLES_PATH)
    N = len(examples["prompt"])

    # Five recorded examples: observations recorded against this same deployment
    # together with the chunks it returned then.
    #
    # The frames are synthetic and deterministic: LIBERO renders would be better
    # but are not cheaply obtainable (no PyPI package, from-source install, asset
    # downloads, torch<2.6), and frames from another embodiment would not be
    # LIBERO observations. So this is a protocol fixture, not a demonstration of
    # task behaviour. Task competence is the 98.5% LIBERO-Long figure, measured
    # in the real simulator -- which ../libero/ in this repo runs against
    # this same model. Full details: examples/PROVENANCE.md
    print(f"{N} recorded examples, frames keyed by track name (never positionally):")
    for v in VIEWS:
        print(f"  {v:<14} {examples[v].shape} {examples[v].dtype}")
    print(f"\ntask: {str(examples['prompt'][0])!r}")
    print(f"pinned seed row: {np.round(examples['seed_pinned_row'], 6).tolist()}")
    print(f"run-to-run exact: {bool(examples['run_to_run_exact'])}"
          f"   (so the checks below compare properties, not exact numbers)")

    import time

    live = []
    t_start = time.perf_counter()

    for i in range(N):
        frames = {v: examples[v][i] for v in VIEWS}   # keyed by track name

        # The first call starts the episode (task -> clear echo -> reset {}); every
        # call after it echoes the previous chunk's executable rows, which is what
        # a client that ran the whole chunk reports.
        pred = await client.predict(frames, task=str(examples["prompt"][i]))
        live.append(pred)

        print(
            f"example {i}: step={pred.step}  shape={pred.actions.shape}  "
            f"executable={pred.executable.shape[0]:>2} rows  "
            f"{pred.latency_ms:6.1f} ms  hold={pred.window_s:.2f}s"
        )

    print(f"\n{N} chunks in {time.perf_counter() - t_start:.1f}s")
    print(f"echo row counts sent : {client.echo_rows_sent}  (12 first, then 16)")
    print(f"echo duplicates      : {client.echo_duplicates}  (any non-zero stalls "
          f"the episode)")

    # ---------------------------------------------------------------------------
    # Deterministic checks; see the section above.
    # L2 spread at recording time: mean 1.08, standard deviation 0.36 over 3 passes.
    # Tolerances come from examples/PROVENANCE.md.
    # ---------------------------------------------------------------------------
    from reactor_robotics.lingbot_va import NORM_Q01, NORM_Q99

    L2_BAND = float(examples["l2_band"])
    QUANTILE_SLACK = 0.05      # rows may sit slightly outside the box; measured 0.008

    actions = np.stack([p.actions for p in live])          # (N, 16, 7)
    steps = [p.step for p in live]
    pinned = examples["seed_pinned_row"]

    checks = []
    checks.append((f"shape is {ACTION_SHAPE} on every chunk",
                   all(p.actions.shape == ACTION_SHAPE for p in live)))
    checks.append(("all actions finite", bool(np.isfinite(actions).all())))
    checks.append((f"step strictly increasing {steps}",
                   all(b > a for a, b in zip(steps, steps[1:]))))
    checks.append((f"episode starts at step 0", steps[0] == 0))
    checks.append((f"executable rows are [12, 16, 16, ...]",
                   [p.executable.shape[0] for p in live]
                   == [ACTION_SHAPE[0] - SEED_SKIP_STEPS]
                   + [ACTION_SHAPE[0]] * (N - 1)))

    # The pinned placeholder rows: an exact match, and identical across all 4.
    seed_rows = live[0].actions[:SEED_SKIP_STEPS]
    checks.append((f"seed chunk's {SEED_SKIP_STEPS} skipped rows are identical "
                   f"to each other",
                   bool(np.array_equal(seed_rows,
                                       np.broadcast_to(seed_rows[0], seed_rows.shape)))))
    checks.append(("pinned seed row matches the recorded one exactly",
                   bool(np.array_equal(seed_rows[0], pinned))))

    # Action semantics: deltas in raw LIBERO units live inside the quantile box.
    # Absolute poses would not, which is how a mis-wired client shows up.
    rows = actions.reshape(-1, ACTION_SHAPE[1])
    inside = bool((rows >= NORM_Q01 - QUANTILE_SLACK).all()
                  and (rows <= NORM_Q99 + QUANTILE_SLACK).all())
    checks.append((f"rows inside the training quantile box (+-{QUANTILE_SLACK}) "
                   f"-> deltas, not poses", inside))
    checks.append(("gripper channel within [-1, 1] (+-slack)",
                   bool((np.abs(rows[:, 6]) <= 1.0 + QUANTILE_SLACK).all())))
    checks.append(("zero echo duplicates", client.echo_duplicates == 0))
    checks.append(("no unexpected extra chunks",
                   all(not p.discarded for p in live)))

    print(f"{'check':<64} result")
    print("-" * 74)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"{label:<64} {'PASS' if passed else 'FAIL'}")
    print("-" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")

    # --- reported only ---------------------------------------------------------
    steady = [p.latency_ms for p in live[1:]]        # echo -> chunk
    seed_ms = live[0].latency_ms                     # reset -> chunk, not comparable

    print()
    print("Latency. The two are timed from different events, so they are never pooled:")
    print(f"  steady state (echo -> chunk) : p50 {np.median(steady):.1f} ms   "
          f"min {min(steady):.1f}   max {max(steady):.1f}")
    print(f"  seed chunk  (reset -> chunk) : {seed_ms:.1f} ms  "
          f"(also contains the server gathering its first 12 frames)")
    print(f"  reference at recording time  : p50 207.8 ms over 12 steady-state chunks")
    print()
    print(f"  Per OBSERVATION you also pay the {client.window_s:.2f}s frame-window")
    print("  hold, which is this model's frame budget rather than its latency: a")
    print("  real client is executing the previous chunk during that window, and")
    print("  its cameras are filling the window as a side effect. A LIBERO rollout")
    print("  at 20 Hz renders exactly 16 frames while executing 16 actions.")
    print()
    l2 = np.array([
        float(np.linalg.norm(live[i].actions - examples["expected_actions"][i]))
        for i in range(N)
    ])
    mag = float(np.mean([np.linalg.norm(a) for a in actions]))
    print(f"Run-to-run L2 vs the recorded chunks (band {L2_BAND:.3f}, measured from")
    print(f"the model's own spread; {100 * L2_BAND / mag:.0f}% of chunk magnitude "
          f"{mag:.2f}):")
    for i, d in enumerate(l2):
        print(f"  chunk {i}: {d:.4f}   {'ok' if d <= L2_BAND else 'OUTSIDE BAND'}")
    print(f"  worst {l2.max():.4f}")
    print()
    print("At 55% of chunk magnitude this band only catches gross drift, so it is")
    print("reported only; the deterministic checks above carry the result.")

    # ---------------------------------------------------------------------------
    # Negative tests: the two failure modes you are most likely to hit. ~35 s.
    # ---------------------------------------------------------------------------
    import asyncio

    # (1) An echo identical to the previous one reads as "nothing new"; no chunk.
    await client.session.send(
        "set_executed_action_json", {"executed_action_json": client.last_echo}
    )
    try:
        await client.session.next_message("action_prediction", timeout_s=12.0)
        print("unexpected: an identical echo was answered")
    except asyncio.TimeoutError:
        print("identical echo: no chunk in 12s, as documented.")
        print("=> the echo signals by changing. A repeat produces no reply at all.")

    # (2) The keepalive matters. The runtime disconnects a client that stays quiet
    #     for 20 s, and this protocol sends nothing while a robot executes a chunk.
    #     Sit idle for longer than that and confirm the session is still usable --
    #     session.py has been pinging every 10 s the whole time.
    print("\nsitting idle for 25 s to exercise the keepalive...")
    await asyncio.sleep(25.0)
    print("status after 25s idle:", " -> ".join(client.session.status_log))

    # A CHANGED echo recovers the episode, which also proves the session survived.
    recovered = await client.predict(
        {v: examples[v][0] for v in VIEWS}, task=str(examples["prompt"][0]),
        executed=live[-1].actions * 0.999,          # a changed value
    )
    print(f"recovered after 25s idle: step={recovered.step} "
          f"shape={recovered.actions.shape} {recovered.latency_ms:.1f} ms")
    print("=> 25 s of silence did not drop the session. Without the 10 s ping it "
          "would have.")

    # Always close. A live session holds a real GPU worker.
    await client.close()
    print("closed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
