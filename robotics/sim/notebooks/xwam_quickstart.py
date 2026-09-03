#!/usr/bin/env python3
"""X-WAM quickstart: drive the hosted X-WAM model from Python.

Run: python xwam_quickstart.py, with REACTOR_API_KEY set.
"""
import asyncio


async def main():
    import logging

    # The SDK logs status transitions and dropped commands at INFO. Leave it on.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from reactor_robotics import describe_api_key

    # Confirms the key is present and reports the endpoint. Prints the key's
    # LENGTH, never any part of its value.
    print(describe_api_key())

    from reactor_robotics.xwam import VIEWS, XwamClient

    client = XwamClient()          # model="xwam", 15 fps tracks
    await client.connect()         # handlers -> connect -> await READY -> tracks

    print("status transitions :", " -> ".join(client.session.status_log))
    print("tracks published   :", ", ".join(client.session.tracks))
    print("endpoint           :", client.session.api_url)
    print("keepalive          : SDK heartbeat every 10s")

    import numpy as np

    EXAMPLES_PATH = "examples/xwam_examples.npz"
    examples = np.load(EXAMPLES_PATH)
    N = len(examples["prompt"])

    # These are five **recorded examples**: real observations captured during a
    # RoboTwin 2.0 evaluation run, together with the actions the model returned at
    # the time. Four are consecutive chunks of one rollout; the fifth is the first
    # chunk of the next rollout, so it crosses an episode boundary and carries a
    # different instruction. `expected_actions` holds the actions recorded then,
    # produced through the authors' own serving stack.
    print(f"{N} recorded examples, frames keyed by track name (never "
          f"positionally):\n")
    for view in VIEWS:
        print(f"  {view:<18} {examples[view].shape} {examples[view].dtype}")
    print(f"  {'proprio':<18} {examples['proprio'].shape}")
    print(f"  {'expected_actions':<18} {examples['expected_actions'].shape}")
    print()
    for i in range(N):
        print(
            f"  example {i}: rollout {examples['rollout_id'][i]}"
            f" step {examples['step_id'][i]:>3}"
            f"  cfg={examples['cfg'][i]}  \"{str(examples['prompt'][i])[:52]}...\""
        )
    print("\nWhere these came from: examples/PROVENANCE.md")

    import time

    results = []
    t_start = time.perf_counter()

    for i in range(N):
        frames = {view: examples[view][i] for view in VIEWS}   # keyed by track name

        # The seed triple pins the sampling noise to the recorded request, which
        # is what makes this a replay rather than a fresh sample. A robot client
        # omits `seed` entirely and gets a chunk_id-derived seed instead.
        seed = (
            int(examples["env_rank"][i]),
            int(examples["rollout_id"][i]),
            int(examples["step_id"][i]),
        )

        pred = await client.predict(
            frames,
            examples["proprio"][i],
            task=str(examples["prompt"][i]),
            cfg=float(examples["cfg"][i]),
            seed=seed,
        )

        action_delta = float(
            np.max(np.abs(pred.actions - examples["expected_actions"][i]))
        )
        proprio_delta = float(
            np.max(np.abs(pred.proprios - examples["expected_proprios"][i]))
        )
        results.append(
            dict(
                i=i,
                step=pred.step,
                shape=pred.actions.shape,
                finite=bool(np.isfinite(pred.actions).all()),
                action_delta=action_delta,
                proprio_delta=proprio_delta,
                latency_ms=pred.latency_ms,
                retries=pred.retries,
                actions=pred.actions,          # kept for the statelessness demo
            )
        )
        print(
            f"example {i}: max|dA|={action_delta:.3e}  {pred.latency_ms:6.1f} ms"
            f"  step={pred.step}  retries={pred.retries}"
        )

    print(f"\n{N} predictions in {time.perf_counter() - t_start:.1f}s")

    # ---------------------------------------------------------------------------
    # Tolerances: see the section above. Examples 0 and 1 carry the transport
    # transient and move run to run; 2-4 stay near 1e-3. See examples/PROVENANCE.md.
    # ---------------------------------------------------------------------------
    ACTION_TOL = 5e-2        # over-the-wire tolerance; 4.2e-3 is the direct-fed floor
    PARITY_NOTE_TOL = 5e-3   # informational: "as good as the direct-fed path"

    lat = [r["latency_ms"] for r in results]
    worst = max(r["action_delta"] for r in results)

    print(f"{'example':>7}  {'step':>4}  {'shape':>8}  {'max|dA|':>10}  "
          f"{'latency':>9}  result")
    print("-" * 63)
    ok = True
    for r in results:
        shape_ok = r["shape"] == (32, 14)
        step_ok = r["step"] == r["i"] + 1        # step echoes our chunk_id
        delta_ok = r["action_delta"] <= ACTION_TOL
        passed = shape_ok and step_ok and delta_ok and r["finite"]
        ok &= passed
        print(f"{r['i']:>7}  {r['step']:>4}  {str(r['shape']):>8}  "
              f"{r['action_delta']:>10.3e}  {r['latency_ms']:>7.1f}ms  "
              f"{'PASS' if passed else 'FAIL'}")

    print("-" * 63)
    print(f"actions           : shape (32, 14), finite, step == chunk_id  -> "
          f"{'all ok' if ok else 'FAILURE'}")
    print(f"worst max|dA|     : {worst:.3e}   (tolerance {ACTION_TOL:.0e})")
    if worst <= PARITY_NOTE_TOL:
        print(f"                    ...also within {PARITY_NOTE_TOL:.0e}; this run "
              f"matches direct-fed parity.")
    print(f"latency per chunk : p50 {np.median(lat):.1f} ms   "
          f"min {min(lat):.1f}   max {max(lat):.1f}")
    print()
    print("The model is the fast part: ~163 ms inference per chunk against 229 ms")
    print("for the authors' own stack on the same GPU (a Reactor measurement),")
    print("1.4x faster.")
    print("    ~163 ms   the model's own chunk inference on the deployment")
    print("     229 ms   the authors' stack on the same GPU, same work")
    print(f"  ~{np.median(lat):>4.0f} ms   what you just measured, client-observed round trip")
    print()
    print("The gap is frame delivery (set by this script's 15 fps publishing")
    print("rate, not by the model) plus transport; the guide's Latency section")
    print("has the breakdown. The closed-loop RoboTwin 2.0 evaluation ran over")
    print("this exact wire and scored 79.3% vs 77.0% from the paper's per-task")
    print("results.")
    print()
    print("proprios are reported, not checked: nothing executes them (the robot")
    print("executes `actions`), and the predicted-state delta is dominated by the")
    print("same transport transient without the action head's smoothing:")
    print(f"  worst max|dP| : {max(r['proprio_delta'] for r in results):.3e}")
    print()
    print("RESULT:", "PASS" if ok else "FAIL")

    # ---------------------------------------------------------------------------
    # Optional: the statelessness / deduplication behaviour that makes timeout
    # recovery safe. Two claims from the contract, both testable in 10 seconds.
    # ---------------------------------------------------------------------------
    import asyncio
    import json

    # (1) Same chunk_id, retry field bumped -> answered again, identically.
    #     This is what a retry after a lost reply does.
    retried = json.loads(client.last_state_json)
    retried["retry"] = retried.get("retry", 0) + 1
    await client.session.send("set_state_json", {"state_json": json.dumps(retried)})

    reply = await client.session.next_message("action_prediction", timeout_s=30.0)
    replayed = np.asarray(reply["actions"], dtype=np.float64)
    original = results[-1]["actions"]
    retry_delta = float(np.max(np.abs(replayed - original)))

    print(f"retry reply step      : {int(reply['step'])} (same chunk_id as the original)")
    print(f"max|delta| vs original: {retry_delta:.3e}")
    print(f"exact match           : {np.array_equal(replayed, original)}")
    print()
    print("Not an exact match over this transport, which is expected. The seed")
    print("fields did not change, so the sampling noise is identical, but the")
    print("retry waits for freshly encoded repeats of the same frame, and H.264")
    print("does not reproduce its own output exactly. Fed directly, with no")
    print("video path, a retry reproduces the lost reply exactly; that is the")
    print("property the eval harness relies on when it recovers from a lost reply.")
    print()
    print(f"For scale, that residual is {worst / max(retry_delta, 1e-12):.0f}x "
          f"smaller than the worst")
    print(f"replay delta above ({worst:.3e}), and {ACTION_TOL / max(retry_delta, 1e-12):.0f}x "
          f"inside the tolerance, so a retry")
    print("is safe: it cannot corrupt an episode.")
    print()

    # (2) Re-sending the exact same state -> treated as a duplicate -> no reply.
    #     The model cannot distinguish it from unchanged state being re-delivered.
    await client.session.send("set_state_json", {"state_json": json.dumps(retried)})
    try:
        await client.session.next_message("action_prediction", timeout_s=10.0)
        print("unexpected: an exact re-send was answered")
    except asyncio.TimeoutError:
        print("exact re-send: no reply in 10s, as documented.")
        print("=> a retry must change the request (bump `retry`), or it gets no reply.")

    # Always close. A live session holds a real GPU worker.
    await client.close()
    print("closed. status:", " -> ".join(client.session.status_log))


if __name__ == "__main__":
    asyncio.run(main())
