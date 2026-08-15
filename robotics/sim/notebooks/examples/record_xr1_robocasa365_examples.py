# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record ``xr1_robocasa365_examples.npz`` against the live XR-1 RoboCasa365
deployment.

    REACTOR_API_KEY=... python examples/record_xr1_robocasa365_examples.py

Three jobs:

1. **Record 5 observation to chunk pairs**, exercising the echo-gated path that
   this model uses for every chunk including the first.
2. **Calibrate the run-to-run band**, by replaying the identical sequence twice
   more and measuring the per-chunk L2 spread. The policy samples, so replaying
   an observation does not reproduce its chunk and the band is what makes any
   later comparison meaningful.
3. **Calibrate the per-step magnitude band** over the chunk's live columns, so
   a later run can tell a plausible chunk from a broken one without knowing the
   task.

## The frames are synthetic

The observations here are a deterministic five-step kitchen approach at the
benchmark's native 256x256, with three genuinely different views: two agentview
cameras from different sides and a wrist close-up. ``synthetic_observation(i)``
is a pure function of ``i`` (no RNG, no clock), so re-running regenerates
identical inputs.

**The recorded actions are therefore not a behavioural demonstration.** They
are a protocol and invariant fixture. Task competence for this model comes from
the RoboCasa365 benchmark itself, where the same wire contract scored 56.8% and
60.2% (replan 16 and 8) against 56.4% and 59.2% for the vendor's own TCP socket
over 1000 paired episodes, with the vendor's published anchor at 57.28%.

## Why there is no anchor check here

This checkpoint emits the vendor's packed layout, decoded with its own
per-step relative-action statistics, so there is no absolute pose to anchor
a chunk's first row against. The invariants recorded
instead are shape, finiteness, a strictly increasing ``step``, the per-step
magnitude band and the run-to-run L2 band.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from reactor_robotics.xr1_robocasa365 import (
    ACTION_SHAPE,
    LIVE_DIMS,
    OBS_HISTORY,
    STATE_ROW_DIM,
    TRACKS,
    Xr1Robocasa365Client,
)

TASK = "close the blender lid"

FRAME_HW = (256, 256)

log = logging.getLogger("record_xr1_robocasa365")


def _rect(img, y0, y1, x0, x1, colour) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = colour


def _disc(img, cy, cx, ry, rx, colour) -> None:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    img[((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0] = colour


def synthetic_observation(i: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Observation ``i`` of a deterministic five-step kitchen approach.

    Returns ``(frames, state_row)``. Pure function of ``i``. The three views are
    genuinely different scenes, not copies: a left agentview, a right agentview
    at a different angle and warmer light, and a wrist close-up.
    """
    h, w = FRAME_HW
    t = i / 4.0

    def counter() -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.38)
        img[:horizon] = (206, 208, 212)          # backsplash
        img[horizon:] = (176, 152, 120)          # worktop
        grade = np.linspace(1.0, 0.82, h - horizon, dtype=np.float32)[:, None, None]
        img[horizon:] = (img[horizon:] * grade).astype(np.uint8)
        return img

    # left_agentview: camera on the left, gripper closing from the left.
    left = counter()
    lid_x = int(w * (0.30 + 0.34 * t))
    _disc(left, int(h * 0.66), int(w * 0.62), int(h * 0.10), int(w * 0.09),
          (228, 230, 236))
    _rect(left, int(h * 0.50), int(h * 0.70), int(w * 0.55), int(w * 0.70),
          (120, 124, 132))
    grip_y = int(h * (0.22 + 0.28 * t))
    _rect(left, grip_y - 12, grip_y + 24, lid_x - 16, lid_x - 7, (52, 54, 60))
    _rect(left, grip_y - 12, grip_y + 24, lid_x + 7, lid_x + 16, (52, 54, 60))

    # right_agentview: the other side of the counter, mirrored geometry and
    # warmer light, so the two agentviews are not the same picture.
    right = counter()
    right[int(h * 0.38):] = (right[int(h * 0.38):] * 0.90).astype(np.uint8)
    lid_x2 = w - lid_x
    _disc(right, int(h * 0.62), int(w * 0.38), int(h * 0.095), int(w * 0.085),
          (228, 230, 236))
    _rect(right, int(h * 0.47), int(h * 0.66), int(w * 0.30), int(w * 0.45),
          (120, 124, 132))
    _rect(right, grip_y - 10, grip_y + 22, lid_x2 - 14, lid_x2 - 6, (52, 54, 60))
    _rect(right, grip_y - 10, grip_y + 22, lid_x2 + 6, lid_x2 + 14, (52, 54, 60))

    # wrist_view: looking down, closing in on the lid.
    wrist = np.zeros((h, w, 3), dtype=np.uint8)
    wrist[:] = (176, 152, 120)
    scale = 1.0 + 1.5 * t
    _disc(wrist, int(h * 0.54), w // 2, int(26 * scale), int(34 * scale),
          (228, 230, 236))
    _rect(wrist, 0, h, 0, int(w * 0.11), (52, 54, 60))
    _rect(wrist, 0, h, int(w * 0.89), w, (52, 54, 60))

    frames = {"left_agentview": left, "right_agentview": right, "wrist_view": wrist}
    assert set(frames) == set(TRACKS), frames.keys()

    # One state row: [0:3] left EE xyz, [3:6] left EE axis-angle, [6] left
    # gripper, [7:10] right EE xyz, [10:13] right EE axis-angle, [13] right
    # gripper. This embodiment is single-arm, so the right block stays at rest.
    state = np.zeros(STATE_ROW_DIM, dtype=np.float64)
    state[0:3] = [0.42 + 0.06 * i, -0.10 + 0.03 * i, 0.94 - 0.04 * i]
    state[3:6] = [0.02 * i, -0.01 * i, 0.03 * i]
    state[6] = round(min(1.0, 0.05 + 0.2 * i), 3)
    return frames, state


def state_window(i: int) -> np.ndarray:
    """The OBS_HISTORY rows that pair with observation ``i``.

    Oldest first, one environment step apart here, clamped to the earliest row
    while the episode is younger than the window. That clamping is upstream's
    own behaviour at the start of an episode, not an approximation.
    """
    rows = [
        synthetic_observation(max(0, i - k))[1]
        for k in range(OBS_HISTORY - 1, -1, -1)
    ]
    return np.stack(rows)


async def connect_with_capacity_retry(
    client: Xr1Robocasa365Client, attempts: int, wait_s: float
) -> None:
    """Connect, retrying while the platform reports no free capacity.

    This model serves one session at a time, so a busy or not-yet-scaled
    deployment answers session creation with ``429 no available capacity``
    rather than queueing. Wait and retry; it is not a client error.
    """
    for attempt in range(1, attempts + 1):
        try:
            await client.connect()
            return
        except Exception as exc:
            transient = "429" in str(exc) or "no available capacity" in str(exc)
            if not transient or attempt == attempts:
                raise
            print(
                f"[connect] attempt {attempt}/{attempts}: no capacity free; "
                f"retrying in {wait_s:.0f}s",
                flush=True,
            )
            await asyncio.sleep(wait_s)


async def one_pass(client: Xr1Robocasa365Client, n: int, label: str) -> list[dict]:
    """Drive the observation sequence once, on a freshly connected client.

    ``reset`` first anyway: the model carries an observation history and a
    flow-control counter per session, and a new pass
    is a new episode. On a fresh connection this is a no-op, which is exactly
    what it should be, and it keeps the pass reproducible if a caller ever
    reuses a client.
    """
    await client.reset()
    out = []
    for i in range(n):
        frames, _ = synthetic_observation(i)
        pred = await client.predict(frames, state_window(i), TASK)
        live = pred.live
        # PER-COLUMN, not a single number over the whole chunk. The gripper
        # column swings the full [-1, 1] range, so one global band would be set
        # by that column alone and would say nothing about the other eleven.
        step_mag = (
            np.max(np.abs(np.diff(live, axis=0)), axis=0)
            if len(live) > 1
            else np.zeros(LIVE_DIMS)
        )
        out.append(
            {
                "i": i,
                "actions": pred.actions,
                "step": pred.step,
                "latency_ms": pred.latency_ms,
                "discarded": len(pred.discarded),
                "step_delta": np.asarray(step_mag, dtype=np.float64),
            }
        )
        print(
            f"[{label}] obs {i}: step={pred.step} shape={pred.actions.shape} "
            f"max|step delta|={float(np.max(step_mag)):.4f} "
            f"(col {int(np.argmax(step_mag))}) "
            f"latency={pred.latency_ms:.0f}ms discarded={len(pred.discarded)}",
            flush=True,
        )
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/xr1_robocasa365_examples.npz")
    ap.add_argument("--n", type=int, default=5)
    # 4 repeats, not 2: each pass is now its own session (see main), and 5
    # sessions x 5 observations = 25 samples per column is the smallest basis
    # that gave a band a fresh run did not immediately blow through.
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--connect-attempts", type=int, default=20)
    ap.add_argument("--connect-wait-s", type=float, default=60.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # ONE SESSION PER PASS, deliberately. This model varies more between
    # sessions than within one, so a band measured inside a single session
    # does not describe the quantity the check compares against: the check
    # always runs in a fresh session. Reconnecting per pass makes the
    # calibration sample the same thing the check measures.
    passes = []
    ready_s = 0.0
    labels = ["record"] + [f"repeat{r + 1}" for r in range(args.repeats)]
    for n_pass, label in enumerate(labels):
        client = Xr1Robocasa365Client()
        t_connect = time.perf_counter()
        await connect_with_capacity_retry(
            client, args.connect_attempts, args.connect_wait_s
        )
        if n_pass == 0:
            # Connect-to-READY for the first session, recorded in the
            # manifest as `ready_seconds`.
            ready_s = time.perf_counter() - t_connect
        try:
            passes.append(await one_pass(client, args.n, label))
        finally:
            await client.close()

    reference = passes[0]
    chunks = np.stack([p["actions"] for p in reference])

    # Run-to-run spread: the same observations replayed, so this is the
    # policy's own sampling noise and nothing else. Mean + 3 sigma.
    l2 = [
        float(np.linalg.norm(passes[k][i]["actions"] - reference[i]["actions"]))
        for k in range(1, len(passes))
        for i in range(args.n)
    ]
    l2_mean = float(np.mean(l2)) if l2 else 0.0
    l2_std = float(np.std(l2)) if l2 else 0.0
    l2_band = l2_mean + 3.0 * l2_std

    # Per-column band, shape (LIVE_DIMS,).
    #
    # Mean + 3 sigma alone is not usable here: on this model that bound can
    # come out WIDER than a column's entire observed range, which makes the
    # check unfailable. Several columns are bimodal (they hold, or flip the
    # full [-1, 1], as columns 6 and 11 do) and a Gaussian bound on a bimodal
    # variable is meaningless.
    #
    # So the band is clamped to the column's observed span: a single step can
    # never move a column further than the full range that column was seen to
    # occupy, which keeps the bound a real check rather than a decoration.
    # Both components are recorded so the choice stays auditable.
    step_deltas = np.stack([p["step_delta"] for pass_ in passes for p in pass_])
    step_band_3sigma = np.mean(step_deltas, axis=0) + 3.0 * np.std(step_deltas, axis=0)
    live_all = np.stack(
        [p["actions"][:, :LIVE_DIMS] for pass_ in passes for p in pass_]
    )
    col_span = live_all.max(axis=(0, 1)) - live_all.min(axis=(0, 1))
    step_band = np.minimum(step_band_3sigma, col_span)
    clamped = [c for c in range(LIVE_DIMS) if step_band_3sigma[c] > col_span[c]]

    latencies = [p["latency_ms"] for pass_ in passes for p in pass_]

    obs = [synthetic_observation(i) for i in range(args.n)]
    payload = {track: np.stack([o[0][track] for o in obs]) for track in TRACKS}
    # Key names follow the conventions of the fixtures in examples/:
    # `expected_actions` for the recorded chunks, each band carried with the
    # mean/std it was built from, so one script can read every fixture.
    payload.update(
        prompt=np.asarray([TASK] * args.n),
        state_history=np.stack([state_window(i) for i in range(args.n)]),
        expected_actions=chunks,
        step=np.asarray([p["step"] for p in reference]),
        l2_band=np.asarray(l2_band),
        l2_band_mean=np.asarray(l2_mean),
        l2_band_std=np.asarray(l2_std),
        step_delta_band=np.asarray(step_band, dtype=np.float64),
        step_delta_band_3sigma=np.asarray(step_band_3sigma, dtype=np.float64),
        step_delta_band_span=np.asarray(col_span, dtype=np.float64),
        run_to_run_exact=np.asarray(bool(l2) and max(l2) == 0.0),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    # Field names and shapes follow the manifests in examples/: model/task/
    # n_examples/passes/recorded_utc/recorded_run/out_bytes/ready_seconds/
    # latency_ms/action_l2_norms are common to all of them, and the
    # model-specific extras sit alongside rather than replacing any of them.
    manifest = {
        "model": "reactor/xr1-robocasa365",
        "task": TASK,
        "n_examples": args.n,
        "passes": len(passes),
        "repeats": args.repeats,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ready_seconds": round(ready_s, 1),
        "out_bytes": out.stat().st_size,
        "tracks": list(TRACKS),
        "frame_hw": list(FRAME_HW),
        "action_shape": list(ACTION_SHAPE),
        "live_dims": LIVE_DIMS,
        "obs_history": OBS_HISTORY,
        "state_row_dim": STATE_ROW_DIM,
        "frames": "synthetic, deterministic; a protocol fixture, not a demo",
        "action_l2_norms": [
            round(float(np.linalg.norm(p["actions"])), 4) for p in reference
        ],
        "band": {
            "l2_band": l2_band,
            "step_delta_band_per_column": [float(v) for v in step_band],
            "step_delta_band_3sigma": [float(v) for v in step_band_3sigma],
            "observed_column_span": [float(v) for v in col_span],
            "span_clamped_columns": clamped,
            "step_delta_band_note": (
                "one band per live column, min(mean + 3 sigma, observed span). "
                "Per column because columns 6 and 11 swing the full [-1, 1] "
                "and a single global band would be set by them alone. Clamped "
                "to the span because on this sample size the 3 sigma bound "
                "exceeded the column's whole range on most columns, making the "
                "check unfailable; the clamped columns are listed above."
            ),
            "l2_band_mean": l2_mean,
            "l2_band_std": l2_std,
            "samples_per_column": int(step_deltas.shape[0]),
            "pairs": len(l2),
        },
        "latency_ms": {
            f"run{k}": [round(p["latency_ms"], 1) for p in pass_]
            for k, pass_ in enumerate(passes)
        },
        "latency_summary_ms": {
            "n": len(latencies),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        },
        "recorded_run": [
            {
                "i": p["i"],
                "step": p["step"],
                "latency_ms": round(p["latency_ms"], 1),
                "discarded": p["discarded"],
                "max_step_delta": round(float(np.max(p["step_delta"])), 4),
                "worst_column": int(np.argmax(p["step_delta"])),
            }
            for p in reference
        ],
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {out} and {man}")
    print(f"\n{'col':>4} {'3-sigma':>9} {'span':>9} {'band':>9}  source")
    for c in range(LIVE_DIMS):
        src = "span (3-sigma was unfailable)" if c in clamped else "3-sigma"
        print(f"{c:>4} {step_band_3sigma[c]:9.4f} {col_span[c]:9.4f} "
              f"{step_band[c]:9.4f}  {src}")
    print(f"\nrun-to-run L2 band: {l2_band:.4f} (reported only)")
    print(json.dumps(manifest["latency_summary_ms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
