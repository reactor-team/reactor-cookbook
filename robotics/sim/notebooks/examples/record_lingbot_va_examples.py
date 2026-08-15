# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record ``lingbot_va_examples.npz`` against the live LingBot-VA deployment.

    REACTOR_API_KEY=... python examples/record_lingbot_va_examples.py

Two jobs:

1. **Record 5 observation → chunk pairs** inside one episode, so the fixture
   exercises the seed chunk's 4-row skip, the 12-then-16-row echo pattern, and
   ``step`` incrementing across a real closed-loop sequence.
2. **Calibrate the run-to-run band.** The script replays the identical
   observation sequence twice more in fresh episodes and measures the per-chunk
   L2 distance between runs. Whether that spread is zero decides what the
   notebook is allowed to claim, so it is measured rather than assumed.

## The frames are synthetic

LIBERO renders would be better and are not cheaply obtainable: LIBERO is not on
PyPI, needs a from-source install plus asset downloads and `torch<2.6`, and no
recorded LIBERO clip is vendored anywhere. Rather than pass off frames from a
different embodiment as LIBERO observations, ``synthetic_observation(i)`` builds
a deterministic five-step reaching sequence at LIBERO's native 128×128 — a
tabletop with a bowl and a plate from a fixed third-person camera
(``agentview``) plus a wrist close-up (``eye_in_hand``). Pure function of ``i``:
no RNG, no clock, so re-running regenerates identical inputs.

**The recorded actions are therefore not a behavioural demonstration.** They
are a protocol and invariant fixture. Task competence is the published **98.5%
on LIBERO-Long**, measured in the real simulator — which
[`../libero/`](../libero) in this repo runs against this same model.

To close that gap, record against real renders instead:

    python -m libero_sim.main --task-id 0 --record out.mp4   # in ../libero
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from reactor_robotics.lingbot_va import (
    ACTION_SHAPE,
    CAM_SIZE,
    SEED_SKIP_STEPS,
    VIEWS,
    LingbotVaClient,
)

#: A LIBERO-phrasing instruction matching the synthetic scene.
TASK = "pick up the black bowl and place it on the plate"

log = logging.getLogger("record_lingbot_va")


def _rect(img, y0, y1, x0, x1, colour) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = colour


def _disc(img, cy, cx, ry, rx, colour) -> None:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
    img[mask] = colour


def synthetic_observation(i: int) -> dict[str, np.ndarray]:
    """Observation ``i`` of a deterministic five-step reach, at 128×128.

    Pure function of ``i``. Returns frames keyed by track name.
    """
    n = CAM_SIZE
    t = i / 4.0  # 0 -> 1 across the five observations

    # ---- agentview: fixed third-person camera on a tabletop.
    av = np.zeros((n, n, 3), dtype=np.uint8)
    horizon = int(n * 0.38)
    av[:horizon] = (196, 198, 203)                     # back wall
    av[horizon:] = (172, 138, 96)                      # table
    grade = np.linspace(1.0, 0.80, n - horizon, dtype=np.float32)[:, None, None]
    av[horizon:] = (av[horizon:] * grade).astype(np.uint8)
    _disc(av, int(n * 0.80), int(n * 0.70), int(n * 0.07), int(n * 0.15),
          (238, 238, 242))                             # plate, on the right
    bowl_x = int(n * (0.30 + 0.22 * t))                # bowl, travelling right
    _disc(av, int(n * 0.72), bowl_x, int(n * 0.08), int(n * 0.10), (38, 38, 44))
    grip_y = int(n * (0.22 + 0.30 * t))                # gripper descending
    _rect(av, grip_y - 9, grip_y + 20, bowl_x - 14, bowl_x - 7, (58, 60, 68))
    _rect(av, grip_y - 9, grip_y + 20, bowl_x + 7, bowl_x + 14, (58, 60, 68))

    # ---- eye_in_hand: wrist camera, closing in as the gripper drops.
    eh = np.zeros((n, n, 3), dtype=np.uint8)
    eh[:] = (172, 138, 96)
    scale = 1.0 + 1.5 * t
    _disc(eh, int(n * 0.56), n // 2, int(20 * scale), int(24 * scale), (38, 38, 44))
    _rect(eh, 0, n, 0, int(n * 0.11), (58, 60, 68))    # finger, left
    _rect(eh, 0, n, int(n * 0.89), n, (58, 60, 68))    # finger, right

    frames = {"agentview": av, "eye_in_hand": eh}
    assert set(frames) == set(VIEWS), frames.keys()
    return frames


async def connect_with_capacity_retry(
    client: LingbotVaClient, attempts: int, wait_s: float
) -> None:
    """Connect, retrying while the platform reports no free capacity.

    A busy cluster answers session creation with ``429 no available capacity``
    rather than queueing. That is a wait-and-retry condition, not a client bug,
    and worth retrying because a recording run is unattended.
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
                f"[connect] attempt {attempt}/{attempts}: no B200 capacity free; "
                f"retrying in {wait_s:.0f}s",
                flush=True,
            )
            await asyncio.sleep(wait_s)


async def one_pass(client: LingbotVaClient, n: int, label: str) -> list[dict]:
    """Drive the observation sequence once in a fresh episode."""
    await client.start_episode(TASK)
    out = []
    for i in range(n):
        frames = synthetic_observation(i)
        pred = await client.predict(frames, TASK)
        out.append(
            {
                "i": i,
                "actions": pred.actions,
                "executable_rows": int(pred.executable.shape[0]),
                "step": pred.step,
                "latency_ms": pred.latency_ms,
                "discarded": list(pred.discarded),
            }
        )
        print(
            f"[{label}] obs {i}: step={pred.step} shape={pred.actions.shape} "
            f"executable={pred.executable.shape[0]} "
            f"latency={pred.latency_ms:.0f}ms window={pred.window_s:.2f}s",
            flush=True,
        )
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/lingbot_va_examples.npz")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument(
        "--repeats", type=int, default=2,
        help="extra passes over the same observations, for the band",
    )
    ap.add_argument("--connect-attempts", type=int, default=20)
    ap.add_argument("--connect-wait-s", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    client = LingbotVaClient()
    passes: list[list[dict]] = []
    try:
        t0 = time.perf_counter()
        await connect_with_capacity_retry(
            client, args.connect_attempts, args.connect_wait_s
        )
        ready_s = time.perf_counter() - t0
        print(f"[connect] READY after {ready_s:.1f}s", flush=True)
        for r in range(args.repeats + 1):
            passes.append(await one_pass(client, args.n, f"run{r}"))
    finally:
        await client.close()

    recorded = passes[0]

    # ---- band from run-to-run spread ---------------------------------------
    dists: list[float] = []
    pairs = []
    for a in range(len(passes)):
        for b in range(a + 1, len(passes)):
            per_chunk = [
                float(np.linalg.norm(passes[a][i]["actions"] - passes[b][i]["actions"]))
                for i in range(args.n)
            ]
            pairs.append({"runs": [a, b], "per_chunk_l2": per_chunk})
            dists.extend(per_chunk)
    mean = float(np.mean(dists))
    std = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
    band = mean + 3.0 * std
    exact = max(dists) == 0.0

    norms = [float(np.linalg.norm(r["actions"])) for r in recorded]
    # Seed and steady-state latencies measure different gates (`reset` vs the
    # echo), so they are summarised separately and never pooled.
    seed_lat = [p[0]["latency_ms"] for p in passes]
    steady_lat = [r["latency_ms"] for p in passes for r in p[1:]]

    # Is the pinned seed slot really identical across passes? That is the
    # one part of this model's output a replay can gate numerically.
    seed_rows = np.stack([p[0]["actions"][:SEED_SKIP_STEPS] for p in passes])
    seed_pinned_exact = bool(
        np.array_equal(seed_rows, np.broadcast_to(seed_rows[0], seed_rows.shape))
    )
    seed_row_identical_within = bool(
        np.array_equal(
            seed_rows[0], np.broadcast_to(seed_rows[0][0], seed_rows[0].shape)
        )
    )

    frames_by_view: dict[str, list[np.ndarray]] = {v: [] for v in VIEWS}
    for i in range(args.n):
        f = synthetic_observation(i)
        for v in VIEWS:
            frames_by_view[v].append(f[v])

    payload = {v: np.stack(frames_by_view[v]) for v in VIEWS}
    payload.update(
        prompt=np.asarray([TASK] * args.n),
        expected_actions=np.stack([r["actions"] for r in recorded]),
        step=np.asarray([r["step"] for r in recorded], dtype=np.int64),
        executable_rows=np.asarray(
            [r["executable_rows"] for r in recorded], dtype=np.int64
        ),
        l2_band=np.asarray(band),
        l2_band_mean=np.asarray(mean),
        l2_band_std=np.asarray(std),
        run_to_run_exact=np.asarray(exact),
        seed_skip_steps=np.asarray(SEED_SKIP_STEPS),
        #: The pinned conditioning slot, and whether it reproduced exactly
        #: across all passes. The notebook gates on this numerically.
        seed_pinned_row=np.asarray(recorded[0]["actions"][0]),
        seed_pinned_exact=np.asarray(seed_pinned_exact),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    manifest = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "reactor/lingbot-va",
        "task": TASK,
        "frames": f"SYNTHETIC (deterministic, synthetic_observation()), "
                  f"{CAM_SIZE}x{CAM_SIZE} — LIBERO's native camera size",
        "action_shape": list(ACTION_SHAPE),
        "seed_skip_steps": SEED_SKIP_STEPS,
        "n_examples": args.n,
        "passes": len(passes),
        "ready_seconds": round(ready_s, 1),
        "out_bytes": out.stat().st_size,
        "recorded_run": [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if k != "actions"}
            for r in recorded
        ],
        "action_l2_norms": [round(v, 4) for v in norms],
        "band": {
            "pairs": pairs,
            "n_samples": len(dists),
            "mean_l2": mean,
            "std_l2": std,
            "band_mean_plus_3sigma": band,
            "max_observed_l2": max(dists),
            "run_to_run_exact": exact,
        },
        "latency_ms": {
            f"run{r}": [round(g["latency_ms"], 1) for g in p]
            for r, p in enumerate(passes)
        },
        "latency_steady_state_ms": {
            "gate": "echo -> chunk",
            "p50": round(float(np.median(steady_lat)), 1),
            "min": round(min(steady_lat), 1),
            "max": round(max(steady_lat), 1),
            "n": len(steady_lat),
        },
        "latency_seed_chunk_ms": {
            "gate": "reset -> chunk (includes gathering the first 12 frames)",
            "p50": round(float(np.median(seed_lat)), 1),
            "values": [round(v, 1) for v in seed_lat],
        },
        "seed_pinned_row": [round(float(v), 6) for v in recorded[0]["actions"][0]],
        "seed_pinned_exact_across_passes": seed_pinned_exact,
        "seed_rows_identical_within_chunk": seed_row_identical_within,
        "echo_duplicates": client.echo_duplicates,
        "echo_rows_sent": client.echo_rows_sent,
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2))
    print("\n=== BAND CALIBRATION ===")
    print(json.dumps(manifest["band"], indent=2))
    print("\n=== LATENCY ===")
    print(json.dumps(manifest["latency_steady_state_ms"], indent=2))
    print(json.dumps(manifest["latency_seed_chunk_ms"], indent=2))
    print("\n=== PINNED SEED SLOT ===")
    print(f"row: {manifest['seed_pinned_row']}")
    print(f"identical across all {len(passes)} passes: {seed_pinned_exact}")
    print(f"all {SEED_SKIP_STEPS} skipped rows identical: {seed_row_identical_within}")
    print(f"\nrun-to-run exact: {exact}   band (mean + 3s) = {band:.5f}")
    print(f"chunk L2 norms {[round(v, 3) for v in norms]}")
    print(f"band is {100 * band / float(np.mean(norms)):.0f}% of mean chunk magnitude")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.1f} kB) and {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
