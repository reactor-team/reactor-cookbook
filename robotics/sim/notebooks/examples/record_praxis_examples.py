# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record ``praxis_<embodiment>_examples.npz`` against the live Praxis-1 endpoint.

    uv run python examples/record_praxis_examples.py --embodiment droid
    uv run python examples/record_praxis_examples.py --embodiment yam

Three jobs:

1. **Record 5 observation -> chunk pairs** in one episode: the first chunk
   (no echo, 21 rows) and four echo-gated chunks (24 rows each).
2. **Calibrate the run-to-run band** by replaying the same episode twice more,
   each started with ``reset_context``, and measuring the per-chunk L2 spread
   of the poses.
3. **Calibrate the physical bands.** Targets are absolute end-effector poses,
   so row 0 must sit near the measured pose that was sent, and adjacent rows
   must be close. Each tolerance is the mean plus three standard deviations
   over every recorded chunk. Rotation orthonormality is recorded in the
   manifest only; the quickstart checks it against a fixed tolerance.

The observations come from ``synthetic_observation()`` in
``praxis_quickstart.py``: deterministic frames and a 25-row state history of a
constant-velocity reach. **The recorded actions are a protocol and invariant
fixture, not a demonstration of task behaviour.** See ``PROVENANCE.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

NOTEBOOKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOKS))

from praxis_quickstart import (  # noqa: E402
    CHUNK_ROWS,
    N_EXAMPLES,
    SPECS,
    connect,
    grip_key,
    pose_key,
    run_episode,
    synthetic_observation,
)


def _band(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std, mean + 3.0 * std


def _pad(arr: np.ndarray) -> np.ndarray:
    """Pad a chunk to 24 rows with NaN so 21- and 24-row chunks stack."""
    out = np.full((CHUNK_ROWS, *arr.shape[1:]), np.nan)
    out[: len(arr)] = arr
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embodiment", choices=sorted(SPECS), required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    spec = SPECS[args.embodiment]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    observations = [synthetic_observation(spec, i) for i in range(N_EXAMPLES)]
    t0 = time.perf_counter()
    session = await connect(spec, attempts=20, wait_s=60.0)
    ready_s = time.perf_counter() - t0
    passes: list[list[dict]] = []
    try:
        await session.send("set_task", {"task": spec.task})
        if spec.pose_frame:
            await session.send("set_pose_frame", {"pose_frame": spec.pose_frame})
        for r in range(args.repeats + 1):
            records = await run_episode(session, spec, observations, reset_first=r > 0)
            for i, rec in enumerate(records):
                print(
                    f"[run{r}] obs {i}: step={rec['step']} rows={rec['rows']} "
                    f"anchor={rec['anchor_m'] * 1000:.1f}mm dxyz={rec['dxyz_m'] * 1000:.1f}mm "
                    f"ortho={rec['ortho']:.2e} latency={rec['latency_ms']:.0f}ms",
                    flush=True,
                )
            passes.append(records)
    finally:
        await session.close()

    recorded = passes[0]
    rows = [rec["rows"] for rec in recorded]
    for p in passes:
        if [rec["rows"] for rec in p] != rows:
            raise SystemExit(f"row counts differ between passes: {[r['rows'] for r in p]} vs {rows}")

    l2 = [
        float(np.linalg.norm(
            np.concatenate([passes[a][i]["poses"][arm] - passes[b][i]["poses"][arm] for arm in spec.arms])
        ))
        for a in range(len(passes))
        for b in range(a + 1, len(passes))
        for i in range(N_EXAMPLES)
    ]
    every = [rec for p in passes for rec in p]
    bands = {
        "l2": _band(l2),
        "anchor_m": _band([rec["anchor_m"] for rec in every]),
        "dxyz_m": _band([rec["dxyz_m"] for rec in every]),
        "ortho": _band([rec["ortho"] for rec in every]),
    }

    payload: dict[str, np.ndarray] = {
        t: np.stack([obs[0][t] for obs in observations]) for t in spec.tracks
    }
    payload.update(
        task=np.asarray(spec.task),
        state_json=np.asarray(
            [json.dumps(obs[1], separators=(",", ":")) for obs in observations]
        ),
        rows=np.asarray(rows, dtype=np.int64),
        step=np.asarray([rec["step"] for rec in recorded], dtype=np.int64),
        l2_band=np.asarray(bands["l2"][2]),
        anchor_band_m=np.asarray(bands["anchor_m"][2]),
        dxyz_band_m=np.asarray(bands["dxyz_m"][2]),
    )
    for arm in spec.arms:
        payload["expected_" + pose_key(arm)] = np.stack([_pad(rec["poses"][arm]) for rec in recorded])
        payload["expected_" + grip_key(arm)] = np.stack([_pad(rec["grips"][arm]) for rec in recorded])

    out = NOTEBOOKS / "examples" / f"praxis_{args.embodiment}_examples.npz"
    np.savez_compressed(out, **payload)

    lat = [rec["latency_ms"] for p in passes for rec in p[1:]]
    manifest = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.model,
        "task": spec.task,
        "frames": (
            f"SYNTHETIC (deterministic, synthetic_observation()), "
            f"{spec.frame_hw[1]}x{spec.frame_hw[0]}, views {list(spec.tracks)}"
        ),
        "pose_frame": spec.pose_frame or "rig world frame",
        "rows": rows,
        "n_examples": N_EXAMPLES,
        "passes": len(passes),
        "ready_seconds": round(ready_s, 1),
        "out_bytes": out.stat().st_size,
        "recorded_run": [
            {
                "i": i,
                "step": rec["step"],
                "rows": rec["rows"],
                "latency_ms": round(rec["latency_ms"], 1),
                "anchor_m": round(rec["anchor_m"], 5),
                "dxyz_m": round(rec["dxyz_m"], 5),
                "ortho": round(rec["ortho"], 6),
            }
            for i, rec in enumerate(recorded)
        ],
        "bands": {
            name: {
                "n_samples": len(l2) if name == "l2" else len(every),
                "mean": mean,
                "std": std,
                "band_mean_plus_3sigma": band,
            }
            for name, (mean, std, band) in bands.items()
        },
        "latency_summary_ms": {
            "gate": "set_executed_step -> next chunk",
            "p50": round(float(np.median(lat)), 1),
            "min": round(min(lat), 1),
            "max": round(max(lat), 1),
            "n": len(lat),
            "chunk_ms": round(CHUNK_ROWS / 15 * 1000, 1),
        },
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["bands"], indent=2))
    print(json.dumps(manifest["latency_summary_ms"], indent=2))
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB) and {man.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
