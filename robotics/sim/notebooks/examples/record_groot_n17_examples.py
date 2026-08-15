# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record ``groot_n17_examples.npz`` against the live GR00T N1.7 deployment.

    REACTOR_API_KEY=... python examples/record_groot_n17_examples.py

Three jobs:

1. **Record 5 observation → chunk pairs**, so the fixture exercises the frame
   window, the free-running gate, and ``step`` incrementing.
2. **Calibrate the run-to-run band.** GR00T's action head samples, so replaying
   an observation does not reproduce its chunk. The script replays the identical
   sequence twice more and measures the per-chunk L2 spread. The notebook
   reports that band; it does not gate on it.
3. **Calibrate the anchor tolerance.** This model's ``joint_position`` rows are
   *absolute* targets that the server derives from the ``state_json`` of the
   same tick, so row 0 must sit near the state that was sent. That check is the
   one that catches broken wiring, and its tolerance is measured here rather
   than guessed.

## The frames are real

``exterior_view`` and ``wrist_view`` are real captures from a Franka Research 3
rig running the DROID camera layout, 180×320 — the embodiment this checkpoint
targets, so they are in distribution. The source capture is a Reactor-internal
artifact (internal — not reachable from outside Reactor); the frames themselves
are committed inside ``groot_n17_examples.npz``, so **this script re-runs from
the fixture alone** and needs nothing internal:

    python examples/record_groot_n17_examples.py          # re-records + recalibrates

Pass ``--from-capture PATH`` to rebuild the frame set from the original capture
instead (maintainers only).

## Where the streamed state comes from

The capture stores frames and chunks but no ``state_json``. Each observation's
state is therefore recovered from the **reference chunk's first row**: this
model's row 0 is its absolute-anchored prediction for the state it was handed,
so row 0 is that observation's proprioception to within the model's own
anchoring error (measured below at ≤0.1 rad). That is close enough to keep the
state in distribution with the frames, which is the point. The recovered
vectors are committed in the fixture, so nothing about it is derived at run
time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from reactor_robotics.groot_n17 import (
    ACTION_DIMS,
    ACTION_HORIZON,
    FRAME_HW,
    VIEWS,
    GrootN17Client,
    encode_state,
)

#: Fallback task, used only when the frame source carries none.
DEFAULT_TASK = "Put the blue block in the green bowl"

log = logging.getLogger("record_groot")


def load_frames(path: Path, from_capture: bool) -> dict:
    """Frames + recovered state, from the committed fixture or the capture."""
    z = np.load(path, allow_pickle=True)
    if from_capture:
        # The internal capture's own key names.
        ext, wri = z["ext"], z["wri"]
        chunks = z["chunks"]                       # (N, 40, 17)
        e, g, j = ACTION_DIMS["eef_9d"], ACTION_DIMS["gripper_position"], 7
        return {
            "exterior_view": ext,
            "wrist_view": wri,
            "eef_9d": chunks[:, 0, :e],
            "gripper_position": chunks[:, 0, e : e + g].reshape(-1),
            "joint_position": chunks[:, 0, e + g : e + g + j],
            "task": str(z["task"]) if "task" in z.files else DEFAULT_TASK,
            "reference_chunks": chunks,
            "reference_latency_ms": (
                z["latency_ms"] if "latency_ms" in z.files else None
            ),
        }
    return {
        "exterior_view": z["exterior_view"],
        "wrist_view": z["wrist_view"],
        "eef_9d": z["eef_9d"],
        "gripper_position": z["gripper_position"],
        "joint_position": z["joint_position"],
        "task": str(z["prompt"][0]),
        "reference_chunks": None,
        "reference_latency_ms": None,
    }


async def connect_with_capacity_retry(
    client: GrootN17Client, attempts: int, wait_s: float
) -> None:
    """Connect, retrying while the platform reports no free capacity."""
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


async def one_pass(client: GrootN17Client, src: dict, n: int, label: str) -> list[dict]:
    """Drive the observation sequence once, after a reset."""
    await client.reset()
    out = []
    for i in range(n):
        frames = {v: src[v][i] for v in VIEWS}
        state_json = encode_state(
            src["joint_position"][i], src["eef_9d"][i], float(src["gripper_position"][i])
        )
        pred = await client.predict(frames, state_json, src["task"])
        anchor = float(
            np.max(np.abs(pred.joint_position[0] - np.asarray(src["joint_position"][i])))
        )
        out.append(
            {
                "i": i,
                "joint_position": pred.joint_position,
                "eef_9d": pred.eef_9d,
                "gripper_position": pred.gripper_position,
                "step": pred.step,
                "latency_ms": pred.latency_ms,
                "discarded": len(pred.discarded),
                "anchor_max_abs_rad": anchor,
            }
        )
        print(
            f"[{label}] obs {i}: step={pred.step} joints={pred.joint_position.shape} "
            f"anchor|d|={anchor:.4f}rad latency={pred.latency_ms:.0f}ms "
            f"discarded={len(pred.discarded)}",
            flush=True,
        )
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/groot_n17_examples.npz")
    ap.add_argument(
        "--frames", default=None,
        help="npz to take frames + state from (default: --out, i.e. re-record "
             "from the committed fixture)",
    )
    ap.add_argument(
        "--from-capture", default=None, metavar="PATH",
        help="rebuild the frame set from the original internal capture "
             "(maintainers only)",
    )
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--connect-attempts", type=int, default=20)
    ap.add_argument("--connect-wait-s", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    src_path = Path(args.from_capture or args.frames or args.out)
    src = load_frames(src_path, from_capture=args.from_capture is not None)
    n = min(args.n, len(src["exterior_view"]))
    for v in VIEWS:
        h, w = src[v].shape[1:3]
        if (h, w) != FRAME_HW:
            print(f"note: {v} is {h}x{w}, not {FRAME_HW}; the server resizes to 256x256")
    print(f"source {src_path} — {n} observations, task {src['task']!r}", flush=True)

    client = GrootN17Client()
    passes: list[list[dict]] = []
    try:
        t0 = time.perf_counter()
        await connect_with_capacity_retry(
            client, args.connect_attempts, args.connect_wait_s
        )
        ready_s = time.perf_counter() - t0
        print(f"[connect] READY after {ready_s:.1f}s", flush=True)
        period_s = float("nan")
        for r in range(args.repeats + 1):
            passes.append(await one_pass(client, src, n, f"run{r}"))
            if r == 0:
                # Only meaningful once the model is predicting, which it is not
                # until a task and the first frames have landed.
                period_s = await client.observe_period_s()
                print(
                    f"[probe] free-running chunk period: {period_s * 1e3:.0f} ms",
                    flush=True,
                )
    finally:
        await client.close()

    recorded = passes[0]

    # ---- band from run-to-run spread, on the executed field ----------------
    dists: list[float] = []
    pairs = []
    for a in range(len(passes)):
        for b in range(a + 1, len(passes)):
            per_chunk = [
                float(np.linalg.norm(
                    passes[a][i]["joint_position"] - passes[b][i]["joint_position"]
                ))
                for i in range(n)
            ]
            pairs.append({"runs": [a, b], "per_chunk_l2_joint_position": per_chunk})
            dists.extend(per_chunk)
    mean = float(np.mean(dists))
    std = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
    band = mean + 3.0 * std

    # ---- anchor tolerance: the check that actually catches bad wiring ------
    anchors = [r["anchor_max_abs_rad"] for p in passes for r in p]
    anchor_mean = float(np.mean(anchors))
    anchor_std = float(np.std(anchors, ddof=1)) if len(anchors) > 1 else 0.0
    anchor_band = anchor_mean + 3.0 * anchor_std

    # ---- per-step joint step: is a chunk commandable at 15 Hz as-is? -------
    # Reactor's FR3 client clamps at 0.05 rad per tick (0.75 rad/s at 15 Hz), so
    # a chunk whose own steps stay under that needs no clamping at all.
    dqs = [
        float(np.max(np.abs(np.diff(r["joint_position"], axis=0))))
        for p in passes for r in p
    ]
    dq_mean = float(np.mean(dqs))
    dq_std = float(np.std(dqs, ddof=1)) if len(dqs) > 1 else 0.0
    dq_band = dq_mean + 3.0 * dq_std

    lat = [r["latency_ms"] for p in passes for r in p]
    norms = [float(np.linalg.norm(r["joint_position"])) for r in recorded]

    payload = {v: np.asarray(src[v][:n]) for v in VIEWS}
    payload.update(
        prompt=np.asarray([src["task"]] * n),
        joint_position=np.asarray(src["joint_position"][:n], dtype=np.float64),
        eef_9d=np.asarray(src["eef_9d"][:n], dtype=np.float64),
        gripper_position=np.asarray(src["gripper_position"][:n], dtype=np.float64),
        expected_joint_position=np.stack([r["joint_position"] for r in recorded]),
        expected_eef_9d=np.stack([r["eef_9d"] for r in recorded]),
        expected_gripper_position=np.stack([r["gripper_position"] for r in recorded]),
        step=np.asarray([r["step"] for r in recorded], dtype=np.int64),
        l2_band=np.asarray(band),
        l2_band_mean=np.asarray(mean),
        l2_band_std=np.asarray(std),
        anchor_band_rad=np.asarray(anchor_band),
        anchor_band_mean=np.asarray(anchor_mean),
        anchor_band_std=np.asarray(anchor_std),
        dq_band_rad=np.asarray(dq_band),
        dq_band_mean=np.asarray(dq_mean),
        dq_band_std=np.asarray(dq_std),
        chunk_period_ms=np.asarray(period_s * 1e3),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    manifest = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "reactor/groot-n17",
        "task": src["task"],
        "frames": "REAL — Franka Research 3 rig, DROID camera layout, 180x320; "
                  "source capture is a Reactor-internal artifact, frames "
                  "committed in this fixture",
        # Basename only: the original capture lives on a Reactor-internal path,
        # and a manifest committed to a repo shared outside the team should not
        # carry a machine path nobody outside can resolve.
        "frame_source": (
            f"{src_path.name} (Reactor-internal capture)"
            if args.from_capture
            else src_path.name
        ),
        "action_horizon": ACTION_HORIZON,
        "action_dims": ACTION_DIMS,
        "state_recovered_from": "reference chunk row 0 (absolute-anchored)",
        "n_examples": n,
        "passes": len(passes),
        "ready_seconds": round(ready_s, 1),
        "free_running_chunk_period_ms": round(period_s * 1e3, 1),
        "out_bytes": out.stat().st_size,
        "recorded_run": [
            {"i": r["i"], "step": r["step"], "latency_ms": round(r["latency_ms"], 1),
             "discarded": r["discarded"],
             "anchor_max_abs_rad": round(r["anchor_max_abs_rad"], 5)}
            for r in recorded
        ],
        "joint_position_l2_norms": [round(v, 4) for v in norms],
        "band": {
            "field": "joint_position",
            "pairs": pairs,
            "n_samples": len(dists),
            "mean_l2": mean,
            "std_l2": std,
            "band_mean_plus_3sigma": band,
            "max_observed_l2": max(dists),
        },
        "anchor": {
            "what": "max|joint_position[0] - streamed joint_position| in rad",
            "n_samples": len(anchors),
            "values": [round(v, 5) for v in anchors],
            "mean": anchor_mean,
            "std": anchor_std,
            "band_mean_plus_3sigma": anchor_band,
            "max_observed": max(anchors),
        },
        "per_step_joint_delta": {
            "what": "max|joint_position[k+1] - joint_position[k]| in rad, per chunk",
            "reference": "Reactor's FR3 client clamps at 0.05 rad/tick "
                         "(0.75 rad/s at 15 Hz) — internal",
            "n_samples": len(dqs),
            "values": [round(v, 5) for v in dqs],
            "mean": dq_mean,
            "std": dq_std,
            "band_mean_plus_3sigma": dq_band,
            "max_observed": max(dqs),
        },
        "latency_ms": {
            f"run{r}": [round(g["latency_ms"], 1) for g in p]
            for r, p in enumerate(passes)
        },
        "latency_summary_ms": {
            "gate": "end of frame-window hold -> chunk (one chunk discarded first)",
            "p50": round(float(np.median(lat)), 1),
            "min": round(min(lat), 1),
            "max": round(max(lat), 1),
            "n": len(lat),
        },
    }
    if src["reference_latency_ms"] is not None:
        manifest["capture_reference_latency_ms"] = [
            round(float(v), 1) for v in src["reference_latency_ms"]
        ]
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2))
    print("\n=== BAND CALIBRATION (joint_position L2) ===")
    print(json.dumps({k: v for k, v in manifest["band"].items() if k != "pairs"}, indent=2))
    print("\n=== ANCHOR CALIBRATION ===")
    print(json.dumps(manifest["anchor"], indent=2))
    print("\n=== PER-STEP JOINT DELTA ===")
    print(json.dumps(
        {k: v for k, v in manifest["per_step_joint_delta"].items() if k != "values"},
        indent=2,
    ))
    print("\n=== LATENCY ===")
    print(json.dumps(manifest["latency_summary_ms"], indent=2))
    print(f"\nchunk L2 norms {[round(v, 2) for v in norms]}; band {band:.4f} "
          f"({100 * band / float(np.mean(norms)):.0f}% of magnitude)")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
