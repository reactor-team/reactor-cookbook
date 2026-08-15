# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record ``cosmos_droid_examples.npz`` against the live Cosmos-Nano-Policy-DROID
deployment.

    REACTOR_API_KEY=... python examples/record_cosmos_droid_examples.py

Three jobs:

1. **Record 5 observation → chunk pairs**, exercising the first-chunk path (no
   echo) and the executed-step flow gate for every chunk after it.
2. **Calibrate the run-to-run band**, by replaying the identical sequence twice
   more and measuring the per-chunk L2 spread.
3. **Calibrate the anchor tolerance.** The chunk rows are absolute joint
   targets, so row 0 must sit near the proprio that was sent. That check is
   what catches broken wiring, and its tolerance is measured, not guessed.

## The frames are synthetic

This checkpoint takes **three** views (wrist + two exterior). The real DROID
captures available locally come from a two-camera rig, and duplicating one
exterior view into the second slot would misrepresent a second camera. So
``synthetic_observation(i)`` builds a deterministic five-step reaching sequence
with three genuinely distinct views at 180×320. Pure function of ``i``: no RNG,
no clock, so re-running regenerates identical inputs.

**The recorded actions are therefore not a behavioural demonstration.** They
are a protocol and invariant fixture. Task competence comes from the real
simulator — and [`../cosmos-droid/`](../cosmos-droid) in this repo runs
exactly that: NVIDIA's RoboLab DROID benchmark against this same model, where
the same contract reached **40.0% over 120 tasks × 3 rollouts against NVIDIA's
published 39.7%**, and **8/10 solves over the production wire** (internal port
report — internal).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from reactor_robotics.cosmos_droid import (
    ACTION_SHAPE,
    CHUNK_BUDGET_MS,
    DROID_RESET_JOINTS,
    EXPECTED_COMPUTE_MS,
    EXPECTED_WIRE_P50_MS,
    TRACKS,
    CosmosDroidClient,
)

TASK = "put the banana in the bowl"

FRAME_HW = (180, 320)

log = logging.getLogger("record_cosmos")


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


def synthetic_observation(i: int) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Observation ``i`` of a deterministic five-step reach.

    Returns ``(frames, joints, gripper)``. Pure function of ``i``. The three
    views are genuinely different scenes, not copies: a left-side exterior, a
    right-side exterior at a different angle, and a wrist close-up.
    """
    h, w = FRAME_HW
    t = i / 4.0

    def tabletop() -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.40)
        img[:horizon] = (190, 192, 198)
        img[horizon:] = (162, 128, 92)
        grade = np.linspace(1.0, 0.80, h - horizon, dtype=np.float32)[:, None, None]
        img[horizon:] = (img[horizon:] * grade).astype(np.uint8)
        return img

    # exterior_view_1: camera on the left, object travelling right.
    e1 = tabletop()
    obj_x = int(w * (0.26 + 0.40 * t))
    _disc(e1, int(h * 0.70), obj_x, int(h * 0.06), int(w * 0.045), (226, 196, 62))
    _disc(e1, int(h * 0.74), int(w * 0.76), int(h * 0.09), int(w * 0.07), (232, 232, 238))
    grip_y = int(h * (0.20 + 0.30 * t))
    _rect(e1, grip_y - 10, grip_y + 22, obj_x - 15, obj_x - 7, (48, 50, 56))
    _rect(e1, grip_y - 10, grip_y + 22, obj_x + 7, obj_x + 15, (48, 50, 56))

    # exterior_view_2: the other side of the table — mirrored geometry, warmer
    # light, so the two exterior views are not the same picture.
    e2 = tabletop()
    e2[int(h * 0.40) :] = (e2[int(h * 0.40) :] * 0.88).astype(np.uint8)
    obj_x2 = w - obj_x
    _disc(e2, int(h * 0.66), obj_x2, int(h * 0.055), int(w * 0.04), (226, 196, 62))
    _disc(e2, int(h * 0.70), int(w * 0.24), int(h * 0.085), int(w * 0.065), (232, 232, 238))
    _rect(e2, grip_y - 8, grip_y + 20, obj_x2 - 13, obj_x2 - 6, (48, 50, 56))
    _rect(e2, grip_y - 8, grip_y + 20, obj_x2 + 6, obj_x2 + 13, (48, 50, 56))

    # wrist_view: looking down, closing in.
    wr = np.zeros((h, w, 3), dtype=np.uint8)
    wr[:] = (162, 128, 92)
    scale = 1.0 + 1.6 * t
    _disc(wr, int(h * 0.56), w // 2, int(20 * scale), int(30 * scale), (226, 196, 62))
    _rect(wr, 0, h, 0, int(w * 0.10), (48, 50, 56))
    _rect(wr, 0, h, int(w * 0.90), w, (48, 50, 56))

    frames = {"wrist_view": wr, "exterior_view_1": e1, "exterior_view_2": e2}
    assert set(frames) == set(TRACKS), frames.keys()

    joints = DROID_RESET_JOINTS + np.array(
        [0.04, 0.05, -0.03, 0.06, 0.02, -0.05, 0.03]
    ) * i
    gripper = round(min(1.0, 0.05 + 0.2 * i), 3)
    return frames, joints, float(gripper)


async def connect_with_capacity_retry(
    client: CosmosDroidClient, attempts: int, wait_s: float
) -> None:
    """Connect, retrying while the platform reports no free capacity.

    This model is single-session on one B200, so a busy or not-yet-scaled
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
                f"[connect] attempt {attempt}/{attempts}: no B200 capacity free; "
                f"retrying in {wait_s:.0f}s",
                flush=True,
            )
            await asyncio.sleep(wait_s)


async def one_pass(client: CosmosDroidClient, n: int, label: str) -> list[dict]:
    """Drive the observation sequence once.

    No reset: the model is stateless per prediction, so there is no episode to
    end and nothing to clear between passes.
    """
    out = []
    for i in range(n):
        frames, joints, gripper = synthetic_observation(i)
        pred = await client.predict(frames, joints, gripper, TASK)
        anchor = float(np.max(np.abs(pred.joint_position[0] - joints)))
        out.append(
            {
                "i": i,
                "actions": pred.actions,
                "step": pred.step,
                "latency_ms": pred.latency_ms,
                "discarded": len(pred.discarded),
                "anchor_max_abs_rad": anchor,
            }
        )
        print(
            f"[{label}] obs {i}: step={pred.step} shape={pred.actions.shape} "
            f"anchor|d|={anchor:.4f}rad latency={pred.latency_ms:.0f}ms "
            f"discarded={len(pred.discarded)}",
            flush=True,
        )
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/cosmos_droid_examples.npz")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--connect-attempts", type=int, default=20)
    ap.add_argument("--connect-wait-s", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    client = CosmosDroidClient()
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

    anchors = [r["anchor_max_abs_rad"] for p in passes for r in p]
    anchor_mean = float(np.mean(anchors))
    anchor_std = float(np.std(anchors, ddof=1)) if len(anchors) > 1 else 0.0
    anchor_band = anchor_mean + 3.0 * anchor_std

    dqs = [
        float(np.max(np.abs(np.diff(r["actions"][:, :7], axis=0))))
        for p in passes for r in p
    ]
    dq_mean = float(np.mean(dqs))
    dq_std = float(np.std(dqs, ddof=1)) if len(dqs) > 1 else 0.0
    dq_band = dq_mean + 3.0 * dq_std

    lat = [r["latency_ms"] for p in passes for r in p]
    norms = [float(np.linalg.norm(r["actions"])) for r in recorded]

    frames_by_track: dict[str, list[np.ndarray]] = {t: [] for t in TRACKS}
    joints_all, grippers = [], []
    for i in range(args.n):
        f, j, g = synthetic_observation(i)
        for t in TRACKS:
            frames_by_track[t].append(f[t])
        joints_all.append(j)
        grippers.append(g)

    payload = {t: np.stack(frames_by_track[t]) for t in TRACKS}
    payload.update(
        prompt=np.asarray([TASK] * args.n),
        joints=np.stack(joints_all),
        gripper=np.asarray(grippers, dtype=np.float64),
        expected_actions=np.stack([r["actions"] for r in recorded]),
        step=np.asarray([r["step"] for r in recorded], dtype=np.int64),
        l2_band=np.asarray(band),
        l2_band_mean=np.asarray(mean),
        l2_band_std=np.asarray(std),
        run_to_run_exact=np.asarray(exact),
        anchor_band_rad=np.asarray(anchor_band),
        anchor_band_mean=np.asarray(anchor_mean),
        anchor_band_std=np.asarray(anchor_std),
        dq_band_rad=np.asarray(dq_band),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    manifest = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "reactor/cosmos-nano-policy-droid",
        "task": TASK,
        "frames": f"SYNTHETIC (deterministic, synthetic_observation()), "
                  f"{FRAME_HW[0]}x{FRAME_HW[1]}, three distinct views",
        "action_shape": list(ACTION_SHAPE),
        "n_examples": args.n,
        "passes": len(passes),
        "ready_seconds": round(ready_s, 1),
        "out_bytes": out.stat().st_size,
        "recorded_run": [
            {"i": r["i"], "step": r["step"], "latency_ms": round(r["latency_ms"], 1),
             "discarded": r["discarded"],
             "anchor_max_abs_rad": round(r["anchor_max_abs_rad"], 5)}
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
        "anchor": {
            "what": "max|action[0][:7] - streamed joint_position| in rad",
            "n_samples": len(anchors),
            "values": [round(v, 5) for v in anchors],
            "mean": anchor_mean,
            "std": anchor_std,
            "band_mean_plus_3sigma": anchor_band,
            "max_observed": max(anchors),
        },
        "per_step_joint_delta": {
            "what": "max|action[k+1][:7] - action[k][:7]| in rad, per chunk",
            "n_samples": len(dqs),
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
            "gate": "executed-step echo -> chunk (first chunk: proprio -> chunk)",
            "p50": round(float(np.median(lat)), 1),
            "min": round(min(lat), 1),
            "max": round(max(lat), 1),
            "n": len(lat),
            "chunk_budget_ms": round(CHUNK_BUDGET_MS, 1),
            "reference_compute_ms": EXPECTED_COMPUTE_MS,
            "reference_wire_p50_ms": EXPECTED_WIRE_P50_MS,
        },
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2))
    print("\n=== BAND CALIBRATION ===")
    print(json.dumps(
        {k: v for k, v in manifest["band"].items() if k != "pairs"}, indent=2))
    print("\n=== ANCHOR CALIBRATION ===")
    print(json.dumps(manifest["anchor"], indent=2))
    print("\n=== LATENCY ===")
    print(json.dumps(manifest["latency_summary_ms"], indent=2))
    print(f"\nrun-to-run exact: {exact}   band {band:.5f} "
          f"({100 * band / float(np.mean(norms)):.0f}% of magnitude)")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
