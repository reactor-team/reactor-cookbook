# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Record `dreamzero_examples.npz` against the live DreamZero deployment.

Unlike the X-WAM fixture — subset from a recorded upstream evaluation tee —
DreamZero has no recorded corpus anywhere, so its examples have to be
*recorded*
by driving the hosted model. This script is that recording, committed so the
artifact is reproducible rather than magic:

    REACTOR_API_KEY=... python examples/record_dreamzero_examples.py

It does two things:

1. **Records 5 observation -> chunk pairs** inside one episode, so the
   fixture exercises `obs_seq` advancing and `chunk_index` incrementing
   across a real closed-loop sequence.
2. **Calibrates the loose L2 band.** DreamZero's pipeline is unseeded, so
   replaying the same observations does not reproduce the same actions and
   there is no parity claim to make. Instead the script replays the identical
   observation sequence twice more in fresh episodes and measures the
   per-chunk L2 distance between runs. The notebook's band is
   ``mean + 3 sigma`` of those distances — a number measured from the model's
   own run-to-run spread, not invented.

## About the frames

They are **synthetic**, and deliberately so. No DROID-appropriate real frames
were available: the local DROID reference checkout carries only
hardware-setup photographs (portrait framing, no wrist camera, no
manipulation scene), and the other local episode captures are RoboCasa
kitchen renders of a different embodiment (Panda-Omron, 256x256) — either
would be a *less* honest fixture than clearly-labelled synthetic frames,
because both would look like real observations while being far out of the
checkpoint's distribution.

Consequence, stated plainly and repeated in PROVENANCE.md and the notebook:
**the recorded actions are not a behavioural demonstration.** They are a
protocol and invariant fixture. Every check the notebook runs against them
(shape, finiteness, monotonic `obs_seq`, joint limits, gripper range,
chunk-boundary continuity, the L2 band) is a statement about the wire
contract and the model's self-consistency, never about task competence.

One fidelity detail is real, though: ``exterior_2`` is all black. That is
what the evaluation harness actually sends. RoboLab's
default ``--cam2-source black`` leaves the second exterior slot black to match
the checkpoint's training-time camera dropout, so a black ``exterior_2`` is
the in-distribution input for that view (the harness's observation adapter,
which implements this, is internal to Reactor's model repository).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from reactor_robotics.dreamzero import FRAME_HW, TRACKS, DreamZeroClient

TASK = "pick up the marker and put it in the cup"

#: A plausible DROID/Franka ready pose (radians), well inside the joint
#: limits. The recorded sequence walks away from it slightly per observation
#: so the streamed state is not constant.
HOME_JOINTS = np.array([0.0, -0.35, 0.0, -2.10, 0.0, 1.80, 0.79])

log = logging.getLogger("record_dz")


def synthetic_observation(i: int) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Build observation ``i`` of a deterministic 5-step reaching sequence.

    Returns ``(frames, joints, gripper)``. Pure function of ``i`` — no RNG, no
    clock — so re-running this script regenerates identical inputs.
    """
    h, w = FRAME_HW
    t = i / 4.0  # 0 -> 1 across the five observations

    # ---- exterior_1: the real primary view. A tabletop, a target object
    # travelling left-to-right, and a dark gripper descending toward it.
    ext = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.42)
    ext[:horizon] = (188, 190, 196)  # back wall
    ext[horizon:] = (150, 111, 74)  # table
    # A soft vertical gradient over the table so H.264 has real low-frequency
    # content to encode rather than a flat fill.
    grade = np.linspace(1.0, 0.78, h - horizon, dtype=np.float32)[:, None, None]
    ext[horizon:] = (ext[horizon:] * grade).astype(np.uint8)

    obj_x = int(w * (0.24 + 0.42 * t))
    obj_y = int(h * 0.70)
    _rect(ext, obj_y - 16, obj_y + 16, obj_x - 11, obj_x + 11, (205, 62, 48))  # marker
    cup_x = int(w * 0.78)
    _rect(ext, obj_y - 22, obj_y + 18, cup_x - 18, cup_x + 18, (232, 232, 238))  # cup
    grip_y = int(h * (0.20 + 0.30 * t))
    _rect(ext, grip_y - 10, grip_y + 22, obj_x - 15, obj_x - 7, (44, 46, 52))
    _rect(ext, grip_y - 10, grip_y + 22, obj_x + 7, obj_x + 15, (44, 46, 52))

    # ---- exterior_2: black, matching RoboLab's --cam2-source black default
    # (the checkpoint's training-time camera dropout). See module docstring.
    ext2 = np.zeros((h, w, 3), dtype=np.uint8)

    # ---- wrist: the same scene from above, closing in as the gripper drops.
    wrist = np.zeros((h, w, 3), dtype=np.uint8)
    wrist[:] = (150, 111, 74)
    scale = 1.0 + 1.6 * t
    half_w, half_h = int(38 * scale), int(26 * scale)
    cx, cy = w // 2, int(h * 0.56)
    _rect(wrist, cy - half_h, cy + half_h, cx - half_w, cx + half_w, (205, 62, 48))
    _rect(wrist, 0, h, 0, int(w * 0.10), (44, 46, 52))  # gripper finger, left
    _rect(wrist, 0, h, int(w * 0.90), w, (44, 46, 52))  # gripper finger, right

    frames = {"exterior_1": ext, "exterior_2": ext2, "wrist": wrist}
    assert set(frames) == set(TRACKS), frames.keys()

    # State: drift away from the ready pose, and close the gripper over time.
    joints = HOME_JOINTS + np.array([0.05, 0.04, -0.03, 0.06, 0.02, -0.05, 0.03]) * i
    gripper = round(min(1.0, 0.05 + 0.2 * i), 3)
    return frames, joints, float(gripper)


def _rect(img, y0, y1, x0, x1, colour) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = colour


async def connect_with_capacity_retry(
    client: DreamZeroClient, attempts: int, wait_s: float
) -> None:
    """Connect, retrying while the platform reports no free capacity.

    DreamZero holds **two** B200s for the life of a session, so a busy cluster
    answers session creation with ``429 no available capacity`` rather than
    queuing. That is a wait-and-retry condition, not a client bug — and it is
    worth retrying here because a recording run is unattended.
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
                f"[connect] attempt {attempt}/{attempts}: no 2x B200 capacity "
                f"free; retrying in {wait_s:.0f}s",
                flush=True,
            )
            await asyncio.sleep(wait_s)


async def one_pass(client: DreamZeroClient, n: int, label: str) -> list[dict]:
    """Drive the 5-observation sequence once in a fresh episode."""
    await client.reset()
    out = []
    for i in range(n):
        frames, joints, gripper = synthetic_observation(i)
        t0 = time.perf_counter()
        pred = await client.predict(frames, joints, gripper, task=TASK)
        out.append(
            {
                "i": i,
                "actions": pred.actions,
                "chunk_index": pred.chunk_index,
                "inference_seconds": pred.inference_seconds,
                "obs_seq": pred.obs_seq,
                "latency_ms": pred.latency_ms,
                "discarded": list(pred.discarded),
                "wall_s": time.perf_counter() - t0,
            }
        )
        print(
            f"[{label}] obs {i}: chunk_index={pred.chunk_index} "
            f"obs_seq={pred.obs_seq} shape={pred.actions.shape} "
            f"inference={pred.inference_seconds:.3f}s "
            f"latency={pred.latency_ms:.0f}ms "
            f"discarded={len(pred.discarded)}",
            flush=True,
        )
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/dreamzero_examples.npz")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="extra passes over the same observations, for the L2 band",
    )
    ap.add_argument("--connect-attempts", type=int, default=20)
    ap.add_argument("--connect-wait-s", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = DreamZeroClient()
    passes: list[list[dict]] = []
    try:
        t_connect = time.perf_counter()
        await connect_with_capacity_retry(
            client, args.connect_attempts, args.connect_wait_s
        )
        print(f"[connect] READY after {time.perf_counter() - t_connect:.1f}s", flush=True)
        for r in range(args.repeats + 1):
            passes.append(await one_pass(client, args.n, f"run{r}"))
    finally:
        await client.close()

    recorded = passes[0]

    # ---- calibrate the loose band from run-to-run spread ------------------
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
    mean, std = float(np.mean(dists)), float(np.std(dists, ddof=1))
    band = mean + 3.0 * std

    # Scale reference: how big is a chunk, so the band can be read as a
    # fraction rather than a bare number.
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
        joints=np.stack(joints_all),
        gripper=np.asarray(grippers, dtype=np.float64),
        prompt=np.asarray([TASK] * args.n),
        expected_actions=np.stack([r["actions"] for r in recorded]),
        chunk_index=np.asarray([r["chunk_index"] for r in recorded], dtype=np.int64),
        obs_seq=np.asarray([r["obs_seq"] for r in recorded], dtype=np.int64),
        inference_seconds=np.asarray(
            [r["inference_seconds"] for r in recorded], dtype=np.float64
        ),
        l2_band=np.asarray(band),
        l2_band_mean=np.asarray(mean),
        l2_band_std=np.asarray(std),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    manifest = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": TASK,
        "frames": "SYNTHETIC (deterministic, from synthetic_observation()); "
        "exterior_2 intentionally black per RoboLab --cam2-source black",
        "n_examples": args.n,
        "passes": len(passes),
        "out_path": str(out.resolve()),
        "out_bytes": out.stat().st_size,
        "recorded_run": [
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in r.items()
                if k != "actions"
            }
            for r in recorded
        ],
        "action_l2_norms": [round(n, 4) for n in norms],
        "band": {
            "pairs": pairs,
            "n_samples": len(dists),
            "mean_l2": mean,
            "std_l2": std,
            "band_mean_plus_3sigma": band,
            "max_observed_l2": max(dists),
        },
        "inference_seconds": {
            f"run{r}": [round(g["inference_seconds"], 4) for g in p]
            for r, p in enumerate(passes)
        },
        "latency_ms": {
            f"run{r}": [round(g["latency_ms"], 1) for g in p]
            for r, p in enumerate(passes)
        },
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2))
    print("\n=== BAND CALIBRATION ===")
    print(json.dumps(manifest["band"], indent=2))
    print(f"\nband (mean + 3s) = {band:.4f}; chunk L2 norms {norms}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
