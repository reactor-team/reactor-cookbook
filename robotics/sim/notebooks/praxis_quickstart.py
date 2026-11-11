#!/usr/bin/env python3
"""Praxis-1 quickstart: drive hosted Praxis-1 DROID or YAM from Python.

Run from this directory, with REACTOR_API_KEY set:

    uv run python praxis_quickstart.py --embodiment droid
    uv run python praxis_quickstart.py --embodiment yam

The script replays five recorded observations against the hosted model and
checks the chunks it returns: shapes, step order, finite values, rotation
matrices, and how far row 0 sits from the measured pose you sent. The
tolerances come from ``examples/praxis_<embodiment>_examples.npz``. Without
that fixture it falls back to the recorder's synthetic observations and
reports the banded checks instead of gating on them.

Praxis-1 is lock-step. The first chunk of an episode (``step`` 0, 21 rows)
arrives once the session has a task, 25 frames of every camera and 25 rows of
measured state. Every later chunk (24 rows) waits for ``set_executed_step``
with the previous ``step``. See ``praxis_quickstart.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 15
#: Frames per camera and state rows each prediction uses.
WINDOW_ROWS = 25
#: Wait after new frames so every camera's 25-frame window holds only them.
WINDOW_S = WINDOW_ROWS / FPS + 0.3
FIRST_ROWS, CHUNK_ROWS = 21, 24
CHUNK_MS = CHUNK_ROWS / FPS * 1000.0
N_EXAMPLES = 5
#: Rotation blocks come back orthonormal to float precision (about 1e-7).
#: A fixed tolerance, not a calibrated band: precision noise is not a signal.
ORTHO_TOL = 1e-3


@dataclass(frozen=True)
class Embodiment:
    model: str
    tracks: tuple[str, ...]
    frame_hw: tuple[int, int]
    #: Arm names as they appear in the state and reply; "" for a single arm.
    arms: tuple[str, ...]
    task: str
    #: Start pose per arm: [x, y, z, qw, qx, qy, qz].
    home: dict[str, tuple[float, ...]]
    #: Velocity of the synthetic reach per arm, m/s.
    velocity: dict[str, tuple[float, float, float]]
    #: DROID only; YAM poses are always in the rig world frame.
    pose_frame: str | None


SPECS = {
    "droid": Embodiment(
        model="reactor/praxis-1-droid",
        tracks=("exterior", "wrist"),
        frame_hw=(288, 512),
        arms=("",),
        task="Put the red block in the bowl.",
        home={"": (0.45, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0)},
        velocity={"": (0.02, 0.01, -0.02)},
        pose_frame="fr3_hand",
    ),
    "yam": Embodiment(
        model="reactor/praxis-1-yam",
        tracks=("top", "wrist_left", "wrist_right"),
        frame_hw=(260, 416),
        arms=("left", "right"),
        task="Pick up the pen from the center of the table, then place the pen in the bag",
        home={
            "left": (0.3, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0),
            "right": (0.3, -0.2, 0.25, 1.0, 0.0, 0.0, 0.0),
        },
        velocity={"left": (0.01, -0.015, -0.01), "right": (0.0, 0.0, 0.0)},
        pose_frame=None,
    ),
}


def pose_key(arm: str) -> str:
    return f"{arm}_poses" if arm else "poses"


def grip_key(arm: str) -> str:
    return f"{arm}_grips" if arm else "grips"


# ----------------------------------------------------------- observations


def _scene(h: int, w: int, t: float, view: str) -> np.ndarray:
    """One synthetic camera view at progress ``t`` in [0, 1]. Pure function."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.4)
    img[:horizon] = (190, 192, 198)
    img[horizon:] = (150, 120, 88)
    yy, xx = np.ogrid[:h, :w]

    def disc(cy: int, cx: int, r: int, colour: tuple[int, int, int]) -> None:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = colour

    if view in ("exterior", "top"):
        disc(int(h * 0.72), int(w * 0.75), int(h * 0.10), (235, 235, 240))
        disc(int(h * 0.70), int(w * (0.30 + 0.30 * t)), int(h * 0.06), (200, 40, 40))
    else:
        right = view.endswith("right")
        scale = 1.0 + 1.5 * t
        disc(int(h * 0.55), int(w * (0.6 if right else 0.4)), int(h * 0.08 * scale), (200, 40, 40))
        img[:, : int(w * 0.08)] = (48, 50, 56)
        img[:, int(w * 0.92) :] = (48, 50, 56)
    return img


def synthetic_observation(spec: Embodiment, i: int) -> tuple[dict[str, np.ndarray], dict]:
    """Observation ``i`` of a deterministic reach: frames and a 25-row state.

    Each arm moves at a constant velocity, so the 25 state rows are a smooth
    history ending at the current pose. Grippers stay open. Pure function of
    ``i``: no RNG and no clock.
    """
    h, w = spec.frame_hw
    t = i / (N_EXAMPLES - 1)
    frames = {view: _scene(h, w, t, view) for view in spec.tracks}
    now = i * CHUNK_ROWS / FPS
    arms = {}
    for arm in spec.arms:
        home = np.asarray(spec.home[arm])
        vel = np.asarray(spec.velocity[arm])
        rows = []
        for k in range(WINDOW_ROWS):
            xyz = home[:3] + vel * (now - (WINDOW_ROWS - 1 - k) / FPS)
            rows.append([*np.round(xyz, 6).tolist(), *home[3:].tolist()])
        arms[arm] = {"ee_pose": rows, "grip": [0.0] * WINDOW_ROWS}
    state = arms[""] if spec.arms == ("",) else arms
    return frames, state


def current_xyz(spec: Embodiment, state: dict, arm: str) -> np.ndarray:
    rows = state["ee_pose"] if arm == "" else state[arm]["ee_pose"]
    return np.asarray(rows[-1][:3], dtype=np.float64)


# --------------------------------------------------------------- protocol


async def connect(spec: Embodiment, attempts: int = 4, wait_s: float = 30.0):
    """Connect, retrying while the platform reports no free capacity."""
    from reactor_robotics.session import ReactorSession

    session = ReactorSession(spec.model, fps=FPS, frame_size=spec.frame_hw)
    for attempt in range(1, attempts + 1):
        try:
            await session.connect(
                spec.tracks,
                subscribe=(
                    "action_prediction",
                    "prediction_failed",
                    "session_status",
                    "context_reset",
                ),
            )
            return session
        except Exception as exc:
            if "429" not in str(exc) and "no available capacity" not in str(exc):
                await session.close()
                raise
            if attempt == attempts:
                await session.close()
                raise RuntimeError(
                    f"no capacity for {spec.model} after {attempts} attempts; "
                    "retry in a few minutes or ask your Reactor contact"
                ) from exc
            print(f"429 no capacity (attempt {attempt}/{attempts}); retrying in {wait_s:.0f}s")
            await asyncio.sleep(wait_s)
    raise AssertionError("unreachable")


async def await_chunk(session, timeout_s: float = 30.0) -> dict:
    """Wait for the next action_prediction; raise on prediction_failed."""
    deadline = time.monotonic() + timeout_s
    while True:
        for failure in session.drain("prediction_failed"):
            raise RuntimeError(f"prediction_failed: {failure}")
        try:
            return await session.next_message("action_prediction", timeout_s=0.25)
        except asyncio.TimeoutError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "no action_prediction; send get_status to see waiting_for"
                ) from None


async def request_chunk(
    session,
    frames: dict[str, np.ndarray],
    state: dict,
    *,
    echo_step: int | None = None,
    reset: bool = False,
) -> tuple[dict, float]:
    """Show one observation and get the chunk computed from it.

    Frames go first and fill the 25-frame window. Then the state, and then
    the trigger: the echo of the executed step, ``reset_context`` to start a
    new episode, or nothing for the first chunk of a session. Returns the
    reply and the milliseconds from the trigger to the reply.
    """
    session.set_frames(frames)
    await asyncio.sleep(WINDOW_S)
    session.drain("action_prediction")
    await session.send(
        "set_robot_state_json",
        {"robot_state_json": json.dumps(state, separators=(",", ":"), allow_nan=False)},
    )
    started = time.perf_counter()
    if reset:
        await session.send("reset_context", {})
    elif echo_step is not None:
        await session.send("set_executed_step", {"executed_step": echo_step})
    reply = await await_chunk(session)
    return reply, (time.perf_counter() - started) * 1000.0


async def run_episode(session, spec: Embodiment, observations, *, reset_first: bool) -> list[dict]:
    """Drive one episode over ``observations``; return one record per chunk."""
    records: list[dict] = []
    for i, (frames, state) in enumerate(observations):
        echo = None if i == 0 else records[-1]["step"]
        reply, latency_ms = await request_chunk(
            session, frames, state, echo_step=echo, reset=(i == 0 and reset_first)
        )
        record = {
            "step": int(reply["step"]),
            "latency_ms": latency_ms,
            "poses": {a: np.asarray(reply[pose_key(a)], dtype=np.float64) for a in spec.arms},
            "grips": {a: np.asarray(reply[grip_key(a)], dtype=np.float64) for a in spec.arms},
        }
        record["rows"] = len(record["poses"][spec.arms[0]])
        record.update(chunk_metrics(spec, record, state))
        records.append(record)
    return records


def chunk_metrics(spec: Embodiment, record: dict, state: dict) -> dict:
    """Distances that hold without knowing the task outcome.

    * ``anchor_m``: row 0's position against the measured position you sent.
      Targets are absolute, so a large gap means a frame or unit mistake.
    * ``dxyz_m``: the largest position change between adjacent rows.
    * ``ortho``: how far each 3 x 3 block is from a rotation matrix.
    """
    anchor = dxyz = ortho = 0.0
    for arm in spec.arms:
        poses = record["poses"][arm]
        anchor = max(anchor, float(np.max(np.abs(poses[0, :3] - current_xyz(spec, state, arm)))))
        dxyz = max(dxyz, float(np.max(np.abs(np.diff(poses[:, :3], axis=0)))))
        rot = poses[:, 3:].reshape(-1, 3, 3)
        gram = rot @ np.transpose(rot, (0, 2, 1))
        ortho = max(ortho, float(np.max(np.abs(gram - np.eye(3)))))
    return {"anchor_m": anchor, "dxyz_m": dxyz, "ortho": ortho}


# ------------------------------------------------------------------ main


def load_examples(name: str, spec: Embodiment):
    """The recorded fixture, or synthetic observations with no bands."""
    path = Path(__file__).resolve().parent / "examples" / f"praxis_{name}_examples.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            fixture = {key: data[key] for key in data.files}
        observations = [
            ({t: fixture[t][i] for t in spec.tracks}, json.loads(str(fixture["state_json"][i])))
            for i in range(len(fixture["rows"]))
        ]
        return observations, fixture, path
    return [synthetic_observation(spec, i) for i in range(N_EXAMPLES)], None, path


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embodiment", choices=sorted(SPECS), default="droid")
    args = parser.parse_args()
    spec = SPECS[args.embodiment]

    # Keep INFO on: status transitions are logged, not raised.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from reactor_robotics.session import describe_api_key

    print(describe_api_key())
    observations, fixture, path = load_examples(args.embodiment, spec)
    if fixture is not None:
        print(f"{len(observations)} recorded examples from {path.name}")
    else:
        print(f"No {path.name}; using the recorder's synthetic observations.")
        print("Banded checks are reported, not enforced. To record the fixture:")
        print(f"  uv run python examples/record_praxis_examples.py --embodiment {args.embodiment}")

    session = await connect(spec)
    try:
        print("status transitions :", " -> ".join(session.status_log))
        print("model              :", spec.model)
        print("tracks             :", ", ".join(spec.tracks), f"({spec.frame_hw[1]}x{spec.frame_hw[0]})")
        await session.send("set_task", {"task": spec.task})
        if spec.pose_frame:
            await session.send("set_pose_frame", {"pose_frame": spec.pose_frame})
        print(f"task               : {spec.task!r}")

        live = await run_episode(session, spec, observations, reset_first=False)
        for i, rec in enumerate(live):
            print(
                f"example {i}: step={rec['step']} rows={rec['rows']} "
                f"anchor={rec['anchor_m'] * 1000:5.1f} mm  max step {rec['dxyz_m'] * 1000:4.1f} mm  "
                f"{rec['latency_ms']:6.1f} ms"
            )

        steps = [r["step"] for r in live]
        values = [a for r in live for a in (*r["poses"].values(), *r["grips"].values())]
        checks = [
            (
                f"rows are {FIRST_ROWS} then {CHUNK_ROWS} ({[r['rows'] for r in live]})",
                [r["rows"] for r in live] == [FIRST_ROWS] + [CHUNK_ROWS] * (len(live) - 1),
            ),
            (
                "every pose row has 12 values",
                all(p.shape[1:] == (12,) for r in live for p in r["poses"].values()),
            ),
            ("all values finite", all(np.isfinite(v).all() for v in values)),
            (f"steps are 0, 1, 2, ... ({steps})", steps == list(range(len(live)))),
            (
                f"rotation blocks orthonormal within {ORTHO_TOL:g} "
                f"(worst {max(r['ortho'] for r in live):.1e})",
                all(r["ortho"] <= ORTHO_TOL for r in live),
            ),
        ]
        reported = []
        bands = (
            ("row 0 within {:.1f} mm of the sent pose", "anchor_m", "anchor_band_m", 1000.0),
            ("adjacent rows within {:.1f} mm", "dxyz_m", "dxyz_band_m", 1000.0),
        )
        for label, metric, band_key, scale in bands:
            worst = max(r[metric] for r in live)
            if fixture is not None:
                band = float(fixture[band_key])
                checks.append(
                    (label.format(band * scale) + f" (worst {worst * scale:.4g})", worst <= band)
                )
            else:
                reported.append(f"{metric}: worst {worst * scale:.4g} mm (no band without the fixture)")

        print(f"\n{'check':<66} result")
        print("-" * 74)
        ok = True
        for label, passed in checks:
            ok &= bool(passed)
            print(f"{label:<66} {'PASS' if passed else 'FAIL'}")
        print("-" * 74)
        print("RESULT:", "PASS" if ok else "FAIL")
        for line in reported:
            print("  reported only:", line)

        echo_lat = [r["latency_ms"] for r in live[1:]]
        print(
            f"\nEcho to next chunk: p50 {np.median(echo_lat):.0f} ms, max {max(echo_lat):.0f} ms, "
            f"against {CHUNK_MS:.0f} ms of motion per chunk. Network and inference add"
        )
        print("this wait between chunks; your controller holds the robot meanwhile.")

        if fixture is not None:
            l2 = [
                float(np.linalg.norm(np.concatenate([
                    r["poses"][arm] - fixture["expected_" + pose_key(arm)][i, : r["rows"]]
                    for arm in spec.arms
                ])))
                for i, r in enumerate(live)
            ]
            print(
                f"Run-to-run L2 vs the recorded poses (band {float(fixture['l2_band']):.4f}): "
                + ", ".join(f"{d:.4f}" for d in l2)
            )
            print("The model is not seeded, so this is a drift signal, reported only.")

        # The session waits for an echo before it computes anything else.
        await session.send("get_status", {})
        status = await session.next_message("session_status", timeout_s=10)
        print(f"\nget_status without an echo: waiting_for={status.get('waiting_for')!r}")

        # A reset starts a new episode: no echo needed, 21 rows again, and
        # step keeps counting.
        frames, state = observations[0]
        reply, ms = await request_chunk(session, frames, state, reset=True)
        try:
            reset_msg = await session.next_message("context_reset", timeout_s=5)
        except asyncio.TimeoutError:
            reset_msg = {}
        rows = len(reply[pose_key(spec.arms[0])])
        print(
            f"after reset_context: step={int(reply['step'])} rows={rows} in {ms:.0f} ms "
            f"(context_reset next_step={reset_msg.get('next_step', '?')})"
        )
        ok &= rows == FIRST_ROWS
        return 0 if ok else 1
    finally:
        # Always close: a live session holds a GPU worker.
        await session.close()
        print("closed. status:", " -> ".join(session.status_log))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
