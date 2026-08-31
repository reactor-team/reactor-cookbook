"""Robot bridge for ``reactor/dreamzero-yam-molmoact2`` on a bimanual i2rt YAM rig.

Publishes three camera views and both arms' measured state, and executes the
model's action chunks on two i2rt arms over CAN. The robot side talks only to
the i2rt ``Robot`` protocol (``get_joint_pos`` / ``command_joint_pos``), so the
hardware arms and the ``--mock`` stand-in are interchangeable.

Wire contract
-------------
Cameras: three sendonly video tracks named exactly ``top``, ``left``, ``right``,
RGB, 176x320 (the training resolution).

State, streamed continuously at ~10 Hz -- this is what paces the model:
``set_left_joint_pos`` {left_joint_pos: [6 rad]}, ``set_left_gripper_pos``
{left_gripper_pos: 0=open..1=closed}, and the two ``right`` equivalents.
``set_prompt`` is sent once, before the state loop.

On model >= 1.1.0 each ``action_chunk`` also reports ``obs_capture_time_us``
and ``view_skew_us`` (spread of the three cameras' newest capture stamps for
that chunk); the bridge logs the skew. ``--pair-by-capture-time`` sends
``set_pair_by_capture_time`` after the prompt, turning on server-side
cross-camera alignment for this session. Note: reactor-sdk 0.8.0 does not
stamp frames at capture -- the transport backfills the stamp at send time --
so skew reflects send-path timing until a stamping SDK ships; the alignment
toggle is still honored either way.

Actions: ``action_chunk`` messages carry ``data.actions``, 24x14 rows of
[left_joint(6) rad, left_gripper, right_joint(6) rad, right_gripper]. With both
arms' state streamed the joint values are absolute targets. Each chunk replaces
the pending buffer.

RUN ON RIG
==========
Hardware assumed: two YAM arms (one per side of the workspace), each on its own
CANable, plus three USB cameras.

0. Dependencies: ``reactor-sdk``, ``aiortc``, ``av``, ``numpy``,
   ``opencv-python``, and i2rt itself (``pip install -e /path/to/i2rt``, or put
   the checkout on ``PYTHONPATH``). i2rt is imported lazily, so ``--mock`` runs
   without it.

1. CAN bring-up. Give each adapter a persistent name (i2rt
   ``docs/guides/set-persistent-can-ids.md``); the i2rt bimanual example names
   the follower arms ``can_follower_l`` / ``can_follower_r``, which are this
   script's defaults. After every reboot::

       sudo ip link set up can_follower_l type can bitrate 1000000
       sudo ip link set up can_follower_r type can bitrate 1000000
       ip link show          # both must read "state UP"

2. Motor zeroing. i2rt checks measured qpos against the arm's joint limits at
   construction and refuses to start if it is outside them
   (``motor_chain_robot.py`` ``_check_current_qpos_in_joint_limits``). If it
   raises, move the arm to its zero pose and power-cycle it, or re-zero each
   motor: ``python i2rt/motor_config_tool/set_zero.py --channel <can> --motor_id N``
   for N in 1..6.

3. Gripper calibration. ``linear_4310`` ships with ``gripper_limits: null`` and
   ``needs_calibration: true`` (``i2rt/robots/config/linear_4310.yml``), so
   ``get_yam_robot`` auto-calibrates at startup: it drives the gripper to both
   mechanical stops under ~0.2 Nm to learn [closed, open]. Keep hands and the
   workpiece clear while it runs. To skip it entirely, pass the limits you
   already measured::

       --left-gripper-limits 0.0,6.57 --right-gripper-limits 0.0,6.57

4. Cameras. Map each model view to a capture device; the numbers are OpenCV
   device indices or /dev/video paths::

       --cam top=0 --cam left=2 --cam right=4

   Views must match what the model was trained on -- ``top`` looking down at the
   workspace, ``left``/``right`` over each shoulder. The i2rt models carry a
   wrist RealSense D405 (``yam_linear_4310_d405.xml``); to use those instead,
   implement ``read_rgb()`` over ``pyrealsense2`` and pass the instance in place
   of ``OpenCVCamera``. RealSense is deliberately not a dependency here.

5. Run::

       python dreamzero_yam_bridge_i2rt.py \
           --left-channel can_follower_l --right-channel can_follower_r \
           --cam top=0 --cam left=2 --cam right=4 \
           --prompt "fold the towel neatly with both arms"

Safety
------
The guards here are a minimum, not a substitute for the rig's e-stop. Keep a
hand on it.

* The bridge starts with ``zero_gravity_mode=False``, so each arm holds a PD
  target at its power-on pose from the moment it connects.
* It will not command anything until a chunk has arrived AND that chunk's first
  row is within ``--arm-tolerance`` rad of the measured pose. A model that wants
  a pose the rig is nowhere near is a lurch, not a start.
* Every commanded step is clamped to ``--max-joint-step`` rad (and
  ``--max-gripper-step`` in normalized gripper units) per control tick.
* i2rt's ``close()`` sets all torques to zero -- an arm that is holding a load
  or is extended will drop. Park the arms before Ctrl-C where you can.
* The DM motors' factory 400 ms command timeout is a real backstop: it drops the
  motors into damping if this loop stalls. Do not disable it for this bridge.

Tuning
------
Against prod the model returns roughly 3 chunks/s, each 24 rows, so at the
default 30 Hz the loop walks about 10 rows before the next chunk replaces the
buffer. In a mock run about 28% of ticks saturated the 0.15 rad clamp -- the
model asks for more than 4.5 rad/s at times. That is the limiter doing its job,
but it also means the arm trails the model's intent. Start conservative on a
real rig and raise ``--max-joint-step`` only once you have watched it move.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from typing import Any, Optional, Protocol

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from reactor_sdk import Reactor, ReactorStatus

MODEL = "reactor/dreamzero-yam-molmoact2"
CAMERAS = ("top", "left", "right")  # model expects exactly these three views
FRAME_H, FRAME_W = 176, 320  # training resolution
ARM_DOF = 6  # joints per arm; the gripper is the 7th element of an i2rt vector
CHUNK_SHAPE = (24, 14)


# ---------------------------------------------------------------------------
# Gripper convention
# ---------------------------------------------------------------------------
# The model speaks 0=open..1=closed. i2rt's command space is the other way
# round: `motor_chain_robot.py:83` declares `gripper_limits` as `[closed, open]`
# and `JointMapper.to_robot_joint_pos_space` (`robots/utils.py:596`) maps a
# command of 0 to gripper_limits[0]=closed and 1 to gripper_limits[1]=open --
# the same convention SimRobot documents at `sim_robot.py:65-67`, and the one
# `motor_chain_robot.py:141` assumes ("initialize as fully open" = 1). So the
# two spaces are exact complements and one flip converts either direction.


def flip_gripper(value: float) -> float:
    """Convert between the model's 0=open..1=closed and i2rt's 0=closed..1=open."""
    return float(np.clip(1.0 - float(value), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
class CameraSource(Protocol):
    def read_rgb(self) -> np.ndarray:
        """Latest frame as (176, 320, 3) uint8 RGB."""

    def close(self) -> None: ...


class OpenCVCamera:
    """A USB camera behind ``cv2.VideoCapture``, resized to the model's input.

    Grabbing runs on its own thread so a slow USB read never stalls the WebRTC
    sender, and so the frame we publish is the newest one rather than the oldest
    one still sitting in the driver's queue.
    """

    def __init__(self, name: str, spec: str):
        import cv2

        self._cv2 = cv2
        self.name = name
        source: Any = int(spec) if spec.isdigit() else spec
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {name!r}: cannot open capture source {spec!r}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab_loop, name=f"cam_{name}", daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            resized = self._cv2.resize(bgr, (FRAME_W, FRAME_H), interpolation=self._cv2.INTER_AREA)
            rgb = self._cv2.cvtColor(resized, self._cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb

    def read_rgb(self) -> np.ndarray:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


class SyntheticCamera:
    """``--mock`` frame source: a moving bar, so live frames are eyeballable."""

    def __init__(self, name: str):
        self.name = name
        self._t0 = time.monotonic()
        self._tint = {"top": 0, "left": 1, "right": 2}.get(name, 0)

    def read_rgb(self) -> np.ndarray:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame[:, :, self._tint] = 40
        x = int(((time.monotonic() - self._t0) * 60) % FRAME_W)
        frame[:, max(0, x - 8) : x + 8, :] = 200
        return frame

    def close(self) -> None:
        pass


class CameraTrack(VideoStreamTrack):
    """Publishes one camera as a WebRTC video track."""

    def __init__(self, source: CameraSource):
        super().__init__()
        self.source = source

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.source.read_rgb(), format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        return frame


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
class MockArm:
    """In-memory stand-in for an i2rt ``Robot`` (``--mock``).

    Implements the slice of the protocol this bridge uses, so the control path
    under test is the same code that drives the CAN bus.
    """

    def __init__(self, name: str, n_dofs: int = ARM_DOF + 1):
        self.name = name
        self._n_dofs = n_dofs
        # Gripper starts at 1.0 = open in i2rt space, matching
        # motor_chain_robot.py:141.
        self._qpos = np.zeros(n_dofs)
        self._qpos[ARM_DOF] = 1.0
        self._lock = threading.Lock()

    def num_dofs(self) -> int:
        return self._n_dofs

    def get_joint_pos(self) -> np.ndarray:
        with self._lock:
            return self._qpos.copy()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        with self._lock:
            self._qpos = np.asarray(joint_pos, dtype=float).copy()

    def close(self) -> None:
        pass


def make_arm(
    channel: str,
    arm_type: str,
    gripper_type: str,
    gripper_limits: Optional[np.ndarray],
) -> Any:
    """Build one real i2rt YAM arm on ``channel``.

    ``zero_gravity_mode=False`` makes i2rt latch a PD target at the arm's
    current pose during construction, so the arm is holding position before this
    bridge sends anything. Swapping ``sim=True`` here returns a MuJoCo-backed
    SimRobot with the identical protocol if you want a bench harness.
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name(arm_type),
        gripper_type=GripperType.from_string_name(gripper_type),
        zero_gravity_mode=False,
        gripper_limits_override=gripper_limits,
    )


class RobotInterface:
    """Both arms, the pending action buffer, and the control loop that walks it."""

    def __init__(
        self,
        left: Any,
        right: Any,
        control_hz: float = 30.0,
        max_joint_step: float = 0.15,
        max_gripper_step: float = 0.10,
        arm_tolerance: float = 0.5,
    ):
        self.arms = {"left": left, "right": right}
        self.control_hz = control_hz
        self.max_joint_step = max_joint_step
        self.max_gripper_step = max_gripper_step
        self.arm_tolerance = arm_tolerance

        self._lock = threading.Lock()
        self._pending: Optional[np.ndarray] = None
        self._row = 0
        self._armed = False
        self._refusals = 0
        self._steps = 0
        self._clamped = 0
        self._max_step = 0.0
        self._targets: dict[str, np.ndarray] = {}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- model-facing -----------------------------------------------------
    def get_state(self) -> dict:
        """Measured state in the model's units."""
        state = {}
        for side, arm in self.arms.items():
            q = np.asarray(arm.get_joint_pos(), dtype=float)
            state[f"{side}_joint_pos"] = [float(v) for v in q[:ARM_DOF]]
            state[f"{side}_gripper_pos"] = flip_gripper(q[ARM_DOF])
        return state

    def execute_chunk(self, actions: np.ndarray) -> None:
        """Replace the pending buffer with ``actions`` (24x14)."""
        if actions.shape != CHUNK_SHAPE:
            print(f"  ! ignoring chunk with shape {actions.shape}, expected {CHUNK_SHAPE}")
            return
        with self._lock:
            self._pending = actions
            self._row = 0

    # ---- control loop -----------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._control_loop, name="yam_control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop commanding, then hand the arms back to i2rt in a known state."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for side, arm in self.arms.items():
            try:
                arm.command_joint_pos(np.asarray(arm.get_joint_pos(), dtype=float))
                if hasattr(arm, "enter_gravity_comp_idle"):
                    arm.enter_gravity_comp_idle()
                arm.close()
            except Exception as exc:  # a failing arm must not block the other
                print(f"  ! {side} arm shutdown: {exc}")

    def _next_row(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._pending is None:
                return None
            row = self._pending[min(self._row, CHUNK_SHAPE[0] - 1)]
            self._row = min(self._row + 1, CHUNK_SHAPE[0])
            return row

    def _try_arm(self, row: np.ndarray) -> bool:
        """Refuse to move until the model's first target is near the real pose."""
        worst = 0.0
        measured = {}
        for side, arm in self.arms.items():
            q = np.asarray(arm.get_joint_pos(), dtype=float)
            measured[side] = q
            target = row[0:ARM_DOF] if side == "left" else row[7 : 7 + ARM_DOF]
            worst = max(worst, float(np.max(np.abs(target - q[:ARM_DOF]))))
        if worst > self.arm_tolerance:
            self._refusals += 1
            if self._refusals <= 3 or self._refusals % 20 == 0:
                print(
                    f"  ! holding position: first action is {worst:.3f} rad from the "
                    f"measured pose (limit {self.arm_tolerance} rad)"
                )
            return False
        self._targets = {side: q.copy() for side, q in measured.items()}
        self._armed = True
        print(f"  armed: first action within {worst:.3f} rad of the measured pose")
        return True

    def _control_loop(self) -> None:
        dt = 1.0 / self.control_hz
        next_t = time.monotonic() + dt
        while not self._stop.is_set():
            row = self._next_row()
            if row is not None:
                if not self._armed:
                    self._try_arm(row)
                if self._armed:
                    self._apply(row)
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            next_t += dt

    def _apply(self, row: np.ndarray) -> None:
        for side, arm in self.arms.items():
            base = 0 if side == "left" else 7
            desired = np.empty(ARM_DOF + 1)
            desired[:ARM_DOF] = row[base : base + ARM_DOF]
            desired[ARM_DOF] = flip_gripper(row[base + ARM_DOF])

            current = self._targets[side]
            delta = desired - current
            step = np.empty_like(delta)
            step[:ARM_DOF] = np.clip(delta[:ARM_DOF], -self.max_joint_step, self.max_joint_step)
            step[ARM_DOF] = np.clip(delta[ARM_DOF], -self.max_gripper_step, self.max_gripper_step)
            if not np.allclose(step, delta):
                self._clamped += 1
            self._max_step = max(self._max_step, float(np.max(np.abs(step[:ARM_DOF]))))

            target = current + step
            self._targets[side] = target
            arm.command_joint_pos(target)
        self._steps += 1

    def snapshot(self) -> dict[str, np.ndarray]:
        """Both arms' raw i2rt joint vectors, read back to back."""
        return {side: np.asarray(arm.get_joint_pos(), dtype=float) for side, arm in self.arms.items()}

    def stats(self) -> dict:
        return {
            "steps": self._steps,
            "clamped": self._clamped,
            "refusals": self._refusals,
            "max_step": round(self._max_step, 4),
        }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class Session:
    """Chunk bookkeeping, so the run can report what the model actually sent."""

    def __init__(self) -> None:
        self.chunks = 0
        self.inference: list[float] = []
        self.skew_us: list[int] = []
        self.first: Optional[float] = None
        self.last: Optional[float] = None

    def record(self, inference_seconds: float, view_skew_us: Optional[int] = None) -> None:
        now = time.monotonic()
        self.chunks += 1
        self.inference.append(inference_seconds)
        if view_skew_us is not None:
            self.skew_us.append(view_skew_us)
        self.first = self.first if self.first is not None else now
        self.last = now

    def report(self) -> str:
        if not self.chunks:
            return "chunks: 0"
        span = (self.last - self.first) if self.last and self.first else 0.0
        rate = (self.chunks - 1) / span if span > 0 else 0.0
        inf = np.array(self.inference)
        return (
            f"chunks: {self.chunks} over {span:.1f}s ({rate:.2f}/s) | "
            f"inference_seconds mean {inf.mean():.3f} p50 {np.median(inf):.3f} "
            f"min {inf.min():.3f} max {inf.max():.3f}"
            + (
                f" | view_skew_us p50 {int(np.median(self.skew_us))} "
                f"max {max(self.skew_us)} (n={len(self.skew_us)})"
                if self.skew_us
                else ""
            )
        )


def build_reactor(robot: RobotInterface, session: Session, ready: asyncio.Event, api_url: str) -> Reactor:
    reactor = Reactor(MODEL, api_key=os.environ["REACTOR_API_KEY"], api_url=api_url)

    @reactor.on_status
    def on_status(status: Any) -> None:
        print(f"status: {status}")
        if status == ReactorStatus.READY:
            ready.set()

    @reactor.on_message
    def on_message(message: Any) -> None:
        data = message.get("data", {}) if isinstance(message, dict) else {}
        if isinstance(message, dict) and message.get("type") == "action_chunk":
            actions = np.asarray(data["actions"], dtype=np.float64)  # (24, 14)
            robot.execute_chunk(actions)
            session.record(float(data["inference_seconds"]), data.get("view_skew_us"))
            if session.chunks <= 3 or session.chunks % 10 == 0:
                print(
                    f"chunk {data['chunk_index']}: inference "
                    f"{data['inference_seconds']:.3f}s | {robot.stats()}"
                )
        else:
            print(f"message: {message}")

    return reactor


async def connect_with_capacity_retry(build: Any, retry_wait: float = 60.0) -> Reactor:
    """Connect, tolerating the one-slot deployment being busy.

    A held slot comes back as ``429 no available capacity`` from session
    creation; wait it out once before giving up.
    """
    for attempt in (1, 2):
        reactor = build()
        try:
            await reactor.connect()
            return reactor
        except RuntimeError as exc:
            if "429" not in str(exc) or attempt == 2:
                raise
            print(f"session slot busy ({exc}); retrying in {retry_wait:.0f}s")
            await asyncio.sleep(retry_wait)
    raise RuntimeError("unreachable")


def synthetic_chunk(state: dict, amplitude: float = 0.08) -> np.ndarray:
    """A 24x14 chunk: a small sinusoid around the measured pose, grippers closing.

    Values are in the model's units -- absolute joint targets in rad and
    0=open..1=closed grippers -- so it exercises the same path a real chunk does.
    """
    rows = np.zeros(CHUNK_SHAPE)
    phase = np.linspace(0, np.pi, CHUNK_SHAPE[0])
    for i, side in enumerate(("left", "right")):
        base = i * 7
        home = np.asarray(state[f"{side}_joint_pos"])
        for j in range(ARM_DOF):
            rows[:, base + j] = home[j] + amplitude * np.sin(phase) * (1 + j * 0.2)
        rows[:, base + ARM_DOF] = np.linspace(0.0, 1.0, CHUNK_SHAPE[0])  # open -> closed
    return rows


async def dry_run(robot: RobotInterface, args: argparse.Namespace) -> int:
    """Offline check of the control path: no Reactor, no CAN, no cameras."""
    checks: list[tuple[str, bool, str]] = []
    before_raw = robot.snapshot()
    before = robot.get_state()
    chunk = synthetic_chunk(before)
    print(f"synthetic chunk {chunk.shape}, row0 left joints {np.round(chunk[0, :ARM_DOF], 4)}")
    print(f"  gripper column: model {chunk[0, ARM_DOF]:.2f} -> {chunk[-1, ARM_DOF]:.2f} (open -> closed)")

    # 1. Walk the chunk, re-posting it the way a live stream of chunks would.
    end = time.monotonic() + args.dry_run
    while time.monotonic() < end:
        robot.execute_chunk(chunk)
        await asyncio.sleep(0.25)

    after_raw = robot.snapshot()
    after = {}
    for side, q in after_raw.items():
        after[f"{side}_joint_pos"] = [float(v) for v in q[:ARM_DOF]]
        after[f"{side}_gripper_pos"] = flip_gripper(q[ARM_DOF])

    print(f"\nbefore (model units): {before}")
    print(f"after  (model units): {after}")
    print(f"raw i2rt qpos (gripper 0=closed..1=open): "
          f"{ {s: np.round(q, 4) for s, q in after_raw.items()} }")

    moved = max(
        float(np.max(np.abs(np.asarray(after[f"{s}_joint_pos"]) - np.asarray(before[f"{s}_joint_pos"]))))
        for s in ("left", "right")
    )
    checks.append(("arms moved", moved > 1e-3, f"max joint motion {moved:.4f} rad"))

    # 2. Gripper flip, both directions. Read: the stub holds i2rt units and
    #    get_state reports model units. Write: the chunk's model-unit gripper
    #    column landed on the stub as its complement.
    read_ok = abs(after_raw["left"][ARM_DOF] + after["left_gripper_pos"] - 1.0) < 1e-9
    checks.append((
        "gripper flip on read",
        read_ok,
        f"i2rt {after_raw['left'][ARM_DOF]:.3f} <-> model {after['left_gripper_pos']:.3f}",
    ))
    write_ok = after_raw["left"][ARM_DOF] < before_raw["left"][ARM_DOF]
    checks.append((
        "gripper flip on write",
        write_ok,
        f"chunk commanded closing (model 0->1), i2rt gripper opened value fell "
        f"{before_raw['left'][ARM_DOF]:.3f} -> {after_raw['left'][ARM_DOF]:.3f}",
    ))

    # 3. Rate limiter: demand a far pose and confirm no tick ever exceeds the clamp.
    far = np.tile(np.concatenate([np.full(ARM_DOF, 2.0), [0.0]] * 2), (CHUNK_SHAPE[0], 1))
    robot.execute_chunk(far)
    await asyncio.sleep(1.0)
    stats = robot.stats()
    checks.append((
        "rate limiter clamps",
        stats["clamped"] > 0 and stats["max_step"] <= args.max_joint_step + 1e-9,
        f"max per-tick step {stats['max_step']:.4f} rad <= {args.max_joint_step} "
        f"({stats['clamped']} clamped ticks)",
    ))

    # 4. Startup guard: a fresh pair whose first chunk is far away must refuse.
    guard = RobotInterface(MockArm("left"), MockArm("right"), control_hz=args.control_hz,
                           arm_tolerance=args.arm_tolerance)
    guard.start()
    guard.execute_chunk(far)
    await asyncio.sleep(0.5)
    guard_stats = guard.stats()
    guard.stop()
    checks.append((
        "startup guard refuses a far first chunk",
        guard_stats["steps"] == 0 and guard_stats["refusals"] > 0,
        f"{guard_stats['refusals']} refusals, {guard_stats['steps']} commands issued",
    ))

    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"control: {robot.stats()}")

    passed = all(ok for _, ok, _ in checks)
    print("DRY RUN PASS" if passed else "DRY RUN FAIL")
    return 0 if passed else 1


async def run(args: argparse.Namespace) -> int:
    # --- cameras ---
    sources: dict[str, CameraSource] = {}
    if args.mock:
        sources = {name: SyntheticCamera(name) for name in CAMERAS}
    else:
        specs = dict(item.split("=", 1) for item in args.cam)
        missing = [n for n in CAMERAS if n not in specs]
        if missing:
            print(f"error: missing --cam for {', '.join(missing)}", file=sys.stderr)
            return 2
        sources = {name: OpenCVCamera(name, specs[name]) for name in CAMERAS}

    # --- arms ---
    if args.mock:
        arms = (MockArm("left"), MockArm("right"))
    else:
        arms = (
            make_arm(args.left_channel, args.arm_type, args.gripper_type, args.left_gripper_limits),
            make_arm(args.right_channel, args.arm_type, args.gripper_type, args.right_gripper_limits),
        )

    robot = RobotInterface(
        *arms,
        control_hz=args.control_hz,
        max_joint_step=args.max_joint_step,
        max_gripper_step=args.max_gripper_step,
        arm_tolerance=args.arm_tolerance,
    )
    print(f"initial state: {robot.get_state()}")
    for name, source in sources.items():
        print(f"camera {name}: {source.read_rgb().shape} {source.read_rgb().dtype}")
    robot.start()

    if args.dry_run:
        try:
            return await dry_run(robot, args)
        finally:
            robot.stop()
            for source in sources.values():
                source.close()
            print("shutdown clean")

    session = Session()
    ready = asyncio.Event()
    reactor: Optional[Reactor] = None
    deadline = time.monotonic() + args.duration if args.duration else None

    try:
        reactor = await connect_with_capacity_retry(
            lambda: build_reactor(robot, session, ready, args.api_url)
        )
        await asyncio.wait_for(ready.wait(), timeout=120)

        for name in CAMERAS:
            await reactor.publish_track(name, CameraTrack(sources[name]))

        # Set the task once; the episode starts when prompt + state are in.
        await reactor.send_command("set_prompt", {"prompt": args.prompt})
        if args.pair_by_capture_time:
            # Model >= 1.1.0; older deployments reject the unknown command.
            await reactor.send_command(
                "set_pair_by_capture_time", {"pair_by_capture_time": True}
            )

        # Closed loop: stream measured state continuously. Every chunk the model
        # returns is conditioned on the latest state we sent.
        tick = 0
        while deadline is None or time.monotonic() < deadline:
            state = robot.get_state()
            await reactor.send_command("set_left_joint_pos", {"left_joint_pos": state["left_joint_pos"]})
            await reactor.send_command("set_left_gripper_pos", {"left_gripper_pos": state["left_gripper_pos"]})
            await reactor.send_command("set_right_joint_pos", {"right_joint_pos": state["right_joint_pos"]})
            await reactor.send_command("set_right_gripper_pos", {"right_gripper_pos": state["right_gripper_pos"]})
            tick += 1
            if tick % int(args.state_hz * 5) == 0:
                lq = np.round(state["left_joint_pos"], 4)
                rq = np.round(state["right_joint_pos"], 4)
                print(f"  qpos L {lq} g={state['left_gripper_pos']:.3f}")
                print(f"       R {rq} g={state['right_gripper_pos']:.3f}")
            await asyncio.sleep(1.0 / args.state_hz)
    finally:
        if reactor is not None:
            await reactor.disconnect()
        robot.stop()
        for source in sources.values():
            source.close()

    print(f"\n{session.report()}")
    print(f"control: {robot.stats()}")
    print(f"final state: {robot.get_state()}")
    return 0 if session.chunks else 1


# ---------------------------------------------------------------------------
def _limits(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    closed, opened = (float(v) for v in text.split(","))
    return np.array([closed, opened])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left-channel", default="can_follower_l", help="CAN interface for the left arm")
    p.add_argument("--right-channel", default="can_follower_r", help="CAN interface for the right arm")
    p.add_argument("--arm-type", default="yam")
    p.add_argument("--gripper-type", default="linear_4310")
    p.add_argument("--left-gripper-limits", type=_limits, default=None, help="closed,open -- skips calibration")
    p.add_argument("--right-gripper-limits", type=_limits, default=None, help="closed,open -- skips calibration")
    p.add_argument("--cam", action="append", default=[], metavar="NAME=SOURCE",
                   help="camera mapping, e.g. --cam top=0 (repeat for left and right)")
    p.add_argument("--prompt", default="fold the towel neatly with both arms")
    p.add_argument("--api-url", default=os.environ.get("REACTOR_API_URL", "https://api.reactor.inc"))
    p.add_argument("--pair-by-capture-time", action="store_true",
                   help="enable server-side cross-camera capture-time alignment (model >= 1.1.0)")
    p.add_argument("--mock", action="store_true", help="stub arms + synthetic frames; no CAN, no cameras")
    p.add_argument("--dry-run", type=float, nargs="?", const=5.0, default=0.0, metavar="SECONDS",
                   help="skip Reactor: feed a synthetic chunk to the control loop for N seconds")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl-C)")
    p.add_argument("--state-hz", type=float, default=10.0)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--max-joint-step", type=float, default=0.15, help="rad per control tick")
    p.add_argument("--max-gripper-step", type=float, default=0.10, help="normalized units per control tick")
    p.add_argument("--arm-tolerance", type=float, default=0.5, help="rad; refuse to start further than this")
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        pass
