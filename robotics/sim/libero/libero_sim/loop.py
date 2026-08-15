# ──────────────────────────────────────────────────────────────────────────
# Rollout loop + shared state.
#
# This policy is LOCK-STEP, not open-loop receding-horizon: the engine
# advances only when the client reports the chunk it executed. So the cycle
# is:
#
#   chunk arrives -> execute N steps -> render -> echo what was executed -> wait
#
# and the sim deliberately does NOT step while waiting for the next chunk.
# The robot holding still between predictions is the honest reading of the
# contract.
#
# Threading: the env is owned by the MAIN thread and the reactor bridge gets
# a thread of its own, forced by macOS (see SimDriver). Everything the
# bridge touches here is lock-guarded.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .contract import ACTION_HORIZON, CAM_SIZE, SEED_SKIP_STEPS, VIEWS, decode_chunk, encode_executed
from .env import LiberoEnv

log = logging.getLogger("libero.loop")


def _to_model_size(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The policy is trained at CAM_SIZE. When the env renders larger (so the
    recorder can capture a high-res clip), area-downsample each view back to
    CAM_SIZE before it reaches the published tracks. A no-op at CAM_SIZE."""
    out: dict[str, np.ndarray] = {}
    for name, img in frames.items():
        if img.shape[0] != CAM_SIZE or img.shape[1] != CAM_SIZE:
            img = cv2.resize(img, (CAM_SIZE, CAM_SIZE), interpolation=cv2.INTER_AREA)
        out[name] = np.ascontiguousarray(img)
    return out


@dataclass
class RolloutDiagnostics:
    """Enough to tell a stalled rollout from a failing one after the fact."""

    chunks_received: int = 0
    steps_executed: int = 0
    echoes_sent: int = 0
    episodes: int = 0
    successes: int = 0
    short_chunks: int = 0  # predictions shorter than ACTION_HORIZON
    # Chunks that arrived while the previous one was still executing.
    # Lock-step says that cannot happen, so each one means some steps were
    # chosen against a state the sim had already left.
    stray_chunks: int = 0

    def note_chunk(self, steps: list[np.ndarray]) -> None:
        self.chunks_received += 1
        if len(steps) < ACTION_HORIZON:
            self.short_chunks += 1


class RolloutState:
    """Thread-safe hand-off between the sim (owns the env; produces frames and
    the executed-action echo) and the bridge (produces chunks, sends the
    echo)."""

    def __init__(
        self,
        env: LiberoEnv,
        *,
        exec_steps: int = ACTION_HORIZON,
        seed_skip: int = SEED_SKIP_STEPS,
        flip_frames: bool = True,
        max_episode_steps: int = 0,
        max_episodes: int = 0,
        stop_on_success: threading.Event | None = None,
    ):
        self._env = env
        self._exec_steps = max(1, min(exec_steps, ACTION_HORIZON))
        self._seed_skip = max(0, min(seed_skip, self._exec_steps - 1))
        self._flip = flip_frames
        self._max_ep_steps = max(0, max_episode_steps)
        # End the run after this many episodes even without a success (0 =
        # keep retrying until stop_on_success). Lets a caller capture exactly
        # one attempt per task for a montage, hit or miss.
        self._max_episodes = max(0, max_episodes)
        self._episodes_done = 0
        self._stop_on_success = stop_on_success
        self._lock = threading.Lock()

        self._frames: dict[str, np.ndarray | None] = {n: None for n in VIEWS}
        # Full-resolution renders for the optional MP4 recorder only. The
        # policy tracks always publish the downsampled CAM_SIZE frames above.
        self._frames_hi: dict[str, np.ndarray | None] = {n: None for n in VIEWS}
        # Bumped on every render. The tracks publish per render rather than
        # at their own rate (see frame_reader), so this is what paces the wire.
        self._frame_seq = 0
        self._queue: deque[np.ndarray] = deque()
        self._executed: list[np.ndarray] = []
        self._pending_echo: str | None = None
        self._reset_requested = False
        self._episode_start = True  # gates the one-time empty echo
        self._ep_chunk = 0  # chunks served since the last reset
        self._ep_steps = 0
        self._is_idle = True
        self.diag = RolloutDiagnostics()

        self._prime()

    # ── reads for the bridge ────────────────────────────────────────────────
    def frame_source(self, name: str) -> Callable[[], np.ndarray | None]:
        def _get() -> np.ndarray | None:
            with self._lock:
                return self._frames_hi.get(name)

        return _get

    def frame_reader(self, name: str) -> Callable[[], tuple[np.ndarray | None, int]]:
        """Latest frame for a view together with the render sequence it came
        from.

        The sequence is what lets a track publish one video frame per env
        step (tracks.CameraTrack) instead of resampling this slot at its own
        rate. The policy pairs one frame with one action (its training data
        is 1:1, and the engine encodes the frames observed during a chunk as
        the video that chunk produced), so repeating the last render while
        the sim waits for the next chunk would hand the model a still video
        alongside a moving arm.
        """

        def _get() -> tuple[np.ndarray | None, int]:
            with self._lock:
                return self._frames.get(name), self._frame_seq

        return _get

    def take_pending_echo(self) -> str | None:
        """Pop the executed-action echo for the chunk that just finished, if
        any. Returns exactly once per completed chunk; sending it is what
        unblocks the next prediction."""
        with self._lock:
            echo, self._pending_echo = self._pending_echo, None
        return echo

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def is_episode_start(self) -> bool:
        with self._lock:
            return self._episode_start

    def is_idle(self) -> bool:
        """True if the most recent tick had no queued step to execute
        (lock-step wait for the next chunk, or a reset in progress)."""
        with self._lock:
            return self._is_idle

    # ── inbound chunk (bridge on_message) ───────────────────────────────────
    def submit_chunk(self, pred: dict) -> None:
        steps = decode_chunk(pred.get("action"))
        if not steps:
            log.warning("ignoring empty action chunk (step=%s)", pred.get("step"))
            return
        self.diag.note_chunk(steps)
        with self._lock:
            self._ep_chunk += 1
            if self._queue:
                # Lock-step means we should be idle when a chunk lands. If we
                # aren't, the model predicted before our echo went out.
                self.diag.stray_chunks += 1
                log.warning(
                    "chunk arrived with %d steps still queued; dropping the remainder",
                    len(self._queue),
                )
            # The seed chunk's leading frame is conditioning, not a
            # prediction, so it is dropped rather than executed.
            skip = self._seed_skip if self._ep_chunk == 1 else 0
            self._queue = deque(steps[skip : self._exec_steps])
            self._executed = []
            self._episode_start = False

    # ── one control tick (main thread; owns the env) ─────────────────────────
    def _tick(self) -> None:
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                self._queue.clear()
                self._executed = []
                self._pending_echo = None
                self._ep_chunk = 0
                do_reset = True
            else:
                do_reset = False
            action = self._queue.popleft() if self._queue else None
            drained = action is not None and not self._queue
            self._is_idle = action is None

        if do_reset:
            self._ep_steps = 0
            self._env.reset(self._env.init_state_id)  # same episode, run again
            self.diag.episodes += 1
            self._publish_frames()
            with self._lock:
                self._episode_start = True
            return

        if action is None:
            return  # lock-step idle: awaiting the next chunk, hold still

        _reward, done = self._env.step(action)
        self.diag.steps_executed += 1
        self._ep_steps += 1
        with self._lock:
            self._executed.append(action)

        # Render after every step so the published tracks stay live, but only
        # emit the echo once the whole chunk has been executed.
        self._publish_frames()

        # A failing episode has no natural end: LIBERO replaces robosuite's
        # done with check_success(), so `done` IS success and only the cap
        # can end a failure.
        capped = self._max_ep_steps > 0 and self._ep_steps >= self._max_ep_steps
        if drained and not (done or capped):
            with self._lock:
                executed = list(self._executed)
            try:
                echo = encode_executed(executed)
            except ValueError as exc:
                log.error("%s", exc)
                return
            with self._lock:
                self._pending_echo = echo

        if done or capped:
            success = self._env.check_success()
            self.diag.successes += int(success)
            self._episodes_done += 1
            episode_cap = self._max_episodes and self._episodes_done >= self._max_episodes
            end_run = (success or episode_cap) and self._stop_on_success is not None
            if end_run:
                log.info(
                    "episode done (init_state=%d success=%s steps=%d); ending run",
                    self._env.init_state_id, success, self._ep_steps,
                )
                self._stop_on_success.set()
                return
            log.info(
                "episode done (init_state=%d success=%s steps=%d); resetting",
                self._env.init_state_id, success, self._ep_steps,
            )
            self.request_reset()

    def _prime(self) -> None:
        """Render once on the constructing thread so tracks have data
        immediately."""
        self._publish_frames()

    def _publish_frames(self) -> None:
        frames = self._env.get_frames(flip=self._flip)
        model = _to_model_size(frames)
        with self._lock:
            self._frames = model
            self._frames_hi = frames
            self._frame_seq += 1


class SimDriver:
    """Advances the env at the control rate ON THE CALLING THREAD.

    This is NOT a thread. On macOS MuJoCo's offscreen GL context can only be
    created on the main thread, and once created it can only be used from
    that thread: rendering from anywhere else segfaults the process rather
    than raising. So run() blocks the main thread and the reactor bridge is
    the one that gets moved off it (bridge.BridgeThread).
    """

    def __init__(
        self,
        rollout: RolloutState,
        hz: int,
        max_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
    ):
        self._rollout = rollout
        self._period = 1.0 / hz
        self._max_seconds = max_seconds
        # Shared with rollout's stop_on_success, when given, so a successful
        # episode ends run() the same way Ctrl-C does.
        self._stop_event = stop_event if stop_event is not None else threading.Event()

    def run(self) -> None:
        log.info("sim driver started at %.1f Hz", 1.0 / self._period)
        deadline = time.perf_counter() + self._max_seconds if self._max_seconds > 0 else None
        while not self._stop_event.is_set():
            if deadline is not None and time.perf_counter() >= deadline:
                log.info("reached --max-seconds")
                break
            t0 = time.perf_counter()
            try:
                self._rollout._tick()
            except Exception:
                log.exception("sim tick failed")
            dt = time.perf_counter() - t0
            if dt < self._period:
                self._stop_event.wait(self._period - dt)
        log.info("sim driver stopped")

    def stop(self) -> None:
        self._stop_event.set()
