# ──────────────────────────────────────────────────────────────────────────
# LIBERO env wrapper: the training distribution itself, not a
# reimplementation. robosuite's OSC_POSE controller solves internally, the
# named MJCF cameras are rendered by robosuite, and env.reset() +
# set_init_state() is in-distribution by construction because the init
# states are the benchmark's own pruned_init tensors.
#
# The one transform this file owns is image orientation; see get_frames().
#
# NOTE: get_task_init_states() loads its .pruned_init files with torch.load()
# under the hood, written back when weights_only defaulted to False. PyTorch
# 2.6 flipped that default and refuses the plain numpy pickle inside them, so
# this example pins torch<2.6 (see pyproject.toml) rather than special-casing
# the load.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from .contract import ACTION_DOF, CAM_SIZE, CONTROL_HZ, VIEWS

log = logging.getLogger("libero.env")


@dataclass
class EnvConfig:
    suite: str = "libero_10"  # LIBERO-Long
    task_id: int = 0
    init_state_id: int = 0
    cam_size: int = CAM_SIZE
    seed: int = 0
    control_hz: int = CONTROL_HZ
    # Zero-action steps after set_init_state, letting the scene settle before
    # the policy sees it. LIBERO's own eval does 5.
    settle_steps: int = 5


class LiberoEnv:
    """Owns the MuJoCo env. Single-threaded by contract: construct it on the
    main thread (macOS creates the GLFW/Cocoa render context there) and step
    it from that same thread only."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        suites = benchmark.get_benchmark_dict()
        if cfg.suite not in suites:
            raise SystemExit(f"unknown suite {cfg.suite!r}; have {sorted(suites)}")
        self._suite = suites[cfg.suite]()
        if not 0 <= cfg.task_id < self._suite.n_tasks:
            raise SystemExit(
                f"task_id {cfg.task_id} out of range for {cfg.suite} "
                f"(0..{self._suite.n_tasks - 1})"
            )
        self._task = self._suite.get_task(cfg.task_id)
        bddl = self._suite.get_task_bddl_file_path(cfg.task_id)

        # ignore_done is load-bearing: LIBERO overwrites robosuite's done with
        # check_success(), but robosuite still latches its own done at the
        # (large) default horizon and then raises on the next step. This
        # keeps that latch from ever firing.
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=cfg.cam_size,
            camera_widths=cfg.cam_size,
            control_freq=cfg.control_hz,
            ignore_done=True,
        )
        self._env.seed(cfg.seed)
        self._init_states = self._suite.get_task_init_states(cfg.task_id)
        self._obs: dict | None = None
        self._done = False
        self._last_gripper = 0.0
        log.info("%s task %d: %r (%d init states)",
                  cfg.suite, cfg.task_id, self.language, len(self._init_states))
        self.reset()

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def language(self) -> str:
        """The task string the policy is conditioned on."""
        return self._task.language

    @property
    def done(self) -> bool:
        return self._done

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self, init_state_id: int | None = None) -> dict:
        idx = self.cfg.init_state_id if init_state_id is None else init_state_id
        if not 0 <= idx < len(self._init_states):
            raise IndexError(
                f"init_state_id {idx} out of range for {self.cfg.suite} task "
                f"{self.cfg.task_id} (0..{len(self._init_states) - 1})"
            )
        self.init_state_id = idx
        self._env.reset()
        self._obs = self._env.set_init_state(self._init_states[idx])
        # Objects are still falling when set_init_state returns; let the
        # scene settle before the policy sees it (LIBERO's own eval does the
        # same).
        for _ in range(max(0, self.cfg.settle_steps)):
            self._obs, _reward, _done, _info = self._env.step(np.zeros(ACTION_DOF))
        self._done = False
        self._last_gripper = 0.0
        return self._obs

    def step(self, action: np.ndarray) -> tuple[float, bool]:
        """Step one 7-DoF OSC_POSE delta. Returns (reward, done)."""
        a = np.asarray(action, dtype=float).ravel()[:ACTION_DOF]
        obs, reward, done, _info = self._env.step(a)
        self._obs = obs
        self._done = bool(done)
        self._last_gripper = float(a[6]) if a.size >= ACTION_DOF else self._last_gripper
        return float(reward), self._done

    def check_success(self) -> bool:
        return bool(self._env.check_success())

    def hold_action(self) -> np.ndarray:
        """Zero deltas, gripper held where it last was. OSC_POSE reads a
        zero delta as 'maintain current pose'."""
        a = np.zeros(ACTION_DOF, dtype=float)
        a[6] = self._last_gripper
        return a

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:  # robosuite is noisy on teardown
            pass

    # ── observation ─────────────────────────────────────────────────────────
    def get_frames(self, flip: bool = True) -> dict[str, np.ndarray]:
        """The two policy views as HxWx3 uint8 RGB, keyed by track name.

        MuJoCo's offscreen renderer returns bottom-up arrays, so the raw obs
        image is upside down. `flip` applies a VERTICAL flip (`[::-1]`) to
        match the training frames.

        NOTE: this is deliberately NOT the 180-degree rotation (`[::-1,
        ::-1]`) some other LIBERO harnesses apply; the two differ by a
        horizontal mirror. Getting this wrong doesn't raise: the wrapper just
        mirrors every observation against the training distribution and the
        policy still emits confident, wrong actions. Task success rate is the
        real adjudicator.
        """
        assert self._obs is not None, "reset() before get_frames()"
        out: dict[str, np.ndarray] = {}
        for name, key in VIEWS.items():
            img = np.asarray(self._obs[key], dtype=np.uint8)
            out[name] = np.ascontiguousarray(img[::-1] if flip else img)
        return out
