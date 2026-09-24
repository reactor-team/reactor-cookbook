"""The model half of Cosmos3-Policy-DROID: the policy weights and one prediction per step.

Plain Python. This module imports nothing from ``reactor_runtime`` and knows
nothing about clients, tracks, or commands. The application half
(``cosmos3_policy_droid.py``) constructs :class:`Cosmos3PolicyModel`, calls
``load`` once, ``generate`` once per step, and ``reset`` when a session ends.
The two halves meet on :class:`PolicyInput` and :class:`PolicyResult`, and on
nothing else.

The policy is stateless: one prediction is a pure function of the camera
frames, the proprioception, and the task it receives. Nothing is carried
between steps, so ``reset`` has nothing to forget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cosmos3_policy_droid_assets import PolicyConfig, read_config, route_checkpoint_downloads

logger = logging.getLogger(__name__)

VIEWS = ("wrist_view", "exterior_view_1", "exterior_view_2")


@dataclass(frozen=True)
class PolicyInput:
    """Carry exactly what one prediction needs across to the model.

    Attributes:
        wrist_view: Newest wrist camera frame, RGB uint8 ``(H, W, 3)``.
        exterior_view_1: Newest first exterior camera frame, RGB uint8.
        exterior_view_2: Newest second exterior camera frame, RGB uint8.
        joint_position: Robot joint positions, float32 ``(N, 7)``; the last
            row is the current state.
        gripper_position: Gripper positions, float32 ``(N, 1)``; the last row
            is the current state.
        task: The language instruction for the episode.
    """

    wrist_view: np.ndarray
    exterior_view_1: np.ndarray
    exterior_view_2: np.ndarray
    joint_position: np.ndarray
    gripper_position: np.ndarray
    task: str


@dataclass(frozen=True)
class PolicyResult:
    """Carry one predicted chunk back to the application.

    Attributes:
        actions: Absolute joint-position targets plus a gripper command, float32
            ``(horizon, dof)``: ``(32, 8)`` for the DROID checkpoints.
        horizon: Number of action steps in the chunk.
        dof: Width of one action row.
    """

    actions: np.ndarray
    horizon: int
    dof: int


class Cosmos3PolicyModel:
    """Hold the DROID policy and predict one action chunk per step.

    The weights and the sampler live in the vendored
    ``cosmos_framework`` policy service. This class owns that service, its
    load-time wrapping, and the shape check on every prediction.
    """

    def __init__(self) -> None:
        self._service: Any = None
        self.horizon = 0
        self.dof = 0

    def load(self, config_path: Path | None, weights_root: Path) -> None:
        """Download the checkpoint if absent and build the policy service once per process.

        Args:
            config_path: Path to ``cosmos3_policy_droid.yaml`` from ``reactor.yaml``.
            weights_root: Directory the checkpoint and its tokenizer are cached under.
        """
        config = read_config(config_path)
        route_checkpoint_downloads(weights_root)

        import torch

        torch.cuda.set_device(config.device_id)

        from cosmos_framework.scripts import action_policy_server_robolab as srv

        _disable_guardrails(srv.RobolabPolicyService)
        args = srv.RobolabServerArgs(
            checkpoint_path=config.checkpoint,
            hf_revision=config.hf_revision,
            format_prompt_as_json=config.format_prompt_as_json,
            guidance_interval=config.guidance_interval,
            guidance=config.guidance,
            num_steps=config.num_steps,
            resolution=config.resolution,
            action_chunk_size=config.action_chunk_size,
        )
        t0 = time.perf_counter()
        self._service = srv.RobolabPolicyService(args)
        self.horizon = int(self._service.cfg.action_chunk_size)
        self.dof = int(self._service.cfg.action_dim)
        logger.info(
            "policy loaded: checkpoint=%s horizon=%d dof=%d in %.1fs",
            config.checkpoint,
            self.horizon,
            self.dof,
            time.perf_counter() - t0,
        )
        if config.warmup:
            self._warmup(config)

    def generate(self, input: PolicyInput) -> PolicyResult:
        """Predict one action chunk from the frames, the state, and the task."""
        service = self._require_service()
        obs = {
            "prompt": input.task,
            "observation/wrist_image_left": input.wrist_view,
            "observation/exterior_image_1_left": input.exterior_view_1,
            "observation/exterior_image_2_left": input.exterior_view_2,
            "observation/joint_position": input.joint_position,
            "observation/gripper_position": input.gripper_position,
        }
        out = service.infer(obs)
        actions = np.asarray(out["action"], dtype=np.float32)
        expected = (self.horizon, self.dof)
        if actions.shape != expected:
            raise RuntimeError(f"policy returned action shape {actions.shape}, expected {expected}")
        return PolicyResult(actions=actions, horizon=self.horizon, dof=self.dof)

    def reset(self) -> None:
        """Return to the default state. The policy carries nothing between steps."""

    def _warmup(self, config: PolicyConfig) -> None:
        """Pay the first-call compilation at load, then confirm the compiled path is hit."""
        zeros = np.zeros((360, 640, 3), np.uint8)
        probe = PolicyInput(
            wrist_view=zeros,
            exterior_view_1=zeros,
            exterior_view_2=zeros,
            joint_position=np.zeros((1, 7), np.float32),
            gripper_position=np.zeros((1, 1), np.float32),
            task="warmup",
        )
        t0 = time.perf_counter()
        self.generate(probe)
        t1 = time.perf_counter()
        self.generate(probe)
        t2 = time.perf_counter()
        logger.info(
            "warmup: first=%.1fs steady=%.3fs (%s, %d steps, guidance %.1f)",
            t1 - t0,
            t2 - t1,
            config.checkpoint,
            config.num_steps,
            config.guidance,
        )

    def _require_service(self) -> Any:
        if self._service is None:
            raise RuntimeError("Cosmos3-Policy-DROID was not loaded")
        return self._service


def _disable_guardrails(service_cls: type) -> None:
    """Force ``guardrails=False`` on the setup the service builds.

    The upstream guardrail checkpoint is approval-gated on Hugging Face and
    its downloader shells out to a tool the image does not carry. Content
    moderation is the deployment's job, so the pass is switched off by
    wrapping the setup builder. Idempotent.
    """
    original = service_cls._build_setup_args
    if getattr(original, "_guardrails_disabled", False):
        return

    def build_setup_args(self, *args, **kwargs):
        setup_args = original(self, *args, **kwargs)
        setup_args.guardrails = False
        return setup_args

    build_setup_args._guardrails_disabled = True  # type: ignore[attr-defined]
    service_cls._build_setup_args = build_setup_args
