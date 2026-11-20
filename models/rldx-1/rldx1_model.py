"""Own RLDX policy weights, RTC inference and episode memory without Reactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rldx1_rtc import RTCTiming, resolve_rtc_timing

VIEWS = ("left_view", "right_view", "wrist_view")
TV = 4
_STATE_DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


@dataclass(frozen=True)
class RLDXSetup:
    """Checkpoint-derived observation and action contract."""

    views: tuple[str, ...]
    video_deltas: tuple[int, ...]
    state_dims: dict[str, int]
    action_dims: dict[str, int]
    timing: RTCTiming
    rtc_mode: str
    embodiment: str


@dataclass(frozen=True)
class RLDXModelInput:
    """One episode identity and an optional aligned native observation."""

    episode_id: int
    observation: dict[str, Any] | None
    options: dict[str, Any] | None


@dataclass(frozen=True)
class RLDXModelResult:
    """The applied episode and successful policy prediction count."""

    episode_id: int
    actions: dict[str, Any] | None
    completed_predictions: int


class PolicyNotLoaded(RuntimeError):
    """Policy weights must be loaded before a step."""


class RLDXModel:
    """Keep policy inference and memory behind a plain three-method interface."""

    def __init__(self) -> None:
        self._policy: Any = None
        self._episode_id: int | None = None
        self._completed_predictions = 0

    def load(self, config: dict[str, Any], weights_root: Path) -> RLDXSetup:
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

        checkpoint = Path(config.get("checkpoint_dir", ""))
        model_path = str(
            checkpoint if checkpoint.is_absolute() else weights_root / checkpoint
        )
        embodiment = getattr(
            EmbodimentTag, config.get("embodiment_tag", "GENERAL_EMBODIMENT")
        )
        device = f"cuda:{config.get('device_id', 0)}"

        policy = RLDXPolicy(
            embodiment_tag=embodiment,
            model_path=model_path,
            device=device,
            strict=False,
            rtc_inference_mode=config.get("rtc_inference_mode"),
            rtc_inference_delay=config.get("rtc_inference_delay"),
            rtc_inference_exec_horizon=config.get("rtc_inference_exec_horizon"),
            rtc_jacobian_beta=config.get("rtc_jacobian_beta"),
            rtc_jacobian_steps_only=config.get("rtc_jacobian_steps_only"),
        )
        # The sim wrapper takes the flat video.*/state.*/annotation.* obs format
        # and returns flat action.* keys — exactly the quickstart recipe.
        self._policy = RLDXSimPolicyWrapper(policy, strict=False)

        # Temporal window contract: the checkpoint's video ``delta_indices`` are
        # chronological action-step offsets with the most-recent frame at 0 —
        # e.g. [-6, -4, -2, 0] for a video_length=4 / video_stride=2 checkpoint,
        # or [0] when the memory/video window is disabled. Reading them here
        # (instead of hardcoding a consecutive TV-frame window) makes the buffer
        # sample frames at the stride the model was trained on, and keeps a
        # single-frame checkpoint working too.
        try:
            video_cfg = self._policy.get_modality_config()["video"]
            video_deltas = list(video_cfg.delta_indices)
            views = tuple(video_cfg.modality_keys)
        except Exception:  # noqa: BLE001 - preserve legacy checkpoint metadata fallback
            video_deltas = list(range(-(TV - 1), 1))  # [-3,-2,-1,0]
            views = VIEWS

        # State/action dims from the checkpoint's normalization params — the
        # same source the policy's own wire-boundary validator checks against.
        try:
            validator = self._policy.policy.validator
            state_dims = {k: int(d) for k, d in validator.expected_state_dims.items()}
            action_dims = {k: int(d) for k, d in validator.expected_action_dims.items()}
        except Exception:  # noqa: BLE001 - preserve legacy checkpoint metadata fallback
            state_dims = dict(_STATE_DIMS)
            action_dims = {}
        action_dim = sum(action_dims.values())

        base_policy = self._policy.policy
        try:
            action_horizon = int(base_policy.model.action_horizon)
        except Exception:  # noqa: BLE001 - preserve legacy checkpoint metadata fallback
            action_horizon = int(config.get("exec_horizon", 16))
        rtc_mode = str(getattr(base_policy, "rtc_inference_mode", "none"))
        rtc_timing = resolve_rtc_timing(
            action_horizon=action_horizon,
            mode=rtc_mode,
            delay=int(getattr(base_policy, "rtc_inference_delay", 0) or 0),
            exec_horizon=int(getattr(base_policy, "rtc_exec_horizon", 0) or 0),
        )
        exec_horizon = rtc_timing.exec_horizon

        if rtc_timing.enabled and action_dim < 1:
            raise ValueError("RTC requires checkpoint-derived action dimensions")
        if rtc_timing.enabled and bool(getattr(base_policy, "use_memory", False)):
            memory_stride = int(
                getattr(base_policy.model.config, "memory_stride", 0) or 0
            )
            if memory_stride != exec_horizon:
                raise ValueError(
                    f"RTC exec_horizon={exec_horizon} must match the "
                    f"checkpoint memory_stride={memory_stride}"
                )

        return RLDXSetup(
            views,
            tuple(video_deltas),
            state_dims,
            action_dims,
            rtc_timing,
            rtc_mode,
            str(getattr(embodiment, "value", embodiment)),
        )

    def generate(self, input: RLDXModelInput) -> RLDXModelResult:
        if self._policy is None:
            raise PolicyNotLoaded("RLDX policy was not loaded")
        if input.episode_id != self._episode_id:
            self.reset()
            self._episode_id = input.episode_id
        actions = None
        if input.observation is not None:
            actions, _info = self._policy.get_action(input.observation, input.options)
            self._completed_predictions += 1
        return RLDXModelResult(self._episode_id, actions, self._completed_predictions)

    def reset(self) -> None:
        """Clear native episode memory while retaining loaded weights."""
        if self._policy is not None:
            try:
                self._policy.reset()
            except Exception:  # noqa: BLE001, S110 - preserve upstream reset tolerance
                pass
        self._episode_id = None
        self._completed_predictions = 0
