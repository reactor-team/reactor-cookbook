"""Define OpenDreamer's configuration and public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from reactor_runtime import (
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

DEMO_CHOICES = ["demo_1", "demo_2", "demo_3"]


@dataclass(frozen=True)
class DemoConfig:
    """Describe a dataset window available as a starting scene."""

    name: str
    video: Path
    actions: Path
    start_frame: int


@dataclass(frozen=True)
class OpenDreamerConfig:
    """Hold validated model, checkpoint, and conditioning settings."""

    source_revision: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    platform: str
    seed: int
    num_steps: int
    tau_ctx_target: float
    conditioning_frames: int
    demos: tuple[DemoConfig, ...]
    warmup_steps: int
    memory_fraction: float


@dataclass(frozen=True)
class RolloutConditioning:
    """Pair consecutive Minecraft frames with their aligned player actions."""

    frames: np.ndarray
    actions: Any


class OpenDreamerOutput(Output):
    """Stream the next generated Minecraft frame on `main_video`."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Emitted after a player-control command is processed."""

    control: str = MessageField(
        description=(
            "Wire name of the command that produced this response. Use it to associate the "
            "snapshot with `set_key_state`, `set_mouse_button_state`, `mouse_move`, or "
            "`mouse_wheel`."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Keyboard keys held after the command is processed. Each key applies to every "
            "generated frame until `set_key_state` releases it or another command clears "
            "controls."
        )
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description=(
            "Mouse buttons held after the command is processed. Each button applies to every "
            "generated frame until `set_mouse_button_state` releases it or another command "
            "clears controls."
        )
    )
    delta_x: float = MessageField(
        description=(
            "Horizontal movement accepted from this `mouse_move` command for the next "
            "generated frame. Zero for other commands."
        )
    )
    delta_y: float = MessageField(
        description=(
            "Vertical movement accepted from this `mouse_move` command for the next generated "
            "frame. Zero for other commands."
        )
    )
    wheel_delta: int = MessageField(
        description=(
            "Hotbar movement accepted from this `mouse_wheel` command for the next generated "
            "frame. Zero for other commands."
        )
    )


class ConditioningChanged(ModelMessage):
    """Emitted after a starting-scene command selects the next rollout input."""

    source: str = MessageField(
        description=(
            "Source selected for the next rollout: `demo` for a configured dataset sample or "
            "`upload` for an image accepted by `set_conditioning_image`."
        )
    )
    selection: str = MessageField(
        description=(
            "Configured demo name or uploaded filename selected for the next rollout. The "
            "selection takes effect at the next inference boundary."
        )
    )


class RolloutReset(ModelMessage):
    """Emitted after `reset` schedules the selected starting scene to restart."""

    seed: int = MessageField(
        description=(
            "Random seed selected for the restarted rollout. It is the existing seed when "
            "the `seed` value passed to `reset` is `-1`, or the supplied non-negative value "
            "otherwise."
        )
    )
    conditioning: str = MessageField(
        description=(
            "Starting scene retained by `reset`: a configured demo name or `uploaded` for the "
            "most recently accepted conditioning image."
        )
    )


class StateUpdate(ModelMessage):
    """Emitted when a viewer connects and after observable OpenDreamer state changes."""

    pressed_keys: list[str] = MessageField(
        description=(
            "Keyboard keys currently held for subsequent generated frames. "
            "`reset` and starting-scene commands clear this list."
        )
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description=(
            "Mouse buttons currently held for subsequent generated frames. "
            "`reset` and starting-scene commands clear this list."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed selected for the current or queued rollout. A non-negative "
            "`seed` value passed to `reset` replaces it; `-1` retains it."
        )
    )
    conditioning: str = MessageField(
        description=(
            "Starting scene selected for the current or queued rollout: a configured demo name "
            "or `uploaded` for the most recently accepted conditioning image."
        )
    )

    @classmethod
    def from_state(
        cls,
        state: OpenDreamerState,
        *,
        conditioning: str,
    ) -> StateUpdate:
        """Build a complete client-facing snapshot from the shared world state."""
        return cls(
            pressed_keys=sorted(state._pressed_keys),
            pressed_mouse_buttons=sorted(state._pressed_mouse_buttons),
            seed=state._seed,
            conditioning=conditioning,
        )


class OpenDreamerState(InputState):
    """Expose the controls shared by one playable OpenDreamer world."""

    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _delta_x: float = 0.0
    _delta_y: float = 0.0
    _wheel_delta: int = 0
    _reset_requested: bool = True
    _seed: int = 0
