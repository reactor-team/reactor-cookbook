"""Client contracts and session state for Open-Oasis."""

from __future__ import annotations

import numpy as np

from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video

KEYS = [
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "escape",
    "f",
    "q",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
MOUSE_BUTTONS = ["left", "right", "middle"]


class OpenOasisOutput(Output):
    """Stream exactly one newly generated Minecraft frame on `main_video`."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Emitted after a player-control command updates the shared input state."""

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
    camera_x: float = MessageField(
        description=(
            "Accumulated horizontal camera movement queued for the next generated frame, from "
            "-1 to 1. It returns to zero after that frame is generated."
        )
    )
    camera_y: float = MessageField(
        description=(
            "Accumulated vertical camera movement queued for the next generated frame, from -1 "
            "to 1. It returns to zero after that frame is generated."
        )
    )


class ConditioningChanged(ModelMessage):
    """Emitted after a starting-context command queues a fresh rollout."""

    source: str = MessageField(
        description=(
            "Starting-context source accepted by the command: `built_in`, `image`, or `video`."
        )
    )
    selection: str = MessageField(
        description=(
            "Built-in sample identifier or uploaded filename selected for the fresh rollout. "
            "The selection takes effect at the next generated frame."
        )
    )
    prompt_frames: int = MessageField(
        description=(
            "Number of consecutive visual frames that will initialize the fresh rollout: one "
            "for `set_image` and `random_scene`, or the accepted `prompt_frames` for `set_video`."
        )
    )


class RolloutReset(ModelMessage):
    """Emitted after `reset` queues the selected starting context to restart."""

    seed: int = MessageField(
        description=(
            "Random seed selected for the restarted rollout. It is the existing seed when "
            "`reset.seed` is -1, or the supplied non-negative value otherwise."
        )
    )
    conditioning: str = MessageField(
        description=(
            "Built-in sample identifier or uploaded filename retained for the restarted "
            "rollout, or `none` when no starting context has been selected."
        )
    )


class StateUpdate(ModelMessage):
    """Emitted when a viewer connects and after observable Open-Oasis state changes."""

    pressed_keys: list[str] = MessageField(
        description=(
            "Keyboard keys currently held for subsequent generated frames. `reset` and "
            "starting-context commands clear this list."
        )
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description=(
            "Mouse buttons currently held for subsequent generated frames. `reset` and "
            "starting-context commands clear this list."
        )
    )
    camera_x: float = MessageField(
        description=(
            "Accumulated horizontal camera movement queued for the next generated frame, from "
            "-1 to 1; zero means no queued horizontal movement."
        )
    )
    camera_y: float = MessageField(
        description=(
            "Accumulated vertical camera movement queued for the next generated frame, from -1 "
            "to 1; zero means no queued vertical movement."
        )
    )
    seed: int = MessageField(
        description=(
            "Random seed selected for the current or queued rollout. A non-negative `seed` "
            "passed to `reset` replaces it; -1 retains it."
        )
    )
    conditioning: str = MessageField(
        description=(
            "Built-in sample identifier or uploaded filename selected for the current or queued "
            "rollout, or `none` while waiting for `set_image`, `set_video`, or `random_scene`."
        )
    )


class OpenOasisState(InputState):
    """Controls shared by one autoregressive Minecraft world."""

    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _pending_key_pulses: frozenset[str] = frozenset()
    _pending_mouse_pulses: frozenset[str] = frozenset()
    _camera_x: float = 0.0
    _camera_y: float = 0.0
    _seed: int = 0
    _world_id: int = 0
    _applied_world_id: int | None = None
    _conditioning: np.ndarray | None = None
    _conditioning_name: str = "none"
