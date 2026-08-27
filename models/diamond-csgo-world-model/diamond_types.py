"""Define configuration types and the Reactor contract for the DIAMOND adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

KEYS = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
MOUSE_BUTTONS = ["left", "right"]
CONTROLLERS = ["human", "replay"]
DELTA_X_MIN = -1000.0
DELTA_X_MAX = 1000.0
DELTA_Y_MIN = -200.0
DELTA_Y_MAX = 200.0


@dataclass(frozen=True)
class AdapterConfig:
    """Hold the adapter settings read from ``diamond.yaml``."""

    repo_id: str
    revision: str
    device: str
    profile: str
    seed: int


@dataclass(frozen=True)
class PreparedScene:
    """Hold one device-ready initial condition for the next reset."""

    obs: Any
    obs_full_res: Any
    act: Any
    next_act: Any | None


class DiamondOutput(Output):
    """Stream one generated Counter-Strike frame on `main_video`."""

    main_video: Video


class ActionChanged(ModelMessage):
    """Emitted after a control command is accepted with the resulting input state."""

    controller: str = MessageField(
        description=(
            'Active action source: "human" for client controls or "replay" for the '
            "built-in scene's recorded actions."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Native keyboard keys held for forthcoming human-controlled frames; empty "
            "after controls are released or while replay is active."
        )
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description=(
            "Native mouse buttons held for forthcoming human-controlled frames; empty "
            "after controls are released or while replay is active."
        )
    )
    delta_x: float = MessageField(
        description=(
            "Horizontal relative mouse delta accepted for the next human-controlled frame, "
            "or zero when this command supplied none."
        )
    )
    delta_y: float = MessageField(
        description=(
            "Vertical relative mouse delta accepted for the next human-controlled frame, "
            "or zero when this command supplied none."
        )
    )


class SceneChanged(ModelMessage):
    """Emitted when a command queues a scene for the next model-step boundary."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description=(
            'Queued scene source: "uploaded" for `set_spawn_image` or "built_in" for '
            "`random_scene`."
        )
    )
    scene: str = MessageField(
        description="Uploaded filename or built-in scene identifier queued for the fresh world."
    )


class StateUpdate(ModelMessage):
    """Emitted when durable control state changes or a viewer connects."""

    controller: str = MessageField(
        description=(
            'Active action source: "human" for client controls or "replay" for the '
            "built-in scene's recorded actions."
        )
    )
    pressed_keys: list[str] = MessageField(
        description="Native keyboard keys held for forthcoming human-controlled frames."
    )
    pressed_mouse_buttons: list[str] = MessageField(
        description="Native mouse buttons held for forthcoming human-controlled frames."
    )

    @classmethod
    def from_state(cls, state: DiamondState) -> StateUpdate:
        """Build a client snapshot from the session's durable controls."""
        return cls(
            controller=state.controller,
            pressed_keys=[key for key in KEYS if key in state._pressed_keys],
            pressed_mouse_buttons=[
                button
                for button in MOUSE_BUTTONS
                if button in state._pressed_mouse_buttons
            ],
        )


class DiamondState(InputState):
    """Expose durable controls shared by one playable DIAMOND session."""

    controller: str = InputField(
        default="human",
        choices=CONTROLLERS,
        description=(
            'Action source used from the next model step: "human" applies client keyboard '
            'and mouse commands, while "replay" follows the built-in scene\'s recorded '
            "actions. Changing it queues a fresh world and releases held controls."
        ),
    )
    _pressed_keys: frozenset[str] = frozenset()
    _pressed_mouse_buttons: frozenset[str] = frozenset()
    _delta_x: float = 0.0
    _delta_y: float = 0.0
