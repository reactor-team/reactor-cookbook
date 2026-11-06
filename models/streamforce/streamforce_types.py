"""StreamForce client contract: the session state, the output track, and every message."""

from __future__ import annotations

from typing import Any

import numpy as np
from streamforce_model import Force

from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video

PROMPT_TOKEN_BUDGET = 512


class StreamForceOutput(Output):
    """The generated video: 480x832, 12 frames per chunk (9 in a run's first chunk)."""

    main_video: Video


class StreamForceState(InputState):
    """Session values the application owns; every client input has its own command."""

    _phase: str = "idle"  # idle | running | paused
    _first_frame: np.ndarray | None = None
    _prompt: str | None = None
    _force: Force | None = None
    _seed: int = -1  # -1 draws a random seed at start
    _run_seed: int = 0  # the seed the current run uses
    _run_id: int = 0  # bumped by every start
    _applied_run_id: int | None = None  # the run the model reported holding
    _applied_seed: int | None = None
    _applied_prompt: str | None = None
    _applied_force: Force | None = None
    _prompt_from_chunk: int | None = None  # first chunk conditioned on the applied prompt
    _force_from_chunk: int | None = None  # first chunk built with the applied force
    _last_chunk: int = -1
    _frames_in_run: int = 0


class StateUpdate(ModelMessage):
    """Emitted on connect, as the answer to every command, and whenever an applied value changes."""

    state: str = MessageField(description="`idle`, `running` or `paused`.")
    settings: dict = MessageField(
        description=(
            "Per input, the value last sent (`sent`) and the value in use at the last chunk "
            "(`applied`): `first_frame` (bool), `prompt`, `force`, `seed`. `prompt` and `force` "
            "also give `from_chunk`, the first chunk generated with the applied value."
        )
    )
    capabilities: list = MessageField(
        description="Every input command with its kind (session setup or live control) and accepted values."
    )
    last_chunk: int = MessageField(description="Index of the last chunk sent in this run, -1 before the first.")
    frames_per_session: int = MessageField(description="Frames a run produces before it ends.")


class ChunkComplete(ModelMessage):
    """Emitted after every chunk of video, before its frames."""

    index: int = MessageField(description="Chunk index in the run: 0 is the first 9 frames, then 12 frames per chunk.")
    frames: int = MessageField(description="Frames in this chunk.")


class GenerationStarted(ModelMessage):
    """Emitted when `start` begins a run."""

    seed: int = MessageField(description="The seed this run uses.")


class GenerationComplete(ModelMessage):
    """Emitted when a run reaches the model's maximum length and ends on its own."""

    reason: str = MessageField(description="Why the run ended.")
    frames: int = MessageField(description="Frames the run produced.")


CAPABILITIES: list[dict[str, Any]] = [
    {
        "command": "set_first_frame",
        "kind": "session setup",
        "modality": "image",
        "role": "first frame",
        "values": "PNG, JPEG or WebP of any size; resized to cover 832x480 and centre-cropped",
    },
    {
        "command": "set_prompt",
        "kind": "live control",
        "modality": "text",
        "role": "scene description",
        "values": f"any text up to {PROMPT_TOKEN_BUDGET} tokens; an empty prompt is a valid value",
    },
    {
        "command": "set_force",
        "kind": "live control",
        "modality": "force",
        "role": "force acting on the scene",
        "values": {
            "mode": ["point", "wind"],
            "strength": "0..1 of the strongest force in training",
            "angle": "degrees counter-clockwise from +x (0 right, 90 up)",
            "x, y": "point only: 0..1 over the output frame, origin top-left",
        },
    },
    {
        "command": "set_seed",
        "kind": "session setup",
        "modality": "number",
        "role": "randomness",
        "values": "integer 0..2^32-1, or -1 for random",
    },
]


def force_dict(force: Force | None) -> dict[str, Any] | None:
    """Return a force as the plain mapping `state_update` carries."""
    if force is None:
        return None
    value: dict[str, Any] = {"mode": force.mode, "strength": force.strength, "angle": force.angle}
    if force.mode == "point":
        value.update(x=force.x, y=force.y)
    return value
