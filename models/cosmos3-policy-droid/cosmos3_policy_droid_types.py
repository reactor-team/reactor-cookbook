"""The client contract of Cosmos3-Policy-DROID: tracks, state, and messages.

Everything a client can see or set is declared here. Three camera views come
in as video tracks, the robot's proprioception and the task come in as state
fields, and each predicted action chunk goes out as an ``ActionPrediction``
message on the data channel. The model produces no media.
"""

from __future__ import annotations

from reactor_runtime import (
    InputField,
    InputState,
    MediaInput,
    MessageField,
    ModelMessage,
    Video,
)


class PolicyMedia(MediaInput):
    """The three DROID camera views the client publishes, by training-time name."""

    wrist_view: Video
    exterior_view_1: Video
    exterior_view_2: Video


class PolicyState(InputState):
    """The robot's state and task, refreshed by the client each control step."""

    task_description: str = InputField(
        default="",
        max_length=300,
        description=(
            "Natural-language task for the robot. Changing it takes effect "
            "on the next predicted `ActionPrediction` chunk."
        ),
        moderate=True,
    )
    proprio_json: str = InputField(
        default="",
        max_length=8000,
        # Robot proprioception, refreshed each control step: nothing to
        # moderate, and moderating it would cost one check per step.
        moderate=False,
        description=(
            'Latest robot proprioception as JSON: `{"joint_position": '
            '[[<7 floats>], ...], "gripper_position": [[<float>], ...]}`. '
            "Each key holds a list of rows so the client can send a short "
            "history of timesteps in one update; the last row is the "
            "current state. Refreshed by the client each control step."
        ),
    )
    executed_step_json: str = InputField(
        default="",
        max_length=8000,
        # The client's echo of the chunk it executed, sent every step: not
        # user text, so it is not moderated.
        moderate=False,
        description=(
            'JSON `{"step": <int>, "action": [[...]]}` echoing the action '
            "chunk the client just executed. `step` must strictly increase "
            "each time; the next `ActionPrediction` is emitted only once it "
            "does."
        ),
    )

    # Session scratch the client never sees: the highest executed step the
    # client has echoed, and the step number of the last chunk this session
    # sent (-1 before the first). Both gate the next prediction.
    _last_executed: int = -1
    _predicted: int = -1


class ActionPrediction(ModelMessage):
    """Emitted once per predicted action chunk. `action` has shape
    [horizon][dof] — [32][8] for this checkpoint: 7 joint positions plus 1
    gripper command, in the DROID joint-position action convention. Execute
    it, then echo it back via `executed_step_json` to receive the next
    chunk."""

    action: list[list[float]] = MessageField(
        default=None, description="Predicted action chunk, shape [horizon][dof]."
    )
    step: int = MessageField(default=0, description="Monotonic prediction counter for the session.")
