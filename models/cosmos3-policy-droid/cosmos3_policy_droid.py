"""The application half of Cosmos3-Policy-DROID: what a robot client sees and drives.

This is the ``ReactorApp`` the runtime drives one step at a time. It declares
the client contract (``cosmos3_policy_droid_types.py``), decides in
``process_input()`` whether a prediction is due and what the policy gets, and
sends the predicted chunk in ``process_output()``. The policy itself lives in
``cosmos3_policy_droid_model.py``.

One step is one prediction. A prediction is due when every camera view has
delivered at least one frame, the client has sent valid proprioception, and,
after the first chunk, the client has echoed the executed step with a
strictly larger number than the last echo. That echo is the flow control
between the policy and the robot's control loop: a client still executing a
chunk is not run ahead of, and an echo repeated by a stalled loop cannot
trigger a second prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reactor_runtime import (
    ApplicationError,
    ReactorApp,
    StepOutcome,
    event,
    get_logger,
    get_weights_path,
    session_ended,
    session_started,
)

from cosmos3_policy_droid_model import VIEWS, Cosmos3PolicyModel, PolicyInput, PolicyResult
from cosmos3_policy_droid_types import ActionPrediction, PolicyMedia, PolicyState

logger = get_logger(__name__)
metrics = get_logger("reactor.metrics")

LOG_EVERY_N_PREDICTIONS = 50


class Cosmos3PolicyDroid(ReactorApp):
    """Cosmos3-Policy-DROID. Video+proprio-in -> action-out (data channel)."""

    media: PolicyMedia
    state: PolicyState

    def __init__(self) -> None:
        super().__init__()
        self._engine: Cosmos3PolicyModel | None = None
        # The newest frame per camera view. Cameras deliver asynchronously and
        # a step reads each track once, so one step rarely carries every view;
        # a prediction is built from the retained set.
        self._frames: dict[str, np.ndarray] = {}
        self._predictions_since_log = 0
        self._elapsed_since_log = 0.0

    def load(self, config_path: Path | None) -> None:
        """Construct and load the policy once per process."""
        self._engine = Cosmos3PolicyModel()
        self._engine.load(config_path, get_weights_path())

    # -- lifecycle -------------------------------------------------------------

    @session_started
    def on_session_started(self) -> None:
        """Start a session with no retained frames; the gates start closed."""
        self._frames = {}

    @session_ended
    def on_session_ended(self) -> None:
        """Release the session's frames and return the policy to its default state."""
        self._frames = {}
        if self._engine is not None:
            self._engine.reset()

    @event(
        name="reset",
        description="Reset the session's flow-control step counter.",
    )
    async def reset(self) -> None:
        # Rebuild the session's gate: the next chunk is step 0 again and is
        # sent as soon as every view has delivered a fresh frame and proprio
        # is valid, with no echo required.
        self._frames = {}
        self.state._last_executed = -1
        self.state._predicted = -1

    # -- the step --------------------------------------------------------------

    async def process_input(self) -> PolicyInput:
        """Refuse until a prediction is due; otherwise say what the policy gets."""
        for view in VIEWS:
            frames = getattr(self.media, view).try_read(1)
            if frames:
                self._frames[view] = frames[0].data
        if len(self._frames) < len(VIEWS):
            raise ApplicationError("waiting for a frame on every camera view")

        proprio = parse_proprio(self.state.proprio_json)
        if proprio is None:
            raise ApplicationError("waiting for valid proprio_json")
        joint_position, gripper_position = proprio

        if self.state._predicted >= 0:
            executed = parse_executed_step(self.state.executed_step_json)
            if executed is None or executed <= self.state._last_executed:
                raise ApplicationError("waiting for executed_step_json to advance")
            self.state._last_executed = executed

        return PolicyInput(
            wrist_view=self._frames["wrist_view"],
            exterior_view_1=self._frames["exterior_view_1"],
            exterior_view_2=self._frames["exterior_view_2"],
            joint_position=joint_position,
            gripper_position=gripper_position,
            task=self.state.task_description,
        )

    def generate(self, input: PolicyInput) -> PolicyResult:
        """One prediction. The model half does the work."""
        if self._engine is None:
            raise RuntimeError("Cosmos3-Policy-DROID was not loaded")
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> None:
        """Send the predicted chunk as the session's next step. This model emits no media."""
        if outcome.error is not None:
            # process_input refuses every step the policy cannot serve, so an
            # error here is a GPU or checkpoint failure a reset would not
            # repair; it ends the session loudly.
            raise outcome.error
        result: PolicyResult = outcome.result
        self.state._predicted += 1
        self._record_metrics(outcome.elapsed)
        await self.send(ActionPrediction(action=result.actions.tolist(), step=self.state._predicted))
        return None

    def _record_metrics(self, elapsed: float) -> None:
        self._predictions_since_log += 1
        self._elapsed_since_log += elapsed
        if self._predictions_since_log >= LOG_EVERY_N_PREDICTIONS:
            metrics.info(
                "prediction_metrics",
                predictions=self._predictions_since_log,
                mean_ms=self._elapsed_since_log / self._predictions_since_log * 1000.0,
                model="cosmos3-policy-droid",
            )
            self._predictions_since_log = 0
            self._elapsed_since_log = 0.0


def parse_proprio(raw: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse the client's proprioception into ``([N, 7], [N, 1])`` float32 arrays.

    Anything short of the contract (empty, malformed JSON, a missing key, a
    wrong row width) reads as "no proprio yet". A fabricated state would
    command a real arm, so nothing is zero-filled.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
        joint_position = np.asarray(data["joint_position"], dtype=np.float32)
        gripper_position = np.asarray(data["gripper_position"], dtype=np.float32)
    except (ValueError, TypeError, KeyError):
        return None
    if joint_position.ndim != 2 or joint_position.shape[1] != 7:
        return None
    if gripper_position.ndim != 2 or gripper_position.shape[1] != 1:
        return None
    if not (np.isfinite(joint_position).all() and np.isfinite(gripper_position).all()):
        return None
    return joint_position, gripper_position


def parse_executed_step(raw: str) -> int | None:
    """Read ``step`` from the client's executed-step echo; ``None`` when absent or malformed."""
    if not raw:
        return None
    try:
        return int(json.loads(raw)["step"])
    except (ValueError, TypeError, KeyError):
        return None
