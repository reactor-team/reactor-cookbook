"""Execute first-action scripts with real SDK/helpers and mocked network I/O."""

import contextlib
import io
import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from reactor_sdk import Reactor, Track, TrackDirection, TrackKind

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cosmos": (
        "cosmos-nano-policy-droid",
        ("wrist_view", "exterior_view_1", "exterior_view_2"),
        "set_proprio_json",
        "action",
        (32, 8),
    ),
    "flux": (
        "flux3-action-droid",
        ("wrist_view", "exterior_view_1", "exterior_view_2"),
        "set_state_json",
        "actions",
        (32, 8),
    ),
    "xwam": (
        "xwam",
        ("head_view", "left_wrist_view", "right_wrist_view"),
        "set_state_json",
        "actions",
        (32, 14),
    ),
    "lingbot": (
        "lingbot-va",
        ("agentview", "eye_in_hand"),
        "set_task_description",
        "action",
        (16, 7),
    ),
    "dreamzero_droid": (
        "dreamzero",
        ("exterior_1", "exterior_2", "wrist"),
        "set_prompt",
        "actions",
        (24, 8),
    ),
    "dreamzero_yam": (
        "dreamzero-yam-molmoact2",
        ("top", "left", "right"),
        "set_prompt",
        "actions",
        (24, 14),
    ),
    "xr1": (
        "xr1",
        ("ego_view", "wrist_left_view", "wrist_right_view"),
        "set_proprio_json",
        "action",
        (30, 60),
    ),
    "xr1_robocasa": (
        "xr1-robocasa365",
        ("left_agentview", "right_agentview", "wrist_view"),
        "set_executed_step_json",
        "action",
        (16, 60),
    ),
    "fastwam": (
        "fastwam-libero",
        ("exterior_view_1", "wrist_view"),
        "set_state_json",
        "actions",
        (32, 7),
    ),
}

# Praxis-1 sends its first chunk once it has robot state, and every later
# chunk after set_executed_step: (model, tracks, {field: row width}).
PRAXIS_CASES = {
    "praxis_droid": (
        "praxis-1-droid",
        ("exterior", "wrist"),
        {"poses": 12, "grips": 0},
    ),
    "praxis_yam": (
        "praxis-1-yam",
        ("top", "wrist_left", "wrist_right"),
        {"left_poses": 12, "left_grips": 0, "right_poses": 12, "right_grips": 0},
    ),
}
PRAXIS_TRIGGER = "set_robot_state_json"


def praxis_chunk(fields, step, failure):
    rows = 21 if step == 0 else 24
    data = {"step": float(step), "timestamps": [i / 15 for i in range(rows)]}
    for name, width in fields.items():
        arr = np.zeros((rows, width) if width else (rows,))
        if failure == "shape":
            arr = arr[:1]
        if failure == "nan":
            arr.flat[0] = np.nan
        data[name] = arr.tolist()
    return {"type": "action_prediction", "data": data}


class FirstActionsTests(unittest.TestCase):
    def exercise(self, name, failure=None):
        praxis = name in PRAXIS_CASES
        if praxis:
            model, tracks, fields = PRAXIS_CASES[name]
            trigger, field, shape = PRAXIS_TRIGGER, None, None
            pending = {"step": 0, "emitted": False}
        else:
            model, tracks, trigger, field, shape = CASES[name]
        sdk = Reactor("reactor/" + model, api_key="offline-test")
        sdk._require_handle = Mock()
        sdk._async_op = AsyncMock(return_value=None)
        sdk._push_video_frame = Mock()
        sdk._tracks = {
            n: Track(sdk, n, TrackKind.VIDEO, TrackDirection.SENDONLY) for n in tracks
        }
        sdk.connect = AsyncMock(
            side_effect=RuntimeError("connect failed") if failure == "connect" else None
        )
        sdk.disconnect = AsyncMock()
        sdk.close = Mock(wraps=sdk.close)
        sent = []

        async def command(command, payload):
            sent.append((command, payload))
            if failure == "command":
                raise RuntimeError("command failed")
            envelope = None
            if praxis:
                if command == "set_executed_step":
                    pending.update(step=payload["executed_step"] + 1, emitted=False)
                if command in (trigger, "set_executed_step") and not pending["emitted"]:
                    pending["emitted"] = True
                    envelope = praxis_chunk(fields, pending["step"], failure)
            elif command in ("get_checkpoint", "select_checkpoint"):
                envelope = {
                    "type": "checkpoint_selected",
                    "data": {
                        "checkpoint": "base-bf16",
                        "available": ["base-bf16"],
                        "locked": command == "select_checkpoint",
                    },
                }
            elif command == trigger:
                actions = np.zeros(shape)
                if failure == "shape":
                    actions = actions[:1]
                if failure == "nan":
                    actions[0, 0] = np.nan
                step = (
                    json.loads(payload["state_json"])["chunk_id"]
                    if "state_json" in payload
                    else 0
                )
                data = {
                    field: actions.tolist(),
                    "step": step,
                    "chunk_index": 0,
                    "obs_seq": 1,
                    "inference_seconds": 0.1,
                    "prefix_rows": 0,
                    "checkpoint": "base-bf16",
                    "proprios": np.zeros((9, 16)).tolist(),
                }
                envelope = {
                    "type": "action_chunk"
                    if name.startswith("dreamzero")
                    else "action_prediction",
                    "data": data,
                }
            if envelope:
                for handler in sdk._handlers["message"]:
                    handler(envelope)
            return envelope

        sdk.send_command = command
        error = (
            (AssertionError, ValueError)
            if failure in ("nan", "shape")
            else RuntimeError
        )
        try:
            with (
                patch("reactor_sdk.Reactor", return_value=sdk),
                patch.dict("os.environ", REACTOR_API_KEY="offline-test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if failure:
                    with self.assertRaises(error):
                        runpy.run_path(
                            str(ROOT / f"first_{name}_actions.py"), run_name="__main__"
                        )
                else:
                    runpy.run_path(
                        str(ROOT / f"first_{name}_actions.py"), run_name="__main__"
                    )
            sdk.disconnect.assert_awaited_once()
            sdk.close.assert_called_once()
            if not failure:
                self.assertTrue(any(cmd == trigger for cmd, _ in sent))
            return sent
        finally:
            sdk.close()

    def test_first_actions_and_failure_cleanup(self):
        for name in [*CASES, *PRAXIS_CASES]:
            for failure in (None, "shape", "nan", "connect", "command"):
                with self.subTest(model=name, failure=failure):
                    self.exercise(name, failure)


if __name__ == "__main__":
    unittest.main()
