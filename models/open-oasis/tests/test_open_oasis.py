"""Contract, native actions, uploads, and one-frame boundaries for Open-Oasis."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import FrozenInstanceError
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

from open_oasis import OpenOasis
from open_oasis_model import OpenOasisModel
from open_oasis_images import decode_image
from open_oasis_types import OpenOasisState


def ready_model() -> OpenOasis:
    model = OpenOasis()
    model.state = OpenOasisState()
    model._config = SimpleNamespace(seed=0)
    model.engine = OpenOasisModel()
    model.state._conditioning = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    model.send = lambda _message: _awaitable()  # type: ignore[method-assign]
    model.output = SimpleNamespace(flush=lambda: None)
    return model


async def _awaitable() -> None:
    return None


def test_schema_is_complete_and_precise() -> None:
    contract = ModelContract.of(OpenOasis)
    assert set(contract.commands) == {
        "mouse_move",
        "random_scene",
        "release_controls",
        "reset",
        "set_image",
        "set_video",
        "set_key_state",
        "set_mouse_button_state",
    }
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )
    document = contract.render_schema().to_openapi()
    assert document["x-reactor"]["tracks"] == [
        {"name": "main_video", "kind": "video", "direction": "out"}
    ]
    assert set(document["webhooks"]) == {
        "action_changed",
        "conditioning_changed",
        "rollout_reset",
        "state_update",
    }
    assert (
        document["paths"]["/events/set_video"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["properties"]["prompt_frames"]["maximum"]
        == 32
    )


def test_native_actions_cover_all_twenty_five_dimensions() -> None:
    model = ready_model()
    model.state._pressed_keys = frozenset({"w", "space", "e", "9"})
    model.state._pressed_mouse_buttons = frozenset({"left", "right", "middle"})
    model.state._camera_x = -0.5
    model.state._camera_y = 0.75
    action = model._build_action()
    assert action.shape == (25,)
    assert np.count_nonzero(action) == 9
    assert action[15] == -0.5 and action[16] == 0.75


def test_camera_accumulates_then_release_clears_everything() -> None:
    model = ready_model()
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.mouse_move(0.75, -0.8))
    reply = asyncio.run(model.mouse_move(0.75, -0.8))
    assert reply.camera_x == 1 and reply.camera_y == -1
    released = asyncio.run(model.release_controls())
    assert released.pressed_keys == [] and released.camera_x == 0


def test_uploaded_image_is_rgb_native_resolution() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (40, 20), (2, 4, 8, 128)).save(buffer, format="PNG")
    frame = decode_image(buffer.getvalue())
    assert frame.shape == (1, 360, 640, 3)
    assert frame.dtype == np.uint8


def test_conditioning_upload_is_moderated() -> None:
    fields = ModelContract.of(OpenOasis).commands
    assert fields["set_image"].command.__command_fields__["image"].info.moderate
    assert fields["set_video"].command.__command_fields__["video"].info.moderate


def test_viewer_disconnect_preserves_shared_world() -> None:
    model = ready_model()
    model.on_session_started()
    assert model.state._conditioning is None
    assert model.state._conditioning_name == "none"

    model.state._conditioning = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    model.state._conditioning_name = "uploaded.png"
    model.engine = Mock()
    asyncio.run(model.on_disconnected())

    assert model.state._conditioning is not None
    assert model.state._conditioning_name == "uploaded.png"
    model.engine.reset.assert_not_called()
    assert asyncio.run(model.process_input()).conditioning is model.state._conditioning


def test_ten_continuous_steps_and_anchor_handshake() -> None:
    class Backend:
        resets = 0
        calls = 0

        def reset(self, frames, seed):
            self.resets += 1

        def generate_one(self, action):
            self.calls += 1
            return np.zeros((8, 8, 3), dtype=np.uint8)

    model = ready_model()
    backend = Backend()
    model.engine._backend = backend

    async def run():
        for index in range(11):
            input = await model.process_input()
            assert (input.conditioning is not None) == (index == 0)
            result = model.generate(input)
            assert result.index == index
            assert (
                await model.process_output(StepOutcome(result=result, elapsed=0.1))
                is not None
            )

    asyncio.run(run())
    assert (backend.resets, backend.calls) == (1, 10)


def test_generate_forwards_only_input_and_error_does_not_complete() -> None:
    model = ready_model()
    input = asyncio.run(model.process_input())
    model.engine = SimpleNamespace(generate=lambda value: value)
    model.state = None
    assert model.generate(input) is input
    model.state = OpenOasisState()

    def fail(value):
        raise ValueError("sampler failed")

    model.engine = SimpleNamespace(generate=fail)
    with pytest.raises(ValueError) as failure:
        model.generate(input)
    with pytest.raises(ValueError, match="sampler failed"):
        asyncio.run(model.process_output(StepOutcome(error=failure.value)))
    assert model.state._applied_world_id is None


def test_playback_contract_uses_one_frame_chunks_without_explicit_fps() -> None:
    assert "fps" not in OpenOasis.__dict__
    assert OpenOasis.buffer_size == 1


def test_press_and_release_before_sampling_still_produces_one_frame_pulse() -> None:
    model = ready_model()
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.set_key_state("w", False))

    assert model.state._pressed_keys == frozenset()
    assert model._build_action()[11] == 1

    model.state._pending_key_pulses = frozenset()
    assert model._build_action()[11] == 0


def test_refusal_never_invokes_model_and_input_is_frozen() -> None:
    model = ready_model()
    model.state._conditioning = None
    model.engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.engine.generate.assert_not_called()
    model.state._conditioning = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    step = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        step.world_id = 99
    assert asyncio.run(model.process_input()).conditioning is step.conditioning


def test_native_failure_preserves_completed_index() -> None:
    model = ready_model()
    backend = Mock()
    backend.generate_one.return_value = np.zeros((8, 8, 3), np.uint8)
    model.engine._backend = backend
    for _ in range(2):
        result = model.generate(asyncio.run(model.process_input()))
        asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.1)))
    assert result.index == 1
    model.send = Mock()
    backend.generate_one.side_effect = RuntimeError("sampler failure")
    with pytest.raises(RuntimeError) as error:
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError, match="sampler failure"):
        asyncio.run(model.process_output(StepOutcome(error=error.value)))
    model.send.assert_not_called()
    backend.generate_one.side_effect = None
    result = model.generate(asyncio.run(model.process_input()))
    assert result.index == 2


def test_failed_step_retains_pulses_and_camera_until_success():
    from unittest.mock import AsyncMock
    from open_oasis_model import OpenOasisResult

    model = ready_model()
    model.state._applied_world_id = model.state._world_id
    model.send = AsyncMock()
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.set_key_state("w", False))
    asyncio.run(model.mouse_move(0.5, -0.5))
    first = asyncio.run(model.process_input())
    assert first.action[11] == 1
    with pytest.raises(RuntimeError):
        asyncio.run(model.process_output(StepOutcome(error=RuntimeError("failed"))))
    repeated = asyncio.run(model.process_input())
    np.testing.assert_array_equal(first.action, repeated.action)
    result = OpenOasisResult(first.world_id, np.zeros((360, 640, 3), np.uint8), 1)
    asyncio.run(model.process_output(StepOutcome(result=result)))
    assert not model.state._pending_key_pulses
    assert not model.state._camera_x and not model.state._camera_y
    assert model.send.call_args.args[0].camera_x == 0


def test_video_decode_stops_at_last_requested_frame(monkeypatch):
    from open_oasis_images import decode_video
    import open_oasis_images

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, video):
            for index in range(5):
                yield SimpleNamespace(
                    to_ndarray=lambda format, i=index: np.full(
                        (360, 640, 3), i, np.uint8
                    )
                )
            raise AssertionError("Decoder read beyond offset + count")

    monkeypatch.setattr(open_oasis_images.av, "open", lambda _: Container())
    frames = decode_video(b"fixture", 2, 3)
    assert frames.shape == (3, 360, 640, 3)
    assert frames[:, 0, 0, 0].tolist() == [2, 3, 4]


def test_invalid_video_reports_value_error():
    from open_oasis_images import decode_video

    with pytest.raises(ValueError, match="decoded"):
        decode_video(b"not a video", 0, 1)


def test_model_dependency_graph_does_not_import_runtime():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import open_oasis_model, open_oasis_backend
open_oasis_model.OpenOasisModel()
""",
        ],
        cwd=MODEL_DIR,
        check=True,
    )
