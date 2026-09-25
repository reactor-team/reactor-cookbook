from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from matrix_game_3_0 import MatrixGame30
from matrix_game_3_0_assets import read_config
from matrix_game_3_0_backend import MatrixGame30Backend, action_from_controls
from matrix_game_3_0_model import (
    MatrixGame30Anchor,
    MatrixGame30Input,
    MatrixGame30Model,
    MatrixGame30Result,
    NoAnchor,
    RolloutExhausted,
    normalize_output_frames,
)
from matrix_game_3_0_types import MatrixGame30State
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_native_control_mapping_preserves_discrete_keys_and_continuous_camera() -> None:
    action = action_from_controls(frozenset(("w", "a")), -0.5, 1.0)

    assert action.keyboard == (1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert action.mouse == (-0.05, 0.1)


def test_native_chunk_frame_counts_are_enforced() -> None:
    first = np.zeros((57, 8, 8, 3), dtype=np.uint8)
    later = np.zeros((40, 8, 8, 3), dtype=np.uint8)

    assert normalize_output_frames(first, 0).shape[0] == 57
    assert normalize_output_frames(later, 1).shape[0] == 40


def test_session_waits_for_explicit_image_selection() -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State()
    model._config = SimpleNamespace(seed=42, max_chunks=12)

    class Engine:
        def generate(self, _input: object) -> np.ndarray:
            raise AssertionError("generation must wait for image selection")

    model._engine = Engine()

    model.on_session_started()
    message = model._state_update()

    assert model._selected_input is None
    assert message.restart_queued is False
    assert message.image_source == "none"
    assert message.next_chunk is None
    assert message.next_chunk_frames is None
    with pytest.raises(ApplicationError, match="anchor image"):
        asyncio.run(model.process_input())


def test_uploaded_image_and_prompt_start_a_fresh_rollout(tmp_path: Path) -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="original")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = tmp_path / "original.png"

    message = model.set_image(
        UploadedFile(name="anchor.png", mime_type="image/png", data=_png_bytes()),
        "replacement",
    )

    assert message.restart_queued is True
    assert message.prompt == "replacement"
    assert message.image_source == "uploaded"
    assert message.next_chunk_frames == 57


def test_uploaded_image_without_prompt_starts_a_fresh_rollout(
    tmp_path: Path,
) -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="original")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = tmp_path / "original.png"

    message = model.set_image(
        UploadedFile(name="anchor.png", mime_type="image/png", data=_png_bytes()),
        "",
    )

    assert message.restart_queued is True
    assert message.prompt == ""
    assert message.image_source == "uploaded"
    assert message.next_chunk_frames == 57


def test_control_events_hold_discrete_keys_and_continuous_camera() -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="scene")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = Path("anchor.png")
    model.send = AsyncMock()

    async def apply_controls() -> None:
        key_message = await model.set_key_state("w", True)
        pitch_message = await model.set_pitch(0.5)
        yaw_message = await model.set_yaw(-1.0)
        assert key_message.pressed_keys == ["w"]
        assert pitch_message.pitch == 0.5
        assert yaw_message.yaw == -1.0

    asyncio.run(apply_controls())

    action = action_from_controls(
        model.state._pressed_keys,
        model.state.pitch,
        model.state.yaw,
    )
    assert action.keyboard == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert action.mouse == (0.05, -0.1)
    assert model.send.await_count == 3


def test_backend_bridges_one_action_to_each_unmodified_iteration(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    original_action = object()
    original_video = object()
    module = SimpleNamespace(
        get_current_action=original_action,
        process_video=original_video,
    )
    seen_actions: list[dict[str, object]] = []

    class FakePipeline:
        def generate(self, *_args: object, **_kwargs: object) -> None:
            for index, frame_count in enumerate((57, 40)):
                seen_actions.append(module.get_current_action())
                module.process_video(
                    np.full((frame_count, 4, 6, 3), index, dtype=np.uint8),
                    str(tmp_path / f"reactor_current_iteration_{index}.mp4"),
                    None,
                    None,
                )

    config = SimpleNamespace(
        chunk_timeout_seconds=10.0,
        size="704*1280",
        sample_shift=5.0,
        num_inference_steps=3,
        guide_scale=5.0,
    )
    backend = MatrixGame30Backend(config)
    backend._module = module
    backend._pipeline = FakePipeline()
    backend._args = SimpleNamespace()

    image = tmp_path / "anchor.png"
    Image.new("RGB", (16, 16)).save(image)
    backend.reset("scene", 42, image)
    first = backend.generate_chunk(action_from_controls(frozenset(("w",)), 0.0, 0.0))
    second = backend.generate_chunk(action_from_controls(frozenset(("d",)), 0.0, -1.0))
    backend.end_session()

    assert first.shape == (57, 4, 6, 3)
    assert second.shape == (40, 4, 6, 3)
    assert len(seen_actions) == 2
    assert module.get_current_action is original_action
    assert module.process_video is original_video


class FakeModel:
    def __init__(self, max_chunks=12):
        self.inputs = []
        self.world_id = None
        self.chunk = 0
        self.max_chunks = max_chunks
        self.resets = 0

    def generate(self, input):
        self.inputs.append(input)
        if input.world_id != self.world_id:
            assert input.anchor is not None
            self.world_id = input.world_id
            self.chunk = 0
        self.chunk += 1
        return MatrixGame30Result(
            np.zeros((57 if self.chunk == 1 else 40, 8, 8, 3), dtype=np.uint8),
            self.world_id,
            self.chunk,
            self.chunk >= self.max_chunks,
        )

    def reset(self):
        self.resets += 1
        self.world_id = None
        self.chunk = 0


def make_model(max_chunks=12):
    model = MatrixGame30()
    model.state = MatrixGame30State()
    model._config = SimpleNamespace(
        seed=42,
        max_chunks=max_chunks,
    )
    model.send = AsyncMock()
    model._selected_input = Path("anchor.png")
    model._engine = FakeModel(max_chunks)
    model.output = Mock()
    model.state._world_id = 1
    return model


def test_input_waits_for_image_and_rejects_rollout_limit():
    model = make_model()
    model._selected_input = None
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert model._engine.inputs == []
    model._selected_input = Path("anchor.png")
    model.state._limit_reached = True
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())


def test_failure_does_not_advance_progress():
    model = make_model()
    failure = RuntimeError("failed")
    model._engine.generate = Mock(side_effect=failure)
    with pytest.raises(RuntimeError, match="failed"):
        model.generate(asyncio.run(model.process_input()))
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(model.process_output(StepOutcome(error=failure)))
    assert model._chunk_index == 0
    assert model.state._applied_world_id is None
    assert model._state_update().restart_queued
    model.send.assert_not_called()


def test_controls_are_snapshotted_and_native_chunks_remain_continuous():
    model = make_model()
    model.state.yaw = 0.25
    snapshot = asyncio.run(model.process_input())
    model.state.yaw = -0.25
    assert snapshot.action.mouse[1] == 0.025
    result = model.generate(snapshot)
    output = asyncio.run(model.process_output(StepOutcome(result=result)))
    assert output.main_video.shape[0] == 57
    assert model._chunk_index == 1
    assert model.send.called
    for index in range(9):
        model.state.pitch = (index - 4) / 10
        model.state._pressed_keys = frozenset(("wasd"[index % 4],))
        snapshot = asyncio.run(model.process_input())
        assert snapshot.anchor is None
        result = model.generate(snapshot)
        output = asyncio.run(model.process_output(StepOutcome(result=result)))
        assert output.main_video.shape[0] == 40
    assert model._chunk_index == 10
    assert len(model._engine.inputs) == 10
    assert sum(input.anchor is not None for input in model._engine.inputs) == 1
    assert len({input.world_id for input in model._engine.inputs}) == 1


def test_generate_only_forwards_frozen_input():
    app = make_model()
    input = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        input.world_id = 99
    app.state = None
    app._config = None
    app._selected_input = None
    result = app.generate(input)
    assert result.world_id == input.world_id
    assert app._engine.inputs == [input]
    assert not hasattr(result, "input")


async def step(app):
    return await app.process_output(
        StepOutcome(result=app.generate(await app.process_input()))
    )


def test_reset_prompt_and_upload_explicitly_start_new_worlds():
    app = make_model()
    asyncio.run(step(app))
    first_id = app.state._world_id
    app.reset(seed=123)
    input = asyncio.run(app.process_input())
    assert input.world_id == first_id + 1
    assert input.anchor.seed == 123
    asyncio.run(step(app))
    app.set_prompt("A snowy forest")
    input = asyncio.run(app.process_input())
    assert input.world_id == first_id + 2
    assert input.anchor.prompt == "A snowy forest"
    asyncio.run(step(app))
    data = _png_bytes()
    app.set_image(UploadedFile(name="new.png", mime_type="image/png", data=data), "")
    input = asyncio.run(app.process_input())
    assert input.world_id == first_id + 3
    assert input.anchor.image == data
    assert (
        input.anchor.prompt == ""
    )  # Image-only generation is a native supported input.
    asyncio.run(step(app))
    assert app._chunk_index == 1
    assert app.output.flush.call_count == 3
    with pytest.raises(CommandError):
        app.set_prompt(" ")
    assert app.state._world_id == first_id + 3


def test_limit_emits_last_chunk_and_waits_for_client_reset():
    app = make_model(max_chunks=2)
    asyncio.run(step(app))
    app.state._pressed_keys = frozenset(("w",))
    result = asyncio.run(step(app))
    assert result.main_video.shape[0] == 40
    assert app.state._limit_reached
    assert not app.state._pressed_keys
    messages = [call.args[0] for call in app.send.await_args_list[-2:]]
    assert [type(message).__name__ for message in messages] == [
        "RolloutLimitReached",
        "StateUpdate",
    ]
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app.reset(seed=-1)
    asyncio.run(step(app))
    assert app._chunk_index == 1
    assert not app.state._limit_reached


def test_session_cleanup_releases_native_rollout():
    app = make_model()
    asyncio.run(step(app))
    app.on_session_ended()
    assert app._engine.resets == 1
    assert app._selected_input is None
    assert app.state._applied_world_id is None
    app.on_session_started()
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())


def test_model_owns_acknowledgment_counter_and_limit():
    model = MatrixGame30Model()
    backend = Mock()
    model._backend = backend
    model._max_chunks = 2
    action = action_from_controls(frozenset(("w",)), 0.2, -0.2)
    with pytest.raises(NoAnchor):
        model.generate(MatrixGame30Input(1, None, action))
    anchor = MatrixGame30Anchor(_png_bytes(), "", 42)
    backend.generate_chunk.return_value = np.zeros((57, 8, 8, 3), np.uint8)
    first = model.generate(MatrixGame30Input(1, anchor, action))
    assert first.chunk_index == 1 and not first.complete
    backend.reset.assert_called_once_with("", 42, anchor.image)
    backend.generate_chunk.return_value = np.zeros((40, 8, 8, 3), np.uint8)
    second = model.generate(MatrixGame30Input(1, None, action))
    assert second.chunk_index == 2 and second.complete
    assert backend.reset.call_count == 1
    with pytest.raises(RolloutExhausted):
        model.generate(MatrixGame30Input(1, None, action))
    backend.generate_chunk.return_value = np.zeros((40, 8, 8, 3), np.uint8)
    with pytest.raises(RuntimeError, match="57 frames"):
        model.generate(MatrixGame30Input(2, anchor, action))
    assert model._chunk_index == 0
    backend.generate_chunk.side_effect = RuntimeError("gpu failed")
    with pytest.raises(RuntimeError, match="gpu failed"):
        model.generate(MatrixGame30Input(2, anchor, action))
    assert model._chunk_index == 0
    model.reset()
    backend.end_session.assert_called_once()
    assert model._world_id is None


def test_model_and_configuration_import_without_runtime():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import matrix_game_3_0_model
import matrix_game_3_0_backend
import matrix_game_3_0_assets
matrix_game_3_0_model.MatrixGame30Model()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )


def test_explicit_weights_root_preserves_native_configuration(tmp_path):
    config = read_config(Path(__file__).parents[1] / "matrix_game_3_0.yaml", tmp_path)
    assert config.checkpoint_path == tmp_path / "checkpoints" / "Matrix-Game-3.0"
    assert config.max_chunks == 12
    assert config.num_inference_steps == 3


def test_backend_image_bytes_match_path(tmp_path):
    from matrix_game_3_0_backend import _read_image

    data = _png_bytes()
    path = tmp_path / "image.png"
    path.write_bytes(data)
    np.testing.assert_array_equal(
        np.asarray(_read_image(data)), np.asarray(_read_image(path))
    )
