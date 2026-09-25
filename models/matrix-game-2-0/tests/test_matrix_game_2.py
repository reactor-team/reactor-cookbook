"""Exercise application hooks and model bookkeeping without GPU dependencies."""

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
from matrix_game_2 import MatrixGame2
from matrix_game_2_backend import ChunkAction
from matrix_game_2_model import (
    MatrixGame2Anchor,
    MatrixGame2Input,
    MatrixGame2Model,
    MatrixGame2Result,
    NoAnchor,
    RolloutExhausted,
)
from matrix_game_2_types import MatrixGame2State
from PIL import Image
from reactor_runtime import ApplicationError, StepOutcome, UploadedFile


class FakeModel:
    def __init__(self, max_chunks=120):
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
        return MatrixGame2Result(
            np.zeros((9 if self.chunk == 1 else 12, 8, 8, 3), dtype=np.uint8),
            self.world_id,
            self.chunk,
            input.action,
            self.chunk == self.max_chunks,
        )

    def reset(self):
        self.resets += 1
        self.world_id = None
        self.chunk = 0


def make_app(max_chunks=120):
    app = MatrixGame2()
    app.state = MatrixGame2State()
    app._config = SimpleNamespace(seed=42, max_chunks=max_chunks)
    app._engine = FakeModel(max_chunks)
    app.send = AsyncMock()
    app.output = Mock()
    app._selected_input = Path("anchor.png")
    app.state._world_id = 1
    return app


async def step(app):
    return await app.process_output(
        StepOutcome(result=app.generate(await app.process_input()), elapsed=0.25)
    )


def test_input_refusals_never_reach_model():
    app = make_app()
    app._selected_input = None
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app._selected_input = Path("anchor.png")
    app.state._limit_reached = True
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    assert app._engine.inputs == []


def test_generate_only_forwards_frozen_input():
    app = make_app()
    input = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        input.world_id = 99
    app.state = None
    app._config = None
    app._selected_input = None
    result = app.generate(input)
    assert app._engine.inputs == [input]
    assert result.world_id == input.world_id
    assert result.chunk_index == 1
    assert not hasattr(result, "input")


def test_ten_actions_keep_one_world_and_send_anchor_once():
    app = make_app()
    for index in range(10):
        app.state._pressed_keys = frozenset(("wasd"[index % 4],))
        app.state.pitch = (index - 5) / 10
        app.state.yaw = (5 - index) / 10
        input = asyncio.run(app.process_input())
        assert (input.anchor is not None) == (index == 0)
        app.state.yaw = 0.0
        result = app.generate(input)
        output = asyncio.run(
            app.process_output(StepOutcome(result=result, elapsed=0.25))
        )
        assert output.main_video.shape[0] == (9 if index == 0 else 12)
        complete, update = [call.args[0] for call in app.send.await_args_list[-2:]]
        assert complete.inference_seconds == 0.25
        assert complete.yaw == (5 - index) / 10
        assert complete.pressed_keys == ["wasd"[index % 4]]
        assert update.completed_chunks == index + 1
    assert len({input.world_id for input in app._engine.inputs}) == 1
    assert app._chunk_index == 10
    assert not app._state_update().reset_queued


def test_model_error_does_not_acknowledge_or_advance():
    app = make_app()
    failure = RuntimeError("failed")
    app._engine.generate = Mock(side_effect=failure)
    with pytest.raises(RuntimeError, match="failed"):
        app.generate(asyncio.run(app.process_input()))
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(app.process_output(StepOutcome(error=failure)))
    assert app._chunk_index == 0
    assert app.state._applied_world_id is None
    assert app._state_update().reset_queued
    app.send.assert_not_called()


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_and_reset_create_new_worlds():
    app = make_app()
    asyncio.run(step(app))
    first_id = app.state._world_id
    app.reset(seed=123)
    input = asyncio.run(app.process_input())
    assert input.world_id == first_id + 1
    assert input.anchor.seed == 123
    assert input.action == ChunkAction((), 0.0, 0.0)
    asyncio.run(step(app))
    data = image_bytes()
    upload = UploadedFile(
        name="anchor.png", mime_type="image/png", data=data
    )
    app.set_image(upload)
    input = asyncio.run(app.process_input())
    assert input.world_id == first_id + 2
    assert input.anchor.image == data
    assert not isinstance(input.anchor.image, UploadedFile)
    asyncio.run(step(app))
    assert app._chunk_index == 1
    assert app.output.flush.call_count == 2


def test_limit_reports_final_media_and_requires_explicit_reset():
    app = make_app(max_chunks=2)
    asyncio.run(step(app))
    app.state._pressed_keys = frozenset(("w",))
    output = asyncio.run(step(app))
    assert output.main_video.shape[0] == 12
    assert app.state._limit_reached
    assert not app.state._pressed_keys
    messages = [call.args[0] for call in app.send.await_args_list[-3:]]
    assert [type(message).__name__ for message in messages] == [
        "RolloutLimitReached",
        "ChunkComplete",
        "StateUpdate",
    ]
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app.reset(seed=-1)
    asyncio.run(step(app))
    assert not app.state._limit_reached
    assert app._chunk_index == 1


def test_session_cleanup_resets_model_without_defaults():
    app = make_app()
    asyncio.run(step(app))
    app.on_session_ended()
    assert app._engine.resets == 1
    assert app._selected_input is None
    assert app.state._applied_world_id is None
    app.on_session_started()
    assert not app._state_update().reset_queued
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())


def test_model_owns_world_counter_limit_and_image_decode(tmp_path):
    engine = MatrixGame2Model()
    backend = Mock()
    backend.generate_chunk.return_value = np.zeros((9, 8, 8, 3), dtype=np.uint8)
    engine._backend = backend
    engine._max_chunks = 2
    action = ChunkAction(("w", "a"), 0.2, -0.3)
    with pytest.raises(NoAnchor):
        engine.generate(MatrixGame2Input(1, None, action))
    anchor = MatrixGame2Anchor(image_bytes(), 42)
    first = engine.generate(MatrixGame2Input(1, anchor, action))
    second = engine.generate(MatrixGame2Input(1, None, action))
    assert (first.chunk_index, second.chunk_index) == (1, 2)
    assert not first.complete and second.complete
    assert backend.reset.call_count == 1
    image, seed = backend.reset.call_args.args
    assert image.mode == "RGB" and image.size == (16, 12) and seed == 42
    backend.generate_chunk.assert_called_with(action)
    with pytest.raises(RolloutExhausted):
        engine.generate(MatrixGame2Input(1, None, action))
    image_path = tmp_path / "image.png"
    image.save(image_path)
    result = engine.generate(
        MatrixGame2Input(2, MatrixGame2Anchor(image_path, 7), action)
    )
    assert result.world_id == 2 and result.chunk_index == 1
    backend.generate_chunk.side_effect = RuntimeError("gpu failure")
    with pytest.raises(RuntimeError, match="gpu failure"):
        engine.generate(MatrixGame2Input(2, None, action))
    assert engine._chunk_index == 1
    engine.reset()
    backend.end_rollout.assert_called_once()
    assert engine._world_id is None and engine._chunk_index == 0


def test_model_import_graph_is_runtime_free():
    code = """
import builtins
original = builtins.__import__
def isolated(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = isolated
import matrix_game_2_model
matrix_game_2_model.MatrixGame2Model()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )
