"""Exercise take identity, native clip boundaries and application failure paths."""

import asyncio
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from liveavatar_model import (
    LiveAvatarInput,
    LiveAvatarModel,
    TakeConditions,
    TakeFailed,
)
from liveavatar_pipeline import LiveAvatar
from liveavatar_types import LiveAvatarState
from reactor_runtime import ApplicationError, StepOutcome


class Backend:
    def __init__(self):
        self.starts = []
        self.index = 0

    def start(self, **conditions):
        self.starts.append(conditions)
        self.index = 0

    def next(self):
        if self.index == 10:
            return None
        count = 45 if self.index == 0 else 48
        self.index += 1
        return np.full((count, 2, 2, 3), self.index, np.uint8), np.full(
            (1, count * 1920), self.index / 100, np.float32
        )

    def close(self):
        pass


def app():
    result = LiveAvatar()
    result._engine = LiveAvatarModel()
    result._engine.shutdown = Mock()

    # This fixture exercises app/model integration with a CPU backend. Restart
    # creates another fake runner without spawning GPU workers.
    def restart():
        if result._engine is None:
            result._engine = LiveAvatarModel()
            result._engine.shutdown = Mock()
            result._engine._backend = Backend()

    result._start_engine = restart
    result.state = LiveAvatarState()
    result.state._image, result.state._audio = Path("image.png"), Path("audio.wav")
    result.send = AsyncMock()
    result.output = Mock()
    result._engine._backend = Backend()
    return result


def test_refusal_never_calls_model():
    model = app()
    model._engine.generate = Mock(side_effect=AssertionError("must not run"))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model._engine.generate.assert_not_called()


def test_generate_reads_only_snapshot():
    model = app()
    asyncio.run(model.start())
    value = asyncio.run(model.process_input())
    with pytest.raises(FrozenInstanceError):
        value.take_id = 99
    model.state = None
    result = model.generate(value)
    assert result.take_id == value.take_id and result.chunks == 1
    assert not hasattr(result, "input")


def test_ten_continuous_native_chunks_and_completion():
    model = app()
    asyncio.run(model.start())
    for index in range(10):
        value = asyncio.run(model.process_input())
        assert (value.conditions is not None) == (index == 0)
        result = model.generate(value)
        output = asyncio.run(model.process_output(StepOutcome(result=result)))
        assert len(output.main_video) == (45 if index == 0 else 48)
        assert output.main_audio.shape[1] == len(output.main_video) * 1920
        assert model.state._chunks == index + 1
    assert len(model._engine._backend.starts) == 1
    complete = model.generate(asyncio.run(model.process_input()))
    assert complete.complete and complete.frames == 477
    assert asyncio.run(model.process_output(StepOutcome(result=complete))) is None
    assert not model.state._running
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.start())
    fresh = asyncio.run(model.process_input())
    assert fresh.conditions is not None and fresh.take_id != complete.take_id
    assert model.generate(fresh).chunks == 1


def test_failed_chunk_never_acknowledges_or_counts():
    model = app()
    asyncio.run(model.start())
    model._engine._backend.next = Mock(side_effect=RuntimeError("worker failed"))
    with pytest.raises(TakeFailed) as error:
        model.generate(asyncio.run(model.process_input()))
    assert model._engine._chunks == 0
    asyncio.run(model.process_output(StepOutcome(error=error.value)))
    assert model.state._applied_take_id is None and model.state._chunks == 0
    assert model.state._error == "worker failed" and not model.state._running
    assert all(
        type(c.args[0]).__name__ != "ChunkComplete" for c in model.send.call_args_list
    )


def test_unexpected_contract_errors_propagate():
    model = app()
    with pytest.raises(ValueError, match="bad audio"):
        asyncio.run(model.process_output(StepOutcome(error=ValueError("bad audio"))))


def test_conditions_until_ack_and_failure_after_success():
    model = app()
    asyncio.run(model.start())
    first = asyncio.run(model.process_input())
    assert first.conditions is not None
    assert asyncio.run(model.process_input()).conditions == first.conditions
    result = model.generate(first)
    asyncio.run(model.process_output(StepOutcome(result=result)))
    later = asyncio.run(model.process_input())
    assert later.conditions is None
    model.send.reset_mock()
    model._engine._backend.next = Mock(side_effect=RuntimeError("later worker failed"))
    with pytest.raises(TakeFailed) as error:
        model.generate(later)
    assert model._engine._chunks == 1
    asyncio.run(model.process_output(StepOutcome(error=error.value)))
    assert model.state._chunks == 1 and not model.state._running
    assert all(
        type(c.args[0]).__name__ != "ChunkComplete" for c in model.send.call_args_list
    )
    asyncio.run(model.start())
    fresh = asyncio.run(model.process_input())
    assert fresh.conditions is not None and fresh.take_id != first.take_id


def test_model_guards_and_reset():
    model = LiveAvatarModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.generate(LiveAvatarInput(1, None))
    model._backend = Backend()
    with pytest.raises(ValueError, match="requires"):
        model.generate(LiveAvatarInput(1, None))
    model.generate(
        LiveAvatarInput(1, TakeConditions(Path("a"), Path("b"), None, "", "", 420, 10))
    )
    model.reset()
    assert model._take_id is None and model._chunks == model._frames == 0


def test_model_graph_imports_without_runtime():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import liveavatar_model, liveavatar_parallel, liveavatar_assets
liveavatar_model.LiveAvatarModel()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )


@pytest.mark.parametrize(
    "bad_audio",
    [
        np.full((1, 86400), 2, np.float32),
        np.zeros((1, 86400), np.int16),
        np.full((1, 86400), np.nan, np.float32),
    ],
)
def test_invalid_audio_does_not_count_clip(bad_audio):
    model = LiveAvatarModel()
    model._backend = Backend()
    model._backend.next = lambda: (np.zeros((45, 2, 2, 3), np.uint8), bad_audio)
    step = LiveAvatarInput(
        1, TakeConditions(Path("image"), Path("audio"), None, "", "", 420, 1)
    )
    with pytest.raises(ValueError, match="48 kHz"):
        model.generate(step)
    assert model._chunks == model._frames == 0
