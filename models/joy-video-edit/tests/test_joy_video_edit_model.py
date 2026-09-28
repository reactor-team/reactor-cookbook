# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The model half's run bookkeeping, with the JoyOmni runtime replaced by a fake.

Covers opening a run from the conditioning the step carries, continuing it,
replacing its session every ``kv_reset_frames`` frames, the frames each result
asks for next, a failed chunk, and ``reset()``. The fake session keeps the real
one's frame rule: a session's first chunk takes one frame, every later chunk
eight. No GPU, no weights, no torch.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from joy_video_edit_model import (
    ChunkFailed,
    JoyVideoEditInput,
    JoyVideoEditModel,
    NoConditioning,
    RunBroken,
    RunConditioning,
    WrongFrameCount,
)

_MODEL_DIR = Path(__file__).resolve().parent.parent
_REFERENCE = np.full((32, 64, 3), 9, dtype=np.uint8)
_COND = RunConditioning(prompt="watercolor", reference=None, seed=42)


@dataclass
class _Chunk:
    frames: np.ndarray | None
    jpegs: list[np.ndarray] | None = None
    valid_count: int | None = None


class FakeSession:
    def __init__(self, prompt: str, settings: Any, ref_image: Any) -> None:
        self.prompt, self.settings, self.ref_image = prompt, settings, ref_image
        self.chunk_idx = 0
        self.closed = False
        self.fail_next = False
        self.pushed: list[int] = []

    @property
    def frames_per_next_chunk(self) -> int:
        return 1 if self.chunk_idx == 0 else 8

    def push_chunk(self, frames: list[np.ndarray]) -> _Chunk:
        if self.fail_next:
            raise RuntimeError("CUDA error")
        self.pushed.append(len(frames))
        self.chunk_idx += 1
        return _Chunk(frames=np.stack(frames))

    def close(self) -> None:
        self.closed = True


class FakeRuntime:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def create_v2v_session(self, prompt: str, *, settings: Any, ref_image: Any = None) -> FakeSession:
        session = FakeSession(prompt, settings, ref_image)
        self.sessions.append(session)
        return session


def _model(kv_reset_frames: int = 0) -> tuple[JoyVideoEditModel, FakeRuntime]:
    model = JoyVideoEditModel()
    runtime = FakeRuntime()
    model._runtime = runtime
    model._compute = ThreadPoolExecutor(max_workers=1)
    model._kv_reset_frames = kv_reset_frames
    model._num_inference_steps = 2
    model._session_settings = lambda *, seed: ("settings", seed)  # type: ignore[method-assign]
    return model, runtime


def _frames(n: int) -> list[np.ndarray]:
    return [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(n)]


def _step(model: JoyVideoEditModel, n: int, run_id: int = 1, conditioning: RunConditioning | None = None):
    return model.generate(JoyVideoEditInput(frames=_frames(n), run_id=run_id, conditioning=conditioning))


def test_the_model_half_imports_nothing_from_the_runtime() -> None:
    tree = ast.parse((_MODEL_DIR / "joy_video_edit_model.py").read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith("reactor_runtime") for name in imported)


def test_the_first_step_opens_the_run_from_its_conditioning() -> None:
    model, runtime = _model()
    cond = RunConditioning(prompt="watercolor", reference=_REFERENCE, seed=7)
    result = _step(model, 1, conditioning=cond)
    (session,) = runtime.sessions
    assert session.prompt == "watercolor" and session.settings == ("settings", 7)
    assert np.array_equal(np.asarray(session.ref_image), _REFERENCE)
    assert result.run_id == 1 and result.frames.shape == (1, 4, 6, 3)
    assert result.frames_wanted == 8


def test_a_new_run_without_conditioning_is_refused() -> None:
    model, runtime = _model()
    with pytest.raises(NoConditioning):
        _step(model, 1)
    assert runtime.sessions == []


def test_later_steps_continue_the_run_on_its_session() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)
    for _ in range(3):
        result = _step(model, 8)
    assert len(runtime.sessions) == 1 and runtime.sessions[0].pushed == [1, 8, 8, 8]
    assert result.frames.shape[0] == 8


def test_a_new_run_id_ends_the_live_run() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)
    _step(model, 8)
    result = _step(model, 1, run_id=2, conditioning=RunConditioning(prompt="sepia", reference=None, seed=1))
    first, second = runtime.sessions
    assert first.closed and not second.closed
    assert second.prompt == "sepia" and result.run_id == 2


def test_the_session_is_replaced_every_kv_reset_frames_with_the_same_conditioning() -> None:
    model, runtime = _model(kv_reset_frames=17)
    assert _step(model, 1, conditioning=_COND).frames_wanted == 8
    assert _step(model, 8).frames_wanted == 8
    # 1 + 8 + 8 = 17 frames: the next step opens a fresh session, whose first chunk takes one.
    assert _step(model, 8).frames_wanted == 1
    result = _step(model, 1)
    first, second = runtime.sessions
    assert first.closed and first.pushed == [1, 8, 8]
    assert second.prompt == "watercolor" and second.pushed == [1]
    assert result.run_id == 1 and result.frames_wanted == 8


def test_a_step_with_the_wrong_frame_count_is_refused() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)
    with pytest.raises(WrongFrameCount):
        _step(model, 3)
    assert runtime.sessions[0].pushed == [1]


def test_a_failed_chunk_breaks_the_run_until_reset() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)
    runtime.sessions[0].fail_next = True
    with pytest.raises(ChunkFailed) as failed:
        _step(model, 8)
    assert isinstance(failed.value.__cause__, RuntimeError)
    with pytest.raises(RunBroken):
        _step(model, 8)

    model.reset()
    assert runtime.sessions[0].closed
    result = _step(model, 1, conditioning=_COND)
    assert len(runtime.sessions) == 2 and result.run_id == 1


def test_reset_closes_the_session_and_forgets_the_run() -> None:
    model, runtime = _model()
    model.reset()  # nothing live: a no-op
    _step(model, 1, conditioning=_COND)
    model.reset()
    assert runtime.sessions[0].closed
    with pytest.raises(NoConditioning):
        _step(model, 1)


def test_a_session_that_fails_to_close_does_not_fail_the_reset() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)

    def _boom() -> None:
        raise RuntimeError("sync failed")

    runtime.sessions[0].close = _boom  # type: ignore[method-assign]
    model.reset()
    assert _step(model, 1, conditioning=_COND).run_id == 1


def test_the_result_trims_a_padded_chunk_and_stacks_encoded_frames() -> None:
    model, runtime = _model()
    _step(model, 1, conditioning=_COND)
    session = runtime.sessions[0]
    session.push_chunk = lambda frames: _Chunk(frames=np.stack(frames), valid_count=5)  # type: ignore[method-assign]
    assert _step(model, 8).frames.shape[0] == 5
    session.push_chunk = lambda frames: _Chunk(frames=None, jpegs=list(frames))  # type: ignore[method-assign]
    assert _step(model, 8).frames.shape == (8, 4, 6, 3)
