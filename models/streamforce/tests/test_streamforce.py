"""Test the StreamForce application half through its hooks, without a GPU.

A recording model half stands in for StreamForceModel, so the suite needs no GPU, torch or
weights. The application is driven through its handlers and hooks the way the runtime does.
Run from the model folder:  PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile  # noqa: E402
from reactor_runtime.core.model import SessionStarted  # noqa: E402
from reactor_runtime.interface.model.contract import ModelContract  # noqa: E402
from streamforce import StreamForce  # noqa: E402
from streamforce_model import Force, StreamForceInput, StreamForceResult  # noqa: E402
from streamforce_types import (  # noqa: E402
    ChunkComplete,
    GenerationComplete,
    GenerationStarted,
    StateUpdate,
    StreamForceOutput,
)


class RecordingModel:
    """Returns nothing for the first three steps of a run, then 9 frames, then 12 per step."""

    frames_per_session = 4089

    def __init__(self, finish_at: int | None = None) -> None:
        self.steps: list[StreamForceInput] = []
        self.resets = 0
        self.finish_at = finish_at
        self.window = 0
        self.run = None

    def prompt_tokens(self, text: str) -> int:
        return len(text.split()) + 1

    def generate(self, inp: StreamForceInput) -> StreamForceResult:
        if inp.run_id != self.run:
            assert inp.first_frame is not None
            self.run, self.window = inp.run_id, 0
        self.steps.append(inp)
        w = self.window
        self.window += 1
        block = w - 3 if w >= 3 else None
        frames = None if block is None else np.zeros((9 if block == 0 else 12, 4, 4, 3), np.uint8)
        fin = self.finish_at is not None and block == self.finish_at
        return StreamForceResult(run_id=inp.run_id, window=w, block=block, frames=frames, prompt=inp.prompt,
                                 force=inp.force, seed=inp.seed, finished=fin)

    def reset(self) -> None:
        self.resets += 1
        self.run = None


def _png(w: int = 64, h: int = 48) -> UploadedFile:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 100, np.uint8)).save(buf, format="PNG")
    return UploadedFile(name="a.png", mime_type="image/png", data=buf.getvalue())


def _app(model: RecordingModel | None = None) -> tuple[StreamForce, RecordingModel, list[Any]]:
    app = StreamForce()
    m = model or RecordingModel()
    app.engine = m  # type: ignore[assignment]
    app.frames_per_session = m.frames_per_session
    sent: list[Any] = []
    app._on_loop_ready()
    app.bind_output(broadcast=sent.append, addressed=lambda *a: None, media=lambda chunk: None)
    return app, m, sent


async def _ready(app: StreamForce) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app.set_first_frame(_png())
    await app.set_prompt("  a mug on a desk  ")
    app.set_force("point", 0.5, 0.0, 0.25, 0.75)


async def _step(app: StreamForce) -> StreamForceOutput | None:
    inp = await app.process_input()
    return await app.process_output(StepOutcome(result=app.generate(inp), elapsed=0.01))


# -- contract -------------------------------------------------------------------------


def test_commands_are_the_standard_set() -> None:
    assert set(ModelContract.of(StreamForce).commands) == {
        "start", "pause", "resume", "reset", "set_first_frame", "set_prompt", "set_force", "set_seed"}


def test_run_and_submit_command_are_not_overridden() -> None:
    from reactor_runtime import ReactorApp
    assert StreamForce.run is ReactorApp.run
    assert StreamForce.submit_command is ReactorApp.submit_command


# -- lifecycle ------------------------------------------------------------------------


async def test_start_refuses_each_missing_input() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    for code in ("missing_first_frame", "missing_prompt", "missing_force"):
        with pytest.raises(CommandError) as e:
            await app.start()
        assert e.value.code == code
        if code == "missing_first_frame":
            await app.set_first_frame(_png())
        elif code == "missing_prompt":
            await app.set_prompt("")
    await app.set_first_frame(_png())
    app.set_force("wind", 0.3, 90.0)
    assert (await app.start()).state == "running"
    with pytest.raises(CommandError) as e:
        await app.start()
    assert e.value.code == "already_running"


async def test_nothing_steps_before_start_or_while_paused() -> None:
    app, _, _ = _app()
    await _ready(app)
    with pytest.raises(ApplicationError, match="idle"):
        await app.process_input()
    await app.start()
    app.pause()
    with pytest.raises(ApplicationError, match="paused"):
        await app.process_input()
    with pytest.raises(CommandError):
        app.pause()
    app.resume()
    await app.process_input()


async def test_the_first_frame_rides_only_on_the_first_step_of_a_run() -> None:
    app, model, _ = _app()
    await _ready(app)
    await app.start()
    await _step(app)
    await _step(app)
    assert model.steps[0].first_frame is not None and model.steps[1].first_frame is None


async def test_session_setup_is_locked_while_running_and_live_controls_are_not() -> None:
    app, model, _ = _app()
    await _ready(app)
    await app.start()
    for call in (lambda: app.set_seed(3),):
        with pytest.raises(CommandError) as e:
            call()
        assert e.value.code == "locked_while_running"
    with pytest.raises(CommandError) as e:
        await app.set_first_frame(_png())
    assert e.value.code == "locked_while_running"
    await _step(app)
    app.set_force("wind", 1.0, 180.0)
    await app.set_prompt("windy")
    await _step(app)
    assert model.steps[-1].force == Force("wind", 1.0, 180.0) and model.steps[-1].prompt == "windy"


async def test_prompt_is_stripped_and_empty_is_a_value_and_budget_is_enforced() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert (await app.set_prompt("   ")).settings["prompt"]["sent"] == ""
    with pytest.raises(CommandError) as e:
        await app.set_prompt("w " * 600)
    assert e.value.code == "prompt_too_long"


async def test_force_validation() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    for args in [("push", 0.5, 0.0), ("wind", 1.5, 0.0), ("point", 0.5, 0.0, 1.2, 0.5), ("wind", 0.5, float("nan"))]:
        with pytest.raises(CommandError) as e:
            app.set_force(*args)
        assert e.value.code == "unsupported_value"
    app.set_force("wind", 0.5, 10.0, 7.0, 7.0)  # x, y are ignored for wind
    assert app.state._force == Force("wind", 0.5, 10.0)


async def test_upload_refusals() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    with pytest.raises(CommandError) as e:
        await app.set_first_frame(UploadedFile(name="a.txt", mime_type="text/plain", data=b"hi"))
    assert e.value.code == "unsupported_media"
    with pytest.raises(CommandError) as e:
        await app.set_first_frame(UploadedFile(name="a.png", mime_type="image/png", data=b"nope"))
    assert e.value.code == "undecodable_image"


async def test_seed_minus_one_draws_a_concrete_seed_and_a_set_seed_is_used() -> None:
    app, model, sent = _app()
    await _ready(app)
    await app.start()
    await _step(app)
    assert model.steps[-1].seed >= 0
    assert any(isinstance(m, GenerationStarted) for m in sent)
    app.reset()
    await _ready(app)
    app.set_seed(42)
    await app.start()
    await _step(app)
    assert model.steps[-1].seed == 42


async def test_chunks_are_sent_with_chunk_complete_and_nothing_before_the_first_block() -> None:
    app, _, sent = _app()
    await _ready(app)
    await app.start()
    outs = [await _step(app) for _ in range(5)]
    assert outs[:3] == [None, None, None]
    assert outs[3].main_video.shape[0] == 9 if hasattr(outs[3].main_video, "shape") else True
    chunks = [m for m in sent if isinstance(m, ChunkComplete)]
    assert [c.index for c in chunks] == [0, 1] and [c.frames for c in chunks] == [9, 12]


async def test_run_end_returns_to_idle_and_reports_completion() -> None:
    app, model, sent = _app(RecordingModel(finish_at=1))
    await _ready(app)
    await app.start()
    for _ in range(5):
        await _step(app)
    assert app.state._phase == "idle"
    done = [m for m in sent if isinstance(m, GenerationComplete)]
    assert len(done) == 1 and done[0].frames == 21
    assert model.resets >= 1
    await app.start()  # the same inputs start a new run
    await _step(app)
    assert model.steps[-1].first_frame is not None


async def test_reset_clears_every_input_and_the_model() -> None:
    app, model, _ = _app()
    await _ready(app)
    app.set_seed(5)
    await app.start()
    await _step(app)
    snap = app.reset()
    assert snap.state == "idle"
    s = app.state
    assert s._first_frame is None and s._prompt is None and s._force is None and s._seed == -1
    assert model.resets >= 1
    with pytest.raises(CommandError) as e:
        await app.start()
    assert e.value.code == "missing_first_frame"


async def test_session_end_clears_like_reset() -> None:
    app, model, _ = _app()
    await _ready(app)
    await app.start()
    app.on_session_ended()
    assert app.state._phase == "idle" and app.state._first_frame is None and model.resets >= 1


async def test_a_joining_client_gets_the_snapshot() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    got: list[Any] = []

    class C:
        async def send(self, m: Any) -> None:
            got.append(m)

    await app.on_connect(C())  # type: ignore[arg-type]
    assert isinstance(got[0], StateUpdate) and got[0].state == "idle"
    assert {c["command"] for c in got[0].capabilities} == {"set_first_frame", "set_prompt", "set_force", "set_seed"}


async def test_a_stale_step_after_reset_is_dropped() -> None:
    app, model, _ = _app()
    await _ready(app)
    await app.start()
    inp = await app.process_input()
    result = app.generate(inp)
    app.reset()
    assert await app.process_output(StepOutcome(result=result, elapsed=0.01)) is None


async def test_a_refused_command_sends_nothing() -> None:
    app, _, sent = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    before = len(sent)
    with pytest.raises(CommandError):
        await app.start()
    assert len(sent) == before


async def test_a_refused_step_never_reaches_the_model() -> None:
    app, model, _ = _app()
    await _ready(app)
    with pytest.raises(ApplicationError):
        await app.process_input()
    assert model.steps == []


async def test_a_model_error_ends_the_session_and_is_not_a_chunk() -> None:
    app, _, sent = _app()
    await _ready(app)
    await app.start()
    with pytest.raises(RuntimeError):
        await app.process_output(StepOutcome(result=None, error=RuntimeError("gpu"), elapsed=0.01))
    assert not [m for m in sent if isinstance(m, ChunkComplete)]
