"""GPU-free tests: the application half driven through its hooks with a fake model half.

Run from the model folder: PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamdiffusionv2 import StreamDiffusionV2  # noqa: E402
from streamdiffusionv2_model import StreamInput, StreamResult  # noqa: E402
from streamdiffusionv2_types import StateUpdate  # noqa: E402

from reactor_runtime import ApplicationError, CommandError, InputFrame, StepOutcome  # noqa: E402
from reactor_runtime.core.model import EndReason, SessionEnded, SessionStarted  # noqa: E402
from reactor_runtime.interface.model.contract import ModelContract  # noqa: E402


class FakeModel:
    """Stands in for the model half: records inputs, emits nothing on a run's second chunk."""

    def __init__(self, fail: bool = False) -> None:
        self.inputs: list[StreamInput] = []
        self.resets = 0
        self.run: int | None = None
        self.index = 0
        self.fail = fail

    def generate(self, input: StreamInput) -> StreamResult:
        if self.fail:
            raise RuntimeError("CUDA error")
        self.inputs.append(input)
        if self.run != input.run_id:
            self.run, self.index = input.run_id, 0
        else:
            self.index += 1
        n = len(input.frames)
        frames = None if self.index == 1 else np.zeros((n, 480, 832, 3), np.uint8)
        return StreamResult(frames, self.index, input.run_id, input.noise_scale, input.prompt)

    def reset(self) -> None:
        self.resets += 1
        self.run = None


class Sent(list):
    def __call__(self, *args) -> None:
        self.append(args)


async def make_app(fail: bool = False):
    app = StreamDiffusionV2()
    app.engine = FakeModel(fail)  # type: ignore[assignment]
    app.tokenizer = lambda text, add_special_tokens=True: {"input_ids": text.split() + ["</s>"]}  # type: ignore[assignment]
    app._on_loop_ready()
    sent = Sent()
    app.bind_output(broadcast=sent, addressed=sent, media=lambda c: None)
    await app._dispatch_reactor_event(SessionStarted("s"))
    return app, sent


def push(app, n, stamp0=None, dt_us=62_500, size=(240, 320)):
    for i in range(n):
        stamp = None if stamp0 is None else stamp0 + i * dt_us
        app._input_buffers["video_in"].push(InputFrame(data=np.full((*size, 3), i, np.uint8), capture_time_us=stamp))


async def step(app):
    inp = await app.process_input()
    return inp, await app.process_output(StepOutcome(result=app.generate(inp)))


def configure(app, prompt="a dog"):
    app.set_prompt(prompt)
    app.set_seed(0)
    app.set_noise_scale(0.8)


def test_contract_commands():
    assert set(ModelContract.of(StreamDiffusionV2).commands) == {
        "set_prompt", "set_seed", "set_noise_scale", "start", "pause", "resume", "reset"}


async def test_start_refused_without_inputs():
    app, _ = await make_app()
    with pytest.raises(CommandError) as e:
        app.start()
    assert e.value.code == "missing_prompt"
    app.set_prompt("   ")  # strips to empty, which is a value
    assert app.state._prompt == ""
    with pytest.raises(CommandError) as e:
        app.start()
    assert e.value.code == "missing_seed"
    app.set_seed(0)
    with pytest.raises(CommandError) as e:
        app.start()
    assert e.value.code == "missing_noise_scale"


async def test_refused_step_never_reaches_the_model():
    app, _ = await make_app()
    push(app, 5)
    with pytest.raises(ApplicationError):
        await app.process_input()  # idle
    configure(app)
    app.start()
    push(app, 4)
    with pytest.raises(ApplicationError):
        await app.process_input()  # 4 of the 5 frames the first chunk needs
    assert app.engine.inputs == []


async def test_chunks_5_then_4():
    app, _ = await make_app()
    configure(app)
    app.start()
    push(app, 5)
    inp, out = await step(app)
    assert len(inp.frames) == 5 and out is not None
    push(app, 4)
    inp, out = await step(app)
    assert len(inp.frames) == 4 and out is None  # the stream batch fills
    push(app, 4)
    inp, out = await step(app)
    assert len(inp.frames) == 4 and out is not None


async def test_session_setup_locked_prompt_live():
    app, _ = await make_app()
    configure(app)
    app.start()
    for call, code in ((lambda: app.set_seed(3), "locked_while_running"),
                       (lambda: app.set_noise_scale(0.5), "locked_while_running"),
                       (app.start, "already_running"), (app.resume, "not_paused")):
        with pytest.raises(CommandError) as e:
            call()
        assert e.value.code == code
    app.set_prompt("  a cat  ")
    push(app, 5)
    inp, _ = await step(app)
    assert inp.prompt == "a cat"


async def test_refusals_keep_previous_value():
    app, _ = await make_app()
    app.set_seed(7)
    with pytest.raises(CommandError) as e:
        app.set_seed(-5)
    assert e.value.code == "invalid_seed" and app.state._seed == 7
    app.set_noise_scale(0.8)
    with pytest.raises(CommandError) as e:
        app.set_noise_scale(1.5)
    assert e.value.code == "invalid_noise_scale" and app.state._noise_scale == 0.8
    app.set_prompt("a dog")
    with pytest.raises(CommandError) as e:
        app.set_prompt(" ".join(["dog"] * 600))
    assert e.value.code == "prompt_too_long" and app.state._prompt == "a dog"
    app.set_seed(-1)
    assert 0 <= app.state._seed < 2**31


async def test_resample_by_capture_time():
    app, _ = await make_app()
    configure(app)
    app.start()
    push(app, 10, stamp0=1_000_000, dt_us=33_333)  # 30 fps in -> about 16 fps kept
    app._drain_video()
    assert len(app.state._pending) == 5


async def test_backlog_is_bounded():
    app, _ = await make_app()
    configure(app)
    app.start()
    push(app, 40)
    app._drain_video()
    assert len(app.state._pending) == 5 + 8


async def test_mixed_frame_sizes_in_one_chunk_go_as_a_list():
    app, _ = await make_app()
    configure(app)
    app.start()
    push(app, 3, size=(240, 320))
    push(app, 2, size=(360, 640))  # WebRTC rescaled the track mid-chunk
    inp, out = await step(app)
    assert isinstance(inp.frames, list) and len(inp.frames) == 5 and out is not None


async def test_pause_and_resume():
    app, _ = await make_app()
    configure(app)
    app.start()
    assert app.pause().state == "paused"
    push(app, 5)
    with pytest.raises(ApplicationError):
        await app.process_input()
    with pytest.raises(CommandError) as e:
        app.pause()
    assert e.value.code == "not_running"
    assert app.resume().state == "running"


async def test_reset_and_session_end_clear_everything():
    app, _ = await make_app()
    configure(app)
    app.start()
    snap = await app.reset()
    assert isinstance(snap, StateUpdate) and snap.state == "idle"
    assert app.state._prompt is None and app.state._seed is None and app.state._noise_scale is None
    resets = app.engine.resets
    await app._dispatch_reactor_event(SessionEnded("s", list(EndReason)[0]))
    assert app.engine.resets == resets + 1


async def test_model_error_reaches_process_output_and_is_raised():
    app, _ = await make_app(fail=True)
    configure(app)
    app.start()
    push(app, 5)
    inp = await app.process_input()
    outcome = app._generate(inp)
    assert outcome.error is not None
    with pytest.raises(RuntimeError):
        await app.process_output(outcome)
    assert app.state._chunk_index == -1  # not counted as a completed chunk
