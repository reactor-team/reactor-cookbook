# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The application half, driven through its hooks with a fake model half.

Covers the client contract, each refusal in ``process_input``, the camera read
policies against the runtime's own input buffer, the run's conditioning riding
on the step input only until the model reports the run, every message
``process_output`` sends and its recovery from a failed chunk, the
hand-written commands, and the lifecycle hooks. No GPU, no weights, no torch.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from joy_video_edit import JoyVideoEdit, _InputStats
from joy_video_edit_model import (
    ChunkFailed,
    JoyVideoEditInput,
    JoyVideoEditResult,
    RunConditioning,
    WrongFrameCount,
)
from joy_video_edit_types import (
    ChunkComplete,
    CommandError,
    GenerationComplete,
    GenerationReset,
    GenerationStarted,
    JoyVideoEditMedia,
    JoyVideoEditOutput,
    PromptAccepted,
    ReferenceImageAccepted,
    SessionState,
)
from PIL import Image
from reactor_runtime import ApplicationError, ClientInfo, StepOutcome, UploadedFile
from reactor_runtime.core.model import ClientConnected, ClientDisconnected, EndReason, SessionEnded, SessionStarted
from reactor_runtime.core.values import ConnId, InputFrame
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _frame(value: int) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


class FakeModel:
    """The model half by shape: records the steps and returns scripted results."""

    def __init__(self) -> None:
        self.steps: list[JoyVideoEditInput] = []
        self.resets = 0

    def generate(self, input: JoyVideoEditInput) -> JoyVideoEditResult:
        self.steps.append(input)
        return _result(run_id=input.run_id, n=len(input.frames))

    def reset(self) -> None:
        self.resets += 1


def _result(run_id: int = 1, n: int = 8, frames_wanted: int = 8) -> JoyVideoEditResult:
    return JoyVideoEditResult(frames=np.zeros((n, 4, 6, 3), np.uint8), run_id=run_id, frames_wanted=frames_wanted)


def _app(camera_read: str = "latest", backlog_chunks: int = 3) -> tuple[JoyVideoEdit, FakeModel, list[Any], InputBuffer]:
    app = JoyVideoEdit()
    model = FakeModel()
    app.engine = model  # type: ignore[assignment]  # the model half by shape
    app._camera_read = camera_read
    app._camera_backlog_frames = backlog_chunks * 8
    app._input_stats = _InputStats()
    app._failures_in_a_row = 0
    camera = InputBuffer()
    app.media = JoyVideoEditMedia(camera=camera)
    sent: list[Any] = []
    app._on_loop_ready()
    app.bind_output(broadcast=sent.append, addressed=lambda *args: None, media=lambda chunk: None)
    return app, model, sent, camera


async def _live(app: JoyVideoEdit) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))


def _push(camera: InputBuffer, *values: int) -> None:
    for v in values:
        camera.push(InputFrame(data=_frame(v), pts=float(v)))


async def _step(app: JoyVideoEdit) -> JoyVideoEditOutput | None:
    """One turn of the runtime's step loop: the three hooks in order."""
    input = await app.process_input()
    return await app.process_output(StepOutcome(result=app.generate(input), elapsed=0.2))


def _client(inbox: list[Any]) -> ClientInfo:
    return ClientInfo(id=ConnId(1), joined_at=0.0, _send=inbox.append)


def _upload(size: tuple[int, int] = (64, 32), color: tuple[int, int, int] = (200, 10, 10)) -> UploadedFile:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return UploadedFile(name="ref.png", mime_type="image/png", data=buffer.getvalue())


# -- the client contract ------------------------------------------------------


def test_commands_are_the_four_hand_written_ones_and_set_seed() -> None:
    assert set(ModelContract.of(JoyVideoEdit).commands) == {
        "set_prompt", "set_reference_image", "start", "reset", "set_seed",
    }


def test_the_tracks_are_camera_in_and_main_video_out() -> None:
    assert list(ModelContract.of(JoyVideoEdit).tracks) == ["camera", "main_video"]  # inbound first


def test_manifest_resolves_to_the_app_class() -> None:
    cfg = load_config(_MODEL_DIR / "reactor.yaml")
    assert import_model_class(cfg.model_ref) is JoyVideoEdit
    assert cfg.config_path == _MODEL_DIR / "joy_video_edit.yaml"


def test_the_state_holds_no_loop_flags() -> None:
    state = JoyVideoEdit.__app_state__()
    assert state._started is False and state._run_id == 0 and state._applied_run_id is None
    for gone in ("_do_reset", "_ref_pil"):
        assert not hasattr(state, gone)


# -- process_input ---------------------------------------------------------------


async def test_process_input_refuses_before_start() -> None:
    app, _, _, camera = _app()
    await _live(app)
    _push(camera, 1)
    with pytest.raises(ApplicationError, match="not started"):
        await app.process_input()


async def test_process_input_refuses_until_the_camera_has_the_frames_the_model_wants() -> None:
    app, _, _, camera = _app()
    await _live(app)
    await app.start()
    app.state._frames_wanted = 8
    _push(camera, *range(7))
    with pytest.raises(ApplicationError, match="waiting for 8 camera frames"):
        await app.process_input()
    assert camera.available == 7  # a refused read takes nothing


async def test_the_conditioning_is_snapshotted_once_and_rides_only_until_the_model_reports_the_run() -> None:
    app, model, _, camera = _app()
    await _live(app)
    await app.set_prompt("watercolor")
    app.state.seed = 7
    await app.start()

    _push(camera, 1)
    await _step(app)
    first = model.steps[0]
    assert first.run_id == 1
    assert first.conditioning == RunConditioning(prompt="watercolor", reference=None, seed=7)
    assert app.state._applied_run_id == 1

    # A prompt change during the run does not reach it.
    await app.set_prompt("sepia")
    _push(camera, *range(8))
    await _step(app)
    assert model.steps[1].conditioning is None
    assert app.state._conditioning.prompt == "watercolor"


async def test_latest_takes_the_newest_frames_and_drops_the_backlog() -> None:
    app, model, _, camera = _app("latest")
    await _live(app)
    await app.start()
    _push(camera, *range(5))
    await _step(app)
    assert [int(f[0, 0, 0]) for f in model.steps[0].frames] == [4]
    _push(camera, *range(10, 30))
    await _step(app)
    assert [int(f[0, 0, 0]) for f in model.steps[1].frames] == list(range(22, 30))
    assert camera.available == 0
    assert app._input_stats.dropped == 4 + 12


async def test_fifo_reads_in_arrival_order_after_the_first_read_and_trims_a_long_backlog() -> None:
    app, model, _, camera = _app("fifo", backlog_chunks=1)
    await _live(app)
    await app.start()
    _push(camera, *range(3))
    await _step(app)
    assert [int(f[0, 0, 0]) for f in model.steps[0].frames] == [2]  # the first read of a run is the newest
    _push(camera, *range(10, 30))
    await _step(app)
    assert [int(f[0, 0, 0]) for f in model.steps[1].frames] == list(range(10, 18))
    assert camera.available == 8  # one chunk of backlog kept, the oldest four dropped
    assert app._input_stats.dropped == 2 + 4


async def test_frames_per_step_follow_the_result() -> None:
    app, model, _, camera = _app()
    await _live(app)
    await app.start()
    _push(camera, 1)
    input = await app.process_input()
    await app.process_output(StepOutcome(result=_result(n=1, frames_wanted=1), elapsed=0.1))
    assert app.state._frames_wanted == 1
    _push(camera, 2)
    assert len((await app.process_input()).frames) == 1
    assert input.run_id == 1


# -- process_output -------------------------------------------------------------------


async def test_process_output_reports_the_chunk_and_returns_its_frames() -> None:
    app, _, sent, _ = _app()
    await _live(app)
    await app.start()
    sent.clear()
    result = _result(n=8)
    media = await app.process_output(StepOutcome(result=result, elapsed=0.1954))
    assert isinstance(media, JoyVideoEditOutput) and media.main_video is result.frames
    (chunk,) = sent
    assert isinstance(chunk, ChunkComplete)
    assert (chunk.chunk_index, chunk.frames_emitted, chunk.elapsed_ms) == (0, 8, 195.4)
    await app.process_output(StepOutcome(result=_result(n=8), elapsed=0.2))
    assert sent[1].chunk_index == 1
    assert (app.state._chunk_index, app.state._total_frames) == (2, 16)


async def test_a_failed_chunk_starts_a_new_session_of_the_same_run_once() -> None:
    app, model, sent, camera = _app()
    await _live(app)
    await app.start()
    _push(camera, 1)
    await _step(app)
    sent.clear()

    media = await app.process_output(StepOutcome(error=ChunkFailed("boom")))
    assert media is None and sent == []
    assert model.resets == 1
    assert app.state._applied_run_id is None and app.state._frames_wanted == 1
    assert app.state._started is True

    # The next step reopens the run from the same conditioning; the chunk count carries on.
    _push(camera, 2)
    await _step(app)
    assert model.steps[-1].conditioning == model.steps[0].conditioning
    assert sent[-1].chunk_index == 1

    # Failing again right after a fresh session is not a session fault.
    await app.process_output(StepOutcome(error=ChunkFailed("boom")))
    with pytest.raises(ChunkFailed):
        await app.process_output(StepOutcome(error=ChunkFailed("again")))


async def test_process_output_reraises_any_other_model_error() -> None:
    app, _, _, _ = _app()
    await _live(app)
    with pytest.raises(WrongFrameCount):
        await app.process_output(StepOutcome(error=WrongFrameCount("8 wanted")))


# -- commands and hooks -----------------------------------------------------------


async def test_start_opens_a_run_once() -> None:
    app, model, sent, _ = _app()
    await _live(app)
    await app.set_prompt("watercolor")
    sent.clear()
    reply = await app.start()
    assert isinstance(reply, GenerationStarted)
    assert reply.prompt == "watercolor" and reply.has_reference_image is False
    (state,) = sent
    assert isinstance(state, SessionState) and state.started is True
    assert app.state._run_id == 1

    inbox: list[Any] = []
    sent.clear()
    assert await app.start(client=_client(inbox)) is None
    (error,) = inbox
    assert isinstance(error, CommandError) and error.command == "start"
    assert sent == [] and app.state._run_id == 1
    assert model.resets == 0


async def test_reset_ends_the_run_and_reports_its_totals() -> None:
    app, model, sent, camera = _app()
    await _live(app)
    await app.start()
    _push(camera, 1)
    await _step(app)
    sent.clear()

    await app.reset()
    assert [type(m) for m in sent] == [GenerationReset, GenerationComplete, SessionState]
    assert (sent[1].total_chunks, sent[1].total_frames) == (1, 1)
    assert sent[2].started is False
    assert model.resets == 1
    assert app.state._applied_run_id is None and app.state._conditioning is None

    # A new start is a new run id, so the next step reopens the model.
    await app.start()
    _push(camera, 2)
    await _step(app)
    assert model.steps[-1].run_id == 2 and model.steps[-1].conditioning is not None


async def test_reset_then_start_back_to_back_generates() -> None:
    app, model, _, camera = _app()
    await _live(app)
    await app.reset()
    await app.start()
    _push(camera, 1)
    assert isinstance(await _step(app), JoyVideoEditOutput)
    assert model.steps[0].run_id == 1 and app.state._started is True


async def test_reset_while_waiting_only_reports_the_state() -> None:
    app, model, sent, _ = _app()
    await _live(app)
    await app.reset()
    assert [type(m) for m in sent] == [SessionState]
    assert model.resets == 0


async def test_set_prompt_stores_the_text_and_replies() -> None:
    app, _, sent, _ = _app()
    await _live(app)
    reply = await app.set_prompt("anime hero")
    assert isinstance(reply, PromptAccepted) and reply.prompt == "anime hero"
    assert app.state.prompt == "anime hero"
    assert [type(m) for m in sent] == [SessionState]


async def test_set_reference_image_stages_the_decoded_image() -> None:
    app, _, sent, _ = _app()
    await _live(app)
    reply = await app.set_reference_image(_upload(size=(64, 32)))
    reference = app.state._reference
    assert reference.shape == (32, 64, 3) and reference.dtype == np.uint8
    assert reference[0, 0].tolist() == [200, 10, 10]
    assert isinstance(reply, ReferenceImageAccepted) and (reply.width, reply.height) == (64, 32)
    (state,) = sent
    assert isinstance(state, SessionState) and state.has_reference_image is True


async def test_set_reference_image_is_refused_while_generating() -> None:
    app, _, sent, _ = _app()
    await _live(app)
    await app.start()
    sent.clear()
    inbox: list[Any] = []
    assert await app.set_reference_image(_upload(), client=_client(inbox)) is None
    (error,) = inbox
    assert isinstance(error, CommandError) and error.command == "set_reference_image"
    assert app.state._reference is None and sent == []


async def test_the_last_client_leaving_ends_the_run() -> None:
    app, model, sent, _ = _app()
    await _live(app)
    await app._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    await app._dispatch_reactor_event(ClientConnected(ConnId(2), 2))
    await app.start()
    sent.clear()

    await app._dispatch_reactor_event(ClientDisconnected(ConnId(2), 1))
    assert sent == [] and app.state._started is True

    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1), 0))
    assert [type(m) for m in sent] == [GenerationComplete, SessionState]
    assert app.state._started is False and model.resets == 1


async def test_session_end_resets_the_model_half() -> None:
    app, model, _, _ = _app()
    await _live(app)
    await app.start()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert model.resets == 1
    assert app.state is None
