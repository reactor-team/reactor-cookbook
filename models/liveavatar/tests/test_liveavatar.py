import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from liveavatar_pipeline import LiveAvatar
from liveavatar_model import LiveAvatarModel, TakeFailed
from liveavatar_types import LiveAvatarOutput, LiveAvatarState, StateUpdate


def image_upload():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
    return UploadedFile(name="test.png", mime_type="image/png", data=buf.getvalue())


def audio_upload(seconds=2):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds)), 16000, format="WAV")
    return UploadedFile(name="test.wav", mime_type="audio/wav", data=buf.getvalue())


@pytest.fixture
def model(tmp_path):
    from unittest.mock import Mock

    model = LiveAvatar()
    model._engine = LiveAvatarModel()
    model._engine.shutdown = Mock()
    model.state = LiveAvatarState()
    model.state._directory = type("Directory", (), {"name": str(tmp_path)})()
    return model


def test_contract():
    import inspect

    commands = ModelContract.of(LiveAvatar).commands
    assert set(commands) == {
        "set_avatar_image",
        "set_audio",
        "set_pose_video",
        "set_prompt",
        "start",
        "set_generation_options",
        "stop",
        "reset",
    }
    assert LiveAvatar.fps == 25
    assert list(inspect.signature(LiveAvatar.start).parameters) == ["self"]
    assert LiveAvatarOutput.__tracks__["main_audio"].rate == 48000
    assert StateUpdate.from_state(LiveAvatarState()).ready is False


def test_wire_stop_does_not_override_runtime_shutdown():
    import inspect

    assert not inspect.iscoroutinefunction(LiveAvatar.stop)
    assert inspect.iscoroutinefunction(LiveAvatar.stop_take)


@pytest.mark.asyncio
async def test_empty_upstream_exception_has_visible_reason(model):
    class FailingBackend:
        def start(self, **kwargs):
            pass

        def next(self):
            raise AssertionError()

        def close(self):
            pass

    model._engine._backend = FailingBackend()
    model.state._image = model._path("reference.png")
    model.state._audio = model._path("audio.wav")
    model.state._running = True
    with pytest.raises(TakeFailed) as caught:
        model.generate(await model.process_input())
    assert await model.process_output(StepOutcome(error=caught.value)) is None
    assert model.state._error == "AssertionError"
    assert not model.state._running


@pytest.mark.asyncio
async def test_waits_for_uploads(model):
    with pytest.raises(CommandError):
        await model.start()
    await model.set_avatar_image(image_upload())
    with pytest.raises(CommandError):
        await model.start()
    await model.set_audio(audio_upload())
    assert StateUpdate.from_state(model.state).ready
    await model.set_generation_options(seed=42, max_chunks=2)
    assert not model.state._running
    with pytest.raises(ApplicationError):
        await model.process_input()
    await model.start()
    assert model.state._running
    with pytest.raises(CommandError):
        await model.set_prompt(prompt="new", negative_prompt="")
    with pytest.raises(CommandError):
        await model.set_generation_options(seed=42, max_chunks=2)
    with pytest.raises(CommandError):
        await model.start()
    await model.stop_take()
    assert model.state._image is not None
    await model.reset()
    assert model.state._image is None and not StateUpdate.from_state(model.state).ready


@pytest.mark.asyncio
async def test_invalid_upload_does_not_replace_input(model):
    await model.set_avatar_image(image_upload())
    original = model.state._image.read_bytes()
    with pytest.raises(CommandError):
        await model.set_avatar_image(UploadedFile("bad.png", "image/png", b"broken"))
    assert model.state._image.read_bytes() == original
    await model.set_audio(audio_upload())
    original = model.state._audio.read_bytes()
    with pytest.raises(CommandError):
        await model.set_audio(audio_upload(0.2))
    assert model.state._audio.read_bytes() == original


@pytest.mark.asyncio
async def test_inference_one_clip_per_turn(model):
    class Backend:
        def __init__(self):
            self.calls = 0

        def start(self, **kwargs):
            self.kwargs = kwargs

        def next(self):
            self.calls += 1
            return (
                (np.zeros((45, 32, 32, 3), np.uint8), np.zeros((1, 86400), np.float32))
                if self.calls == 1
                else None
            )

        def close(self):
            pass

    backend = Backend()
    model._engine._backend = backend
    await model.set_avatar_image(image_upload())
    await model.set_audio(audio_upload())
    await model.set_generation_options(seed=9, max_chunks=3)
    await model.start()
    result = await model.process_output(
        StepOutcome(result=model.generate(await model.process_input()))
    )
    assert isinstance(result, LiveAvatarOutput)
    assert backend.calls == 1
    assert model.state._frames == 45 and model.state._chunks == 1
    assert backend.kwargs["seed"] == 9
    await model.process_output(
        StepOutcome(result=model.generate(await model.process_input()))
    )
    assert not model.state._running


@pytest.mark.asyncio
async def test_reset_keeps_upload_directory_and_emits_one_final_snapshot(model):
    from unittest.mock import AsyncMock

    await model.set_avatar_image(image_upload())
    directory = model.state._directory
    model.send = AsyncMock()
    await model.reset()
    assert model.state._directory is directory
    assert model.send.call_count == 1
    assert not model.send.call_args.args[0].ready
    await model.set_avatar_image(image_upload())
    assert model.state._image.is_file()


@pytest.mark.asyncio
async def test_session_cleanup_and_new_session_defaults(tmp_path):
    from unittest.mock import Mock

    model = LiveAvatar()
    model._work = tmp_path
    model._engine = Mock()
    model.state = LiveAvatarState()
    await model.on_session_started()
    directory = model._path("example").parent
    model.state._seed = 99
    model.state._take_id = 7
    await model.on_session_ended()
    assert not directory.exists()
    # Runtime creates a new InputState for each new session.
    model.state = LiveAvatarState()
    await model.on_session_started()
    assert model.state._seed == 420
    assert model.state._take_id == 0
    assert model.state._image is model.state._audio is None
    await model.on_session_ended()


@pytest.mark.asyncio
async def test_application_hooks_with_fake_model(model):
    from unittest.mock import Mock
    from liveavatar_model import LiveAvatarResult

    model._engine = Mock()
    with pytest.raises(ApplicationError):
        await model.process_input()
    model._engine.generate.assert_not_called()
    await model.set_avatar_image(image_upload())
    await model.set_audio(audio_upload())
    await model.start()
    step = await model.process_input()
    assert step.conditions is not None
    result = LiveAvatarResult(
        step.take_id,
        np.zeros((45, 2, 2, 3), np.uint8),
        np.zeros((1, 86400), np.float32),
        1,
        45,
        False,
    )
    model._engine.generate.return_value = result
    assert model.generate(step) is result
    await model.process_output(StepOutcome(result=result))
    assert (await model.process_input()).conditions is None


@pytest.mark.asyncio
async def test_audio_timeout_preserves_selection_and_cleans_upload(model, monkeypatch):
    import subprocess

    await model.set_audio(audio_upload())
    original = model.state._audio.read_bytes()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(CommandError, match="timed out"):
        await model.set_audio(audio_upload())
    assert model.state._audio.read_bytes() == original
    assert not model._path("audio-upload").exists()


@pytest.mark.asyncio
async def test_oversized_image_refused_before_conversion(model, monkeypatch):
    from unittest.mock import MagicMock

    image = image_upload()
    decoded = MagicMock(width=10000, height=10000)
    decoded.__enter__.return_value = decoded
    monkeypatch.setattr(Image, "open", lambda _: decoded)
    with pytest.raises(CommandError, match="40 million"):
        await model.set_avatar_image(image)
    decoded.convert.assert_not_called()
    assert model.state._image is None
