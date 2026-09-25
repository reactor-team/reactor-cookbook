import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from reactor_runtime import CommandError, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from liveavatar_pipeline import LiveAvatar
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
    model = LiveAvatar()
    model.state = LiveAvatarState()
    model._directory = type("Directory", (), {"name": str(tmp_path)})()
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
        def next(self):
            raise AssertionError()

        def close(self):
            pass

    model._backend = FailingBackend()
    model.state._running = True
    generator = model.inference()
    assert await anext(generator) is None
    assert model.state._error == "AssertionError"
    assert not model.state._running
    await generator.aclose()


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
    assert not model.state._running and not model._pending
    idle = model.inference()
    assert await anext(idle) is None
    await idle.aclose()
    await model.start()
    assert model.state._running
    with pytest.raises(CommandError):
        await model.set_prompt(prompt="new", negative_prompt="")
    with pytest.raises(CommandError):
        await model.set_generation_options(seed=42, max_chunks=2)
    with pytest.raises(CommandError):
        await model.start()
    await model.stop_take()
    assert model._image is not None
    await model.reset()
    assert model._image is None and not StateUpdate.from_state(model.state).ready


@pytest.mark.asyncio
async def test_invalid_upload_does_not_replace_input(model):
    await model.set_avatar_image(image_upload())
    original = model._image.read_bytes()
    with pytest.raises(CommandError):
        await model.set_avatar_image(UploadedFile("bad.png", "image/png", b"broken"))
    assert model._image.read_bytes() == original
    await model.set_audio(audio_upload())
    original = model._audio.read_bytes()
    with pytest.raises(CommandError):
        await model.set_audio(audio_upload(0.2))
    assert model._audio.read_bytes() == original


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

    model._backend = Backend()
    await model.set_avatar_image(image_upload())
    await model.set_audio(audio_upload())
    await model.set_generation_options(seed=9, max_chunks=3)
    await model.start()
    gen = model.inference()
    result = await anext(gen)
    assert isinstance(result, LiveAvatarOutput)
    assert model._backend.calls == 1
    assert model.state._frames == 45 and model.state._chunks == 1
    assert model._backend.kwargs["seed"] == 9
    await anext(gen)
    assert not model.state._running
    await gen.aclose()
