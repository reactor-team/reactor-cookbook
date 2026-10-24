import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from inworld import InWorld
from inworld_images import decode_reference, validate_video
from inworld_model import InWorldAnchor, InWorldInput, InWorldModel, InWorldResult
from inworld_types import InWorldState


def test_playout_preserves_native_audio_video_timing():
    assert InWorld.fps == 24
    model, _ = app()
    assert model.output.fps == 24
    # The Runtime step loop omits compute_time for author-pinned FPS.
    assert model._playout_rate(120, None) == 24.0


def test_multirank_preparation_error_releases_followers(tmp_path):
    model = InWorldModel()
    model.world_size = 4
    model._settings = SimpleNamespace(scratch=tmp_path)
    model._backend = Mock()
    with pytest.raises(ValueError, match="anchor"):
        model.generate(InWorldInput(world_id=1, prompt="garden"))
    model._backend.begin_step.assert_called_once()
    model._backend.cancel_unsubmitted_step.assert_called_once()
    assert model._chunk_index == 0


def test_backend_cancels_only_before_dispatch():
    from inworld_backend import InWorldBackend

    backend = InWorldBackend.__new__(InWorldBackend)
    backend._broadcast = Mock()
    backend.begin_step()
    backend.cancel_unsubmitted_step("invalid input")
    backend.cancel_unsubmitted_step("duplicate")
    backend._broadcast.assert_called_once_with(("error", "invalid input"))


def test_follower_propagates_preparation_error_without_inference():
    from inworld_backend import InWorldBackend

    backend = InWorldBackend.__new__(InWorldBackend)
    backend._broadcast = Mock(return_value=("error", "invalid input"))
    backend.worker = Mock()
    with pytest.raises(RuntimeError, match="invalid input"):
        backend.follow_step()
    backend.worker.execute_forward.assert_not_called()


def test_unknown_attention_backend_refused_before_asset_download(tmp_path):
    from inworld_assets import prepare_assets

    config = tmp_path / "inworld.yaml"
    config.write_text("attention_backend: approximate\n")
    with pytest.raises(ValueError, match="attention_backend"):
        prepare_assets(tmp_path / "weights", config)
    assert not (tmp_path / "weights").exists()


def test_cached_prefix_preserves_decoded_yuv(tmp_path):
    from collections import deque

    from inworld_backend import write_prefix

    rng = np.random.default_rng(42)
    source = [rng.integers(0, 256, (48, 32), dtype=np.uint8) for _ in range(17)]
    frames = deque(av.VideoFrame.from_ndarray(x, format="yuv420p") for x in source)
    target = tmp_path / "prefix.mp4"
    write_prefix(target, frames)
    with av.open(str(target)) as container:
        assert container.streams.video[0].average_rate == 24
        decoded = [f.to_ndarray(format="yuv420p") for f in container.decode(video=0)]
    assert len(decoded) == 17
    assert np.array_equal(np.stack(decoded), np.stack(source))


def test_cached_prefix_rejects_incomplete_history(tmp_path):
    from collections import deque

    from inworld_backend import write_prefix

    with pytest.raises(ValueError, match="17 frames"):
        write_prefix(tmp_path / "prefix.mp4", deque())


@pytest.mark.parametrize("prefix", [0, 17])
def test_parallel_clip_decode_preserves_frames_and_audio(tmp_path, prefix):
    from fractions import Fraction

    from inworld_backend import decode_clip

    path = tmp_path / "clip.mkv"
    rng = np.random.default_rng(7)
    with av.open(str(path), "w") as output:
        video = output.add_stream("ffv1", rate=24)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        sound = output.add_stream("pcm_f32le", rate=48000)
        sound.layout = "stereo"
        for i in range(21):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        samples = rng.uniform(-0.1, 0.1, (1, (21 * 2000 - 1000) * 2)).astype(np.float32)
        audio = av.AudioFrame.from_ndarray(samples, format="flt", layout="stereo")
        audio.sample_rate, audio.pts = 48000, 0
        for packet in sound.encode(audio):
            output.mux(packet)
        for packet in sound.encode():
            output.mux(packet)
    with av.open(str(path)) as media:
        expected = np.stack(
            [f.to_ndarray(format="rgb24") for f in media.decode(video=0)]
        )
    frames, audio = decode_clip(path, prefix, tail_path=tmp_path / "tail.mp4")
    assert frames.flags.c_contiguous
    assert np.array_equal(frames, expected[prefix:])
    expected_audio = np.pad(samples.reshape(-1, 2).T, ((0, 0), (0, 1000)))
    assert np.array_equal(audio, expected_audio[:, prefix * 2000 :])
    with pytest.raises(RuntimeError, match="no new video frames"):
        decode_clip(path, 21)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_uint8_processor_preserves_native_quantization(dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.video_processor import VideoProcessor

    from inworld_backend import Uint8VideoProcessor

    native = VideoProcessor(vae_scale_factor=32)
    # Include clipping and quantization boundaries in addition to random values.
    values = torch.linspace(-1.1, 1.1, 2 * 3 * 17 * 8 * 8)
    video = values.reshape(2, 3, 17, 8, 8).to(getattr(torch, dtype))
    original = native.postprocess_video(video, output_type="np")
    expected = (np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8)
    optimized = Uint8VideoProcessor(native).postprocess_video(video)
    assert optimized.dtype == np.uint8
    assert optimized.flags.c_contiguous
    assert np.array_equal(optimized, expected)
    assert torch.equal(
        Uint8VideoProcessor(native).postprocess_video(video, output_type="pt"),
        native.postprocess_video(video, output_type="pt"),
    )


def app():
    instance = InWorld()
    instance.state = InWorldState()
    instance._engine = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    instance.send = send
    return instance, messages


def result(world=1, index=1):
    return InWorldResult(
        world_id=world,
        chunk_index=index,
        frames=np.zeros((1, 32, 32, 3), np.uint8),
        audio=np.zeros((2, 2000), np.float32),
    )


def test_refused_step_never_reaches_model():
    model, _ = app()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model._engine.generate.assert_not_called()


def test_refused_prompt_sends_nothing():
    model, messages = app()
    with pytest.raises(CommandError):
        asyncio.run(model.set_prompt("  "))
    assert not messages


def test_generate_reads_only_its_snapshot():
    model, _ = app()
    snapshot = InWorldInput(world_id=7, prompt="before", anchor=InWorldAnchor())
    model.state = None
    model._engine.generate.return_value = result(7)
    assert model.generate(snapshot).world_id == 7
    model._engine.generate.assert_called_once_with(snapshot)


def test_anchor_until_ack_and_dynamic_prompt_preserves_world():
    model, _ = app()
    asyncio.run(model.set_prompt("a garden"))
    asyncio.run(model.start(seed=42, new_frames=120))
    first = asyncio.run(model.process_input())
    assert first.anchor is not None
    assert asyncio.run(model.process_input()).anchor is first.anchor
    asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=1.0)))
    assert asyncio.run(model.process_input()).anchor is None
    model.state.paused = True
    asyncio.run(model.set_prompt("rain falls in the same garden"))
    assert model.state.paused and model.state._world_id == first.world_id
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.state.paused = False
    updated = asyncio.run(model.process_input())
    assert updated.prompt.startswith("rain") and updated.anchor is None


def test_failure_does_not_ack_or_count():
    model, messages = app()
    failure = RuntimeError("native failed")
    with pytest.raises(RuntimeError, match="native failed"):
        asyncio.run(model.process_output(StepOutcome(error=failure, elapsed=1.0)))
    assert model.state._completed == 0 and model.state._applied_world_id is None
    assert not messages


def test_fresh_world_and_session_defaults():
    model, _ = app()
    asyncio.run(model.set_prompt("garden"))
    asyncio.run(model.start(seed=99, new_frames=120))
    asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=1.0)))
    asyncio.run(model.start(seed=42, new_frames=120))
    assert model.state._world_id == 2 and model.state._completed == 0
    assert asyncio.run(model.process_input()).anchor.seed == 42
    model.on_session_ended()
    model._engine.reset.assert_called_once()
    fresh = InWorldState()
    assert not fresh._references and fresh._anchor is None and fresh.paused


def test_missing_reference_slot_is_refused():
    model, _ = app()
    model.state.prompt = "a scene"
    model.state._references = {2: np.zeros((32, 32, 3), np.uint8)}
    with pytest.raises(CommandError):
        asyncio.run(model.start(seed=42, new_frames=120))
    assert model.state._world_id == 0


@pytest.mark.parametrize("payload", [b"", b"not media"])
def test_invalid_media(payload):
    with pytest.raises((ValueError, OSError)):
        decode_reference(payload)
    with pytest.raises((ValueError, av.FFmpegError)):
        validate_video(payload)


def test_model_owns_parent_and_failed_step_does_not_advance(tmp_path):
    model = InWorldModel()
    model._settings = SimpleNamespace(scratch=tmp_path)

    class Backend:
        def store_references(self, root, refs):
            return ()

        def generate_clip(self, *, root, index, prompt, anchor, parent, references):
            if prompt == "fail":
                raise RuntimeError("native failed")
            path = root / f"{index}.mp4"
            path.write_bytes(b"fake media")
            return path, result().frames, result().audio

    model._backend = Backend()
    with pytest.raises(ValueError, match="anchor"):
        model.generate(InWorldInput(world_id=1, prompt="garden"))
    first = model.generate(
        InWorldInput(world_id=1, prompt="garden", anchor=InWorldAnchor())
    )
    assert first.world_id == 1 and first.chunk_index == 1
    parent = model._parent
    with pytest.raises(RuntimeError):
        model.generate(InWorldInput(world_id=1, prompt="fail"))
    assert model._chunk_index == 1 and model._parent == parent
    assert model.generate(InWorldInput(world_id=1, prompt="rain")).chunk_index == 2
    assert not parent.exists()
    assert (
        model.generate(
            InWorldInput(world_id=2, prompt="sun", anchor=InWorldAnchor())
        ).chunk_index
        == 1
    )
    model.reset()
    assert model._parent is None and model._world_id is None


def test_model_half_has_no_runtime_imports():
    root = Path(__file__).resolve().parents[1]
    for name in ("inworld_model.py", "inworld_backend.py"):
        assert "reactor_runtime" not in (root / name).read_text()


def test_image_upload_stages_pixels_without_starting():
    model, _ = app()
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (10, 20, 30)).save(buffer, format="PNG")
    upload = UploadedFile(
        name="test.png", mime_type="image/png", data=buffer.getvalue()
    )
    asyncio.run(model.set_reference(image=upload, slot=1))
    assert model.state._references[1].shape == (32, 64, 3)
    assert model.state._references[1][0, 0].tolist() == [10, 20, 30]
    assert model.state.paused and model.state._anchor is None
    model._engine.generate.assert_not_called()


def test_oversized_image_refused_before_decode():
    with pytest.raises(ValueError, match="25 MiB"):
        decode_reference(bytes(25 * 1024 * 1024 + 1))


def test_real_step_loop_delivers_model_error_without_completion():
    async def check():
        model, _ = app()
        model._on_loop_ready()
        model._live.set()
        model.state.prompt = "garden"
        await model.start(seed=42, new_frames=120)
        model._engine.generate.side_effect = RuntimeError("real loop failure")
        with pytest.raises(RuntimeError, match="real loop failure"):
            await asyncio.wait_for(model.run(), timeout=2)
        assert model.state._completed == 0
        assert model.state._applied_world_id is None
        model._engine.generate.assert_called_once()

    asyncio.run(check())


def test_real_step_loop_refuses_paused_input():
    async def check():
        model, _ = app()
        model._on_loop_ready()
        model._live.set()
        task = asyncio.create_task(model.run())
        await asyncio.sleep(0.025)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        model._engine.generate.assert_not_called()

    asyncio.run(check())
