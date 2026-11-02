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
from inworld_model import (
    FPS,
    PREFIX_FRAMES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    InWorldAnchor,
    InWorldInput,
    InWorldModel,
    InWorldResult,
)
from inworld_types import InWorldAudio, InWorldState


def test_playout_preserves_native_audio_video_timing():
    assert InWorld.fps == 24
    assert "fps" in vars(InWorld)
    assert InWorldAudio.sample_rate == InWorld.fps * SAMPLES_PER_FRAME == SAMPLE_RATE
    assert SAMPLE_RATE % FPS == 0 and PREFIX_FRAMES == 17


def test_multirank_preparation_error_releases_followers(tmp_path):
    model = InWorldModel()
    model.world_size = 4
    model._settings = SimpleNamespace()
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
@pytest.mark.parametrize("missing", [1000, 2160, 2401])
def test_parallel_clip_decode_preserves_frames_and_audio(tmp_path, prefix, missing):
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
        samples = rng.uniform(-0.1, 0.1, (1, (21 * 2000 - missing) * 2)).astype(
            np.float32
        )
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
    if missing > SAMPLE_RATE // 20:
        with pytest.raises(RuntimeError, match="over 50ms"):
            decode_clip(path, prefix)
        return
    frames, audio = decode_clip(path, prefix, tail_path=tmp_path / "tail.mp4")
    assert frames.flags.c_contiguous
    assert np.array_equal(frames, expected[prefix:])
    stereo = samples.reshape(-1, 2).T
    expected_audio = np.pad(
        (stereo.sum(axis=0, keepdims=True) * np.float32(2**-0.5)),
        ((0, 0), (0, missing)),
    )
    assert audio.shape == (1, (21 - prefix) * SAMPLES_PER_FRAME)
    assert audio.dtype == np.float32
    expected_audio = expected_audio[:, prefix * SAMPLES_PER_FRAME :]
    fade = SAMPLE_RATE // 200
    np.testing.assert_allclose(
        audio[:, fade : -missing - fade],
        expected_audio[:, fade : -missing - fade],
        atol=2e-8,
    )
    assert audio[0, -missing - 1] == 0 and np.all(audio[:, -missing:] == 0)
    if prefix:
        assert audio[0, 0] == 0
    else:
        assert np.isclose(audio[0, 0], expected_audio[0, 0], atol=2e-8)
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
        audio=np.zeros((1, SAMPLES_PER_FRAME), np.float32),
    )


def test_refused_step_never_reaches_model():
    model, _ = app()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model._engine.generate.assert_not_called()


def test_refused_prompt_sends_nothing():
    model, messages = app()
    with pytest.raises(CommandError) as exc:
        asyncio.run(model.set_prompt("  "))
    assert exc.value.code == "prompt_required"
    assert not messages


def test_generate_reads_only_its_snapshot():
    model, _ = app()
    snapshot = InWorldInput(
        world_id=7, prompt="before", anchor=InWorldAnchor(seed=42, new_frames=120)
    )
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
    model._settings = SimpleNamespace()

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
        InWorldInput(
            world_id=1, prompt="garden", anchor=InWorldAnchor(seed=42, new_frames=120)
        )
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
            InWorldInput(
                world_id=2, prompt="sun", anchor=InWorldAnchor(seed=42, new_frames=120)
            )
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


@pytest.mark.parametrize(
    "handler,code", [("set_reference", "invalid_image"), ("set_video", "invalid_video")]
)
def test_invalid_upload_refuses_through_handler_without_broadcast(handler, code):
    model, messages = app()
    upload = UploadedFile(
        name="bad", mime_type="application/octet-stream", data=b"not media"
    )
    with pytest.raises(CommandError) as exc:
        if handler == "set_reference":
            asyncio.run(model.set_reference(upload, slot=1))
        else:
            asyncio.run(model.set_video(upload))
    assert exc.value.code == code and not messages
    assert "0x" not in str(exc.value) and "BytesIO" not in str(exc.value)


def test_clear_media_pause_and_joiner_snapshot():
    model, messages = app()
    model.state._video = b"video"
    model.state._references = {1: np.zeros((2, 2, 3), np.uint8)}
    assert asyncio.run(model.clear_media()).kind == "clear"
    assert model.state._video is None and not model.state._references
    assert asyncio.run(model.set_paused(False)).kind == "paused"
    assert not model.state.paused
    before = len(messages)
    received = []

    async def send(value):
        received.append(value)

    asyncio.run(model.on_connected(SimpleNamespace(send=send)))
    assert len(received) == 1 and len(messages) == before


def test_unhealthy_runner_teardown_preserves_original_error():
    model, _ = app()
    model._engine.healthy = False
    model._engine.reset.side_effect = RuntimeError("secondary")
    model.on_session_ended()
    model._engine.reset.assert_not_called()


def test_replaced_world_result_is_ignored():
    model, messages = app()
    model.state.prompt = "garden"
    asyncio.run(model.start(seed=42, new_frames=120))
    asyncio.run(model.start(seed=77, new_frames=120))
    count = len(messages)
    assert (
        asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=1)))
        is None
    )
    assert model.state._completed == 0 and model.state._applied_world_id is None
    assert len(messages) == count


@pytest.mark.parametrize("rank", [0, 1])
def test_load_warmup_covers_first_and_continuation_for_every_length(rank):
    from inworld_backend import InWorldBackend
    from inworld_model import CHUNK_FRAMES

    backend = InWorldBackend.__new__(InWorldBackend)
    backend.rank, backend._cuda_graph = rank, True
    backend.follow_step = Mock()
    calls = []

    def generate_clip(**kwargs):
        calls.append(kwargs)
        assert kwargs["root"].is_dir()
        assert kwargs["warmup"] is True
        assert (kwargs["parent"] is None) == (kwargs["index"] == 1)
        return kwargs["root"] / "clip.mp4", None, None

    backend.generate_clip = generate_clip
    backend.warmup()
    if rank:
        assert not calls and backend.follow_step.call_count == 2 * len(CHUNK_FRAMES)
    else:
        assert [v["anchor"].new_frames for v in calls] == [
            n for n in (120, *(x for x in CHUNK_FRAMES if x != 120)) for _ in (1, 2)
        ]
        assert all(not v["root"].exists() for v in calls)


def test_video_validation_seeks_last_gop_instead_of_decoding_whole_h264(monkeypatch):
    import inworld_images

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=FPS)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "12", "bf": "2"}
        for i in range(240):
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), i % 256, np.uint8), format="rgb24"
                )
            ):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    original = av.open
    decoded = []

    class Observed:
        def __init__(self, *args, **kwargs):
            self.container = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.container, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def decode(self, *args, **kwargs):
            for frame in self.container.decode(*args, **kwargs):
                decoded.append(frame.pts)
                yield frame

    monkeypatch.setattr(inworld_images.av, "open", Observed)
    validate_video(buffer.getvalue())
    assert PREFIX_FRAMES <= len(decoded) < 40


def test_serving_request_is_not_marked_as_warmup(monkeypatch, tmp_path):
    import sys

    import inworld_backend
    from inworld_backend import InWorldBackend
    from inworld_model import HEIGHT, WIDTH

    backend = InWorldBackend.__new__(InWorldBackend)
    backend.args = object()
    backend._cuda_graph = True
    request = SimpleNamespace(is_warmup=True)
    params = SimpleNamespace(
        prepare_video_request_for_queue=Mock(), cleanup_video_request=Mock()
    )
    monkeypatch.setitem(
        sys.modules, "murea_fast.client", SimpleNamespace(prepare_tail=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.entrypoints.openai.utils",
        SimpleNamespace(build_sampling_params=lambda *args, **kwargs: params),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.entrypoints.utils",
        SimpleNamespace(prepare_request=lambda *args: request),
    )

    def execute(requests):
        assert requests == [request] and request.is_warmup is False
        return SimpleNamespace(
            error=None, output_file_paths=[str(tmp_path / "clip.mp4")]
        )

    backend._execute = execute
    frames = np.broadcast_to(np.zeros((1, 1, 1, 3), np.uint8), (25, HEIGHT, WIDTH, 3))
    monkeypatch.setattr(
        inworld_backend,
        "decode_clip",
        lambda *args, **kwargs: (
            frames,
            np.zeros((1, 25 * SAMPLES_PER_FRAME), np.float32),
        ),
    )
    backend.generate_clip(
        root=tmp_path,
        index=1,
        prompt="garden",
        anchor=InWorldAnchor(seed=42, new_frames=24),
        parent=None,
        references=(),
    )
    params.cleanup_video_request.assert_called_once_with(request)


def test_prepared_connectors_are_verified_without_materialization_lock(tmp_path):
    from inworld_assets import prepare_connectors

    source = tmp_path / "source"
    script = source / "inference/scripts/materialize_connectors.py"
    script.parent.mkdir(parents=True)
    script.write_text("""
CHECKPOINT_SHA256 = "known"
def valid_existing(*args, **kwargs):
    assert not args[1].is_symlink()
    assert kwargs["verify_hashes"] is True
    return True
def sha256(path): return "known"
def fingerprint(path): return {}
def load_json(path): return {"checkpoint_fingerprint": {}}
if __name__ == "__main__":
    raise RuntimeError("Prepared connectors must not take the write lock")
""")
    checkpoint, base, output = (
        tmp_path / "checkpoint",
        tmp_path / "base",
        tmp_path / "overlay",
    )
    target = tmp_path / "checkpoint-blob"
    target.touch()
    checkpoint.symlink_to(target)
    base.mkdir()
    output.mkdir()
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    prepare_connectors(source, checkpoint, base, output)
    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    # Importing the source can create __pycache__, outside the weight directory.
    assert all(
        after[p] == t
        for p, t in before.items()
        if p not in (source / "inference/scripts",)
    )
    assert not list(output.iterdir())


def test_model_refuses_frame_count_outside_served_profile():
    model = InWorldModel()
    model._backend = Mock()
    model._settings = SimpleNamespace()
    with pytest.raises(ValueError, match="offered chunk lengths"):
        model.generate(
            InWorldInput(
                world_id=1,
                prompt="garden",
                anchor=InWorldAnchor(seed=42, new_frames=464),
            )
        )
    model._backend.generate_clip.assert_not_called()
