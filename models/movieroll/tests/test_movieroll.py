"""Drive the real hooks and model bookkeeping with CPU values and fake compute."""

import asyncio
import dataclasses
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from movieroll_app import MovieRoll
from movieroll_assets import REVISIONS, prepare_weights
from movieroll_history import LatentHistory
from movieroll_images import ImageError, decode_image
from movieroll_model import (
    MAX_PROMPTS,
    Anchor,
    HistoryFullError,
    MissingAnchorError,
    MovieRollInput,
    MovieRollModel,
    MovieRollResult,
)
from movieroll_types import MovieRollState


@pytest.mark.parametrize("gpu_count", [1, 4])
def test_application_delegates_gpu_ranks_and_collectives_to_runner(
    monkeypatch, tmp_path, gpu_count
):
    import movieroll_app
    from movieroll_model import Settings

    settings = Settings(tmp_path, tmp_path, tmp_path, 1024, gpu_count)
    monkeypatch.setattr(movieroll_app, "prepare", Mock(return_value=settings))
    monkeypatch.setattr(movieroll_app, "get_weights_path", Mock(return_value=tmp_path))
    runner = Mock()
    monkeypatch.setattr(movieroll_app, "DistributedRunner", runner)
    service = MovieRoll()
    service.load(None)
    assert runner.call_args.kwargs["world_size"] == gpu_count
    assert runner.call_args.kwargs["init_process_group"] == (gpu_count > 1)
    assert runner.call_args.kwargs["load_kwargs"]["settings"] == settings
    runner.return_value.start.assert_called_once_with()


def test_worker_count_mismatch_is_rejected_before_native_import(tmp_path):
    from movieroll_model import Settings

    worker = MovieRollModel()
    with pytest.raises(ValueError, match="Worker count"):
        worker.load(Settings(tmp_path, tmp_path, tmp_path, 1024, 4))


def test_four_gpu_deployment_override_resolves_into_worker_settings(
    tmp_path, monkeypatch
):
    import movieroll_assets

    config = tmp_path / "settings.yaml"
    config.write_text(json.dumps({"source_path": str(tmp_path), "gpu_count": 1}))
    monkeypatch.setenv("MOVIEROLL_GPU_COUNT", "4")
    monkeypatch.setattr(
        movieroll_assets.subprocess,
        "check_output",
        Mock(return_value=movieroll_assets.SOURCE_REVISION),
    )
    prepare_assets = Mock()
    monkeypatch.setattr(movieroll_assets, "prepare_weights", prepare_assets)
    settings = movieroll_assets.prepare(config, tmp_path)
    assert settings.gpu_count == 4
    assert settings.sessions.is_dir()
    prepare_assets.assert_called_once_with(tmp_path.resolve())


@pytest.mark.parametrize("gpu_count", [0, 2, 8, True, "4"])
def test_invalid_gpu_config_is_rejected_before_asset_preparation(
    tmp_path, monkeypatch, gpu_count
):
    import movieroll_assets

    config = tmp_path / "settings.yaml"
    config.write_text(
        json.dumps({"source_path": str(tmp_path), "gpu_count": gpu_count})
    )
    monkeypatch.delenv("MOVIEROLL_GPU_COUNT", raising=False)
    monkeypatch.setattr(
        movieroll_assets.subprocess,
        "check_output",
        Mock(return_value=movieroll_assets.SOURCE_REVISION),
    )
    forbidden = Mock(side_effect=AssertionError("must reject before downloads"))
    monkeypatch.setattr(movieroll_assets, "prepare_weights", forbidden)
    with pytest.raises(ValueError, match="gpu_count"):
        movieroll_assets.prepare(config, tmp_path)
    forbidden.assert_not_called()


def run(awaitable):
    return asyncio.run(awaitable)


def app():
    service = MovieRoll()
    service.state = MovieRollState()
    service.engine = Mock()
    service.output = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    service.send = send
    return service, messages


def result(world=1, index=1):
    return MovieRollResult(
        world_id=world,
        chunk_index=index,
        frames=np.zeros((2, 480, 864, 3), dtype=np.uint8),
    )


def image_bytes(size=(1024, 768), format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, "#2676b0").save(output, format=format)
    return output.getvalue()


def test_real_loop_refuses_then_routes_failure_without_success():
    import threading

    from reactor_runtime.core.model import ClientConnected, SessionStarted
    from reactor_runtime.core.values import ConnId

    service, _ = app()
    failure = RuntimeError("native failed")
    service.engine.generate.side_effect = failure
    failures, outcomes, media = [], [], []
    done = threading.Event()
    process_output = service.process_output

    async def output(outcome):
        outcomes.append(outcome)
        return await process_output(outcome)

    def failed(error):
        failures.append(error)
        done.set()

    service.process_output = output
    service.bind_output(
        broadcast=lambda *args: None,
        addressed=lambda *args: None,
        media=lambda *args: media.append(args),
    )
    service.bind_failure(failed)
    service.start_thread()
    try:
        service.post_reactor_event(SessionStarted("movieroll-cpu-test"))
        service.post_reactor_event(ClientConnected(ConnId(1), total=1))
        command = ModelContract.of(MovieRoll).validate(
            "start", {"prompt": "A garden", "seed": 42, "mode": "t2v"}
        )
        service.submit_command(command, ConnId(1), None)
        assert done.wait(5), (
            "The real Runtime loop did not report the failed model step"
        )
        assert failures == [failure] and not media
        assert len(outcomes) == 1 and outcomes[0].error is failure
        assert service.state._completed == 0 and service.state._applied_world_id is None
    finally:
        service.stop()


def test_refused_step_never_reaches_model_and_forward_reads_only_input():
    service, _ = app()
    with pytest.raises(ApplicationError):
        run(service.process_input())
    run(service.start("A garden", 42, "t2v"))
    run(service.set_paused(True))
    with pytest.raises(ApplicationError):
        run(service.process_input())
    service.engine.generate.assert_not_called()
    request = MovieRollInput(world_id=4, prompt="old caption", anchor=None)
    service.state = None
    service.generate(request)
    service.engine.generate.assert_called_once_with(request)


def test_anchor_repeats_until_result_acknowledges_and_time_is_outcome_elapsed():
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    first = run(service.process_input())
    assert first.anchor == Anchor(seed=42, mode="t2v", image=None)
    assert run(service.process_input()).anchor is first.anchor
    out = run(service.process_output(StepOutcome(result=result(), elapsed=0.3)))
    assert out.main_video.shape == (2, 480, 864, 3)
    assert run(service.process_input()).anchor is None
    assert messages[-1].completed_chunks == 1 and messages[-1].last_chunk_seconds == 0.3


@pytest.mark.parametrize("transition", ["start", "reset", "stop"])
def test_late_output_cannot_acknowledge_or_change_replaced_world(transition):
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    if transition == "start":
        run(service.start("A beach", 7, "t2v"))
    elif transition == "reset":
        run(service.reset(None))
    else:
        run(service.stop_video())
    before = service._status()
    messages.clear()
    assert run(service.process_output(StepOutcome(result=result(), elapsed=1))) is None
    assert service._status() == before and not messages


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("start", (" ", 0, "t2v"), "invalid_prompt"),
        ("start", ("A garden", 0, "i2v"), "image_required"),
        ("set_prompt", (" ",), "invalid_prompt"),
        ("set_paused", (True,), "no_video"),
        ("reset", (None,), "no_video"),
    ],
)
def test_command_refusal_is_caller_only_and_changes_nothing(name, args, code):
    service, messages = app()
    before = service._status()
    with pytest.raises(CommandError) as error:
        run(getattr(service, name)(*args))
    assert error.value.code == code
    assert service._status() == before and not messages
    service.engine.generate.assert_not_called()


def test_live_caption_preserves_world_pause_seed_history_and_media():
    service, _ = app()
    run(service.start("A garden", 42, "t2v"))
    run(service.process_output(StepOutcome(result=result(index=3), elapsed=0.3)))
    run(service.set_paused(True))
    service.output.flush.reset_mock()
    for caption in ["The garden in light rain", "Sunshine returns to the garden"]:
        status = run(service.set_prompt(caption))
        assert status.world_id == 1 and status.completed_chunks == 3
        assert status.paused and status.seed == 42
        with pytest.raises(ApplicationError):
            run(service.process_input())
    run(service.set_paused(False))
    request = run(service.process_input())
    assert request.prompt == caption and request.anchor is None
    service.output.flush.assert_not_called()
    service.engine.reset.assert_not_called()


def test_native_event_limit_refuses_without_replacing_accepted_prompt():
    service, messages = app()
    run(service.start("event 0", 0, "t2v"))
    for index in range(1, MAX_PROMPTS):
        run(service.set_prompt(f"event {index}"))
    before = service._status()
    messages.clear()
    with pytest.raises(CommandError) as error:
        run(service.set_prompt("one too many"))
    assert error.value.code == "prompt_limit"
    assert service._status() == before and not messages
    assert (1 + MAX_PROMPTS + 3 + 5) * 4 + 1 <= 120


def test_upload_stages_without_start_and_reset_keeps_original_image():
    service, _ = app()
    upload = UploadedFile(name="scene.png", mime_type="image/png", data=image_bytes())
    run(service.set_image(upload))
    assert service.state._image.shape == (480, 864, 3)
    assert service.state._anchor is None
    run(service.start("A coastal scene", 42, "i2v"))
    first_image = service.state._anchor.image
    run(
        service.set_image(
            UploadedFile(
                name="other.png", mime_type="image/png", data=image_bytes((864, 480))
            )
        )
    )
    run(service.reset(None))
    assert service.state._anchor.image is first_image
    assert service.state._anchor.seed == 42
    run(service.clear_image())
    assert service.state._image is None and service.state._anchor.image is first_image
    run(service.reset(7))
    assert (
        service.state._anchor.seed == 7 and service.state._anchor.image is first_image
    )


def test_bad_upload_while_paused_preserves_accepted_inputs_and_pause():
    service, messages = app()
    run(
        service.set_image(
            UploadedFile(name="scene.png", mime_type="image/png", data=image_bytes())
        )
    )
    run(service.start("A garden", 42, "i2v"))
    run(service.set_paused(True))
    before, image = service._status(), service.state._image
    messages.clear()
    with pytest.raises(CommandError) as error:
        run(
            service.set_image(
                UploadedFile(name="broken.png", mime_type="image/png", data=b"broken")
            )
        )
    assert error.value.code == "invalid_image"
    assert (
        service.state._image is image and service._status() == before and not messages
    )


@pytest.mark.parametrize(
    "data,mime,code",
    [
        (b"", "image/png", "image_size"),
        (b"bad", "video/mp4", "unsupported_image"),
        (b"bad", "image/png", "invalid_image"),
        (image_bytes((8, 64)), "image/png", "invalid_image"),
        (image_bytes(), "image/jpeg", "invalid_image"),
        (image_bytes((16, 4096)), "image/png", "image_size"),
    ],
)
def test_image_validation(data, mime, code):
    with pytest.raises(ImageError) as error:
        decode_image(data, mime)
    assert error.value.code == code


@pytest.mark.parametrize(
    "format,mime",
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_every_advertised_image_format_is_accepted_by_upload_hook(format, mime):
    # JPEG has no is_animated attribute; valid photos must not become hook errors.
    data = io.BytesIO()
    Image.new("RGB", (320, 512), (20, 80, 140)).save(data, format)
    service, _ = app()
    status = run(
        service.set_image(
            UploadedFile(name="photo", mime_type=mime, data=data.getvalue())
        )
    )
    assert status.has_image and not status.active
    assert service.state._image.shape == (480, 864, 3)
    assert service.state._image.dtype == np.uint8
    assert np.max(np.abs(service.state._image[0, 0].astype(int) - (20, 80, 140))) <= 2


def test_large_image_rejected_before_decompression(monkeypatch):
    from movieroll_images import MAX_IMAGE_BYTES

    with pytest.raises(ImageError) as error:
        decode_image(b"x" * (MAX_IMAGE_BYTES + 1), "image/png")
    assert error.value.code == "image_size"
    opened = Mock(width=8000, height=8000, format="PNG")
    opened.__enter__ = Mock(return_value=opened)
    opened.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(Image, "open", Mock(return_value=opened))
    with pytest.raises(ImageError) as error:
        decode_image(b"encoded", "image/png")
    assert error.value.code == "image_size"
    opened.load.assert_not_called()


def test_join_is_caller_only_and_session_end_clears_only_healthy_model():
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    run(service.set_paused(True))
    messages.clear()
    delivered = []

    async def send(message):
        delivered.append(message)

    run(service.on_connected(SimpleNamespace(send=send)))
    assert delivered == [service._status()] and not messages
    assert delivered[0].paused and delivered[0].world_id == 1
    assert not hasattr(service, "on_disconnected")
    service.engine.healthy = False
    service.on_session_ended()
    service.engine.reset.assert_not_called()
    service.engine.healthy = True
    service.on_session_ended()
    service.engine.reset.assert_called_once()
    service.state = MovieRollState()
    assert service._status().seed is None and not service._status().active


def test_model_error_does_not_count_and_explicit_new_world_is_required():
    model = MovieRollModel()
    model._backend = Mock()
    model._backend.step.return_value = result().frames
    with pytest.raises(MissingAnchorError):
        model.generate(MovieRollInput(world_id=1, prompt="garden", anchor=None))
    anchor = Anchor(seed=42, mode="t2v", image=None)
    request = MovieRollInput(world_id=1, prompt="garden", anchor=anchor)
    assert model.generate(request).chunk_index == 1
    model._backend.begin.assert_called_once_with(anchor, "garden")
    assert (
        model.generate(
            MovieRollInput(world_id=1, prompt="rain", anchor=None)
        ).chunk_index
        == 2
    )
    model._backend.begin.assert_called_once()
    model._backend.step.side_effect = RuntimeError("native failed")
    with pytest.raises(RuntimeError):
        model.generate(request)
    assert model._chunk_index == 2
    model.reset()
    assert model._world_id is None and model._chunk_index == 0


def test_storage_budget_is_explicit_end_and_can_start_again():
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    run(
        service.process_output(StepOutcome(error=HistoryFullError("full"), elapsed=0.1))
    )
    assert not messages[-1].active and messages[-1].end_reason == "history_full"
    assert messages[-1].completed_chunks == 0
    service.engine.reset.assert_called_once()
    run(service.start("A beach", 7, "t2v"))
    assert service._status().active and service._status().end_reason is None


@pytest.mark.parametrize("first_size", [1, 5])
def test_history_preserves_bits_and_random_access_across_recompute_boundaries(
    tmp_path, first_size
):
    archive = LatentHistory(tmp_path, max_bytes=10_000_000)
    directory = archive.path.parent
    rng = np.random.default_rng(7)
    expected = []
    try:
        for index in range(100):
            chunk = rng.integers(
                0, 65536, (1, 2, first_size if index == 0 else 4, 2, 3), dtype=np.uint16
            )
            archive.append(chunk)
            expected.append(chunk)
        for index in (0, 18, 19, 49, 70, 99, -1):
            assert np.array_equal(archive[index], expected[index])
        with pytest.raises(IndexError):
            archive[100]
        assert len(archive) == 100
    finally:
        archive.close()
    assert not directory.exists()


def test_history_budget_refuses_before_writing_extra_chunk(tmp_path):
    archive = LatentHistory(tmp_path, max_bytes=10)
    try:
        archive.append(np.zeros((1, 1, 5, 1, 1), np.uint16))
        with pytest.raises(HistoryFullError):
            archive.append(np.ones((1, 1, 4, 1, 1), np.uint16))
        assert len(archive) == 1 and archive.path.stat().st_size == 10
    finally:
        archive.close()


def test_prepared_relocatable_assets_need_no_network_or_write(tmp_path, monkeypatch):
    target = tmp_path / "weights.bin"
    target.write_bytes(b"prepared")
    (tmp_path / "movieroll-assets.json").write_text(
        json.dumps({"revisions": REVISIONS, "files": {"weights.bin": 8}})
    )
    forbidden = Mock(side_effect=AssertionError("prepared read-only assets"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden)
    )
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    prepare_weights(tmp_path)
    forbidden.assert_not_called()


def test_contract_has_one_control_path_and_moderated_free_content():
    contract = ModelContract.of(MovieRoll)
    names = set(contract.commands)
    assert names == {
        "start",
        "set_prompt",
        "set_paused",
        "set_image",
        "clear_image",
        "reset",
        "stop",
        "get_state",
    }
    assert MovieRollState._public_fields["prompt"].moderate
    assert inspect.signature(MovieRoll.start).parameters["prompt"].default.moderate
    assert inspect.signature(MovieRoll.set_image).parameters["image"].default.moderate
    assert (
        dataclasses.is_dataclass(MovieRollInput)
        and MovieRollInput.__dataclass_params__.frozen
    )
    assert (
        dataclasses.is_dataclass(MovieRollResult)
        and MovieRollResult.__dataclass_params__.frozen
    )


def test_model_dependency_graph_imports_without_runtime_or_native_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; original=builtins.__import__\n"
                "def guarded(name,*args,**kwargs):\n"
                " if name.split('.')[0] in ('reactor_runtime','torch','movieroll','diffusers'):\n"
                "  raise AssertionError('Unexpected model import: '+name)\n"
                " return original(name,*args,**kwargs)\n"
                "builtins.__import__=guarded\n"
                "import movieroll_model,movieroll_backend,movieroll_history,movieroll_attention\n"
            ),
        ],
        check=True,
    )


def test_history_preserves_fp32_sampling_and_bf16_image_anchor(tmp_path):
    # Native DMD scheduler returns FP32, although DiT and the I2V anchor are BF16.
    archive = LatentHistory(tmp_path, max_bytes=100_000)
    anchor = np.arange(12, dtype=np.uint16).reshape(1, 2, 1, 2, 3)
    chunks = [anchor]
    rng = np.random.default_rng(4)
    for _ in range(24):
        chunks.append(rng.standard_normal((1, 2, 4, 2, 3), dtype=np.float32))
    try:
        for chunk in chunks:
            archive.append(chunk)
        for index in (0, 19, 1, 24, 12):
            restored = archive[index]
            assert restored.dtype == chunks[index].dtype
            assert restored.tobytes() == chunks[index].tobytes()
    finally:
        archive.close()


def test_read_snapshot_changes_nothing_and_does_not_broadcast():
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    run(service.set_paused(True))
    before = service._status()
    messages.clear()
    assert service.get_state() == before
    assert service._status() == before and not messages
    service.engine.generate.assert_not_called()
    service.engine.reset.assert_not_called()


def test_old_storage_failure_cannot_end_a_replacement_video():
    service, messages = app()
    run(service.start("A garden", 42, "t2v"))
    run(service.start("A beach", 7, "t2v"))
    before = service._status()
    messages.clear()
    error = HistoryFullError("full")
    error.world_id = 1
    assert run(service.process_output(StepOutcome(error=error, elapsed=0.1))) is None
    assert service._status() == before and not messages
    service.engine.reset.assert_not_called()


def test_model_storage_failure_reports_its_world():
    model = MovieRollModel()
    model._backend = Mock()
    model._backend.step.side_effect = HistoryFullError("full")
    with pytest.raises(HistoryFullError) as error:
        model.generate(
            MovieRollInput(
                world_id=17,
                prompt="garden",
                anchor=Anchor(seed=0, mode="t2v", image=None),
            )
        )
    assert error.value.world_id == 17 and model._chunk_index == 0
