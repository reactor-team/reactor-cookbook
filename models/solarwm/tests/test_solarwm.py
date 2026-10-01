"""Contract tests for the SolarWM Reactor adapter."""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from solarwm_stage2 import SolarWM
from solarwm_stage2_camera import CameraMotionPlanner, MotionConfig
from solarwm_stage2_images import normalize_output_frames, prepare_uploaded_image
from solarwm_stage2_model import NoAnchor, SolarWMInput, SolarWMModel
from solarwm_stage2_types import SolarWMState


@pytest.mark.parametrize("value", ["~/source", "relative/source", "/absolute/source"])
def test_config_paths_expand_home_and_ignore_cwd(tmp_path, monkeypatch, value):
    import yaml

    import solarwm_stage2_config as assets

    path = Path(__file__).parents[1] / "solarwm.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["source"]["path"] = value
    raw["assets"]["root"] = value
    monkeypatch.setattr(assets.yaml, "safe_load", lambda text: raw)
    monkeypatch.chdir(tmp_path)
    config = assets.read_config(path, tmp_path)
    expected = Path(value).expanduser()
    assert config.source_path == (path.parent / expected).resolve()
    assert config.base_path.parent == (tmp_path / expected).resolve()


@pytest.mark.parametrize(
    "damaged",
    [
        "SolarWM-5B-base/text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
        "SolarWM-5B-base/vae/Wan2.2_VAE.pth",
        "SolarWM-5B-base/tokenizer/tokenizer.json",
        "SolarWM-5B-base/tokenizer/tokenizer_config.json",
        "SolarWM-5B-base/tokenizer/spiece.model",
        "SolarWM-5B-base/tokenizer/special_tokens_map.json",
        "SolarWM-5B-sgf-stage2-81f/model.pt",
        "SolarWM-5B-sgf-stage2-81f/release-manifest.json",
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_asset_download_repairs_each_required_file(
    tmp_path, monkeypatch, damaged, empty
):
    import sys
    from dataclasses import replace

    import solarwm_stage2_config as assets

    config = assets.read_config(Path(__file__).parents[1] / "solarwm.yaml", tmp_path)
    config = replace(config, source_path=tmp_path / "source")
    (config.source_path / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=config.source_revision),
    )
    root = config.base_path.parent
    files = [
        "SolarWM-5B-base/text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
        "SolarWM-5B-base/vae/Wan2.2_VAE.pth",
        "SolarWM-5B-base/tokenizer/tokenizer.json",
        "SolarWM-5B-base/tokenizer/tokenizer_config.json",
        "SolarWM-5B-base/tokenizer/spiece.model",
        "SolarWM-5B-base/tokenizer/special_tokens_map.json",
        "SolarWM-5B-sgf-stage2-81f/model.pt",
        "SolarWM-5B-sgf-stage2-81f/release-manifest.json",
    ]

    def populate(**kwargs):
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    populate()
    target = root / damaged
    if empty:
        target.write_bytes(b"")
    else:
        target.unlink()

    def download_snapshot(**kwargs):
        # Simulate Hub reusing stale metadata for an existing zero-byte file.
        populate()
        if empty and not kwargs.get("force_download"):
            target.write_bytes(b"")

    download = Mock(side_effect=download_snapshot)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )

    assets.prepare_runtime(config)
    assert download.call_count == (2 if empty else 1)
    if empty:
        assert download.call_args.kwargs["force_download"] is True
        assert download.call_args.kwargs["allow_patterns"] == [damaged]
    assert download.call_args.kwargs["revision"] == config.checkpoint_revision

    download.reset_mock()
    assets.prepare_runtime(config)
    download.assert_not_called()
    target.unlink()
    download.side_effect = None
    with pytest.raises(RuntimeError, match="missing or empty"):
        assets.prepare_runtime(config)


def _upload() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (1280, 720), (40, 80, 120)).save(stream, format="PNG")
    data = stream.getvalue()
    return UploadedFile(name="anchor.png", mime_type="image/png", data=data)


def test_pipeline_declares_buffer_without_fixed_fps() -> None:
    assert SolarWM.buffer_size == 12
    assert "fps" not in SolarWM.__dict__


def test_camera_first_chunk_preserves_anchor_then_moves() -> None:
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    poses = planner.plan_chunk(strafe=0, vertical=0, forward=1, pitch=0, yaw=0, roll=0)
    assert poses.shape == (3, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-6)
    assert poses[1, 2, 3] > 0
    assert poses[2, 2, 3] > poses[1, 2, 3]
    later = planner.plan_chunk(strafe=0, vertical=0, forward=1, pitch=0, yaw=0, roll=0)
    assert later[0, 2, 3] > poses[2, 2, 3]


def test_camera_normalizes_combined_motion() -> None:
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    poses = planner.plan_chunk(strafe=1, vertical=1, forward=1, pitch=1, yaw=1, roll=1)
    assert np.linalg.norm(poses[1, :3, 3]) == pytest.approx(1.0)


@pytest.mark.parametrize("pitch", [-1.0, 1.0])
def test_pitch_direction_matches_opencv_camera_axes(pitch: float) -> None:
    planner = CameraMotionPlanner(MotionConfig(1.0, 8.0))
    controls = {
        "strafe": 0,
        "vertical": 0,
        "forward": 0,
        "pitch": pitch,
        "yaw": 0,
        "roll": 0,
    }
    first = planner.plan_chunk(**controls)
    later = planner.plan_chunk(**controls)
    np.testing.assert_allclose(first[0], np.eye(4), atol=1e-6)
    # The third rotation column is the viewing direction. Negative OpenCV Y
    # means looking upward, so positive user pitch must produce negative Y.
    for index, pose in enumerate(np.concatenate((first[1:], later)), start=1):
        angle = np.radians(pitch * 8.0 * index)
        np.testing.assert_allclose(
            pose[:3, 2], [0, -np.sin(angle), np.cos(angle)], atol=1e-6
        )
        np.testing.assert_allclose(pose[:3, 3], 0, atol=1e-6)


def test_upload_validation_and_output_normalization() -> None:
    prepare_uploaded_image(_upload())
    frames = normalize_output_frames(np.zeros((12, 480, 864, 3), dtype=np.float32))
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous


def test_set_image_accepts_empty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the configured default when the first image has no prompt."""
    import asyncio

    model = SolarWM()
    model._engine = SolarWMModel()
    model.state = SolarWMState()
    model._config = SimpleNamespace(
        max_chunks=320,
        default_prompt="A realistic cinematic scene with smooth camera motion.",
    )
    model.state._selected_image = None
    model.state._seed = 42
    model.state._chunk_index = 0
    model.state._last_chunk_seconds = None
    model.state.prompt = ""
    model.state._limit_reached = False
    for name in ("forward", "strafe", "vertical", "pitch", "yaw", "roll"):
        setattr(model.state, name, 0.0)
    sent = []

    async def record(message: object) -> None:
        sent.append(message)

    monkeypatch.setattr(model, "send", record)

    reply = asyncio.run(model.set_image(_upload(), ""))

    assert reply.prompt == "A realistic cinematic scene with smooth camera motion."
    assert (
        model.state.prompt == "A realistic cinematic scene with smooth camera motion."
    )
    assert model.state._world_id != model.state._applied_world_id


def test_refused_step_never_reaches_model() -> None:
    app = SolarWM()
    app._engine = SolarWMModel()
    app.state = SolarWMState()
    app._engine.generate = Mock(side_effect=AssertionError("must not run"))
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    app._engine.generate.assert_not_called()


def test_generate_reads_only_frozen_input() -> None:
    app = SolarWM()
    app._engine = SolarWMModel()
    value = SolarWMInput(1, None, np.eye(4)[None])
    with pytest.raises(FrozenInstanceError):
        value.world_id = 2
    result = object()
    app._engine = SimpleNamespace(
        generate=lambda actual: result if actual is value else None
    )
    app.state = None
    assert app.generate(value) is result


def test_ten_continuous_hook_steps_anchor_once_and_native_limit() -> None:
    app = SolarWM()
    app._engine = SolarWMModel()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=10, default_prompt="forest")
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))
    app._engine.max_chunks = 10
    resets = []

    class Backend:
        index = 0

        def reset(self, *args):
            resets.append(args)
            self.index = 0

        def generate_chunk(self, poses):
            assert poses.shape == (3, 4, 4)
            self.index += 1
            return np.zeros((12, 8, 8, 3), np.uint8), self.index

        def end_session(self):
            pass

    app._engine.backend = Backend()

    async def drive():
        await app.set_image(_upload(), "forest")
        for index in range(1, 11):
            value = await app.process_input()
            assert (value.anchor is not None) == (index == 1)
            result = app.generate(value)
            assert result.chunk_index == index
            await app.process_output(StepOutcome(result=result, elapsed=1.25))
        assert len(resets) == 1
        assert app.state._last_chunk_seconds == 1.25
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(43)
        assert (await app.process_input()).anchor is not None

    asyncio.run(drive())


def test_model_error_reaches_output_without_counting_step() -> None:
    app = SolarWM()
    app._engine = SolarWMModel()
    app.state = SolarWMState()
    with pytest.raises(NoAnchor) as caught:
        app.generate(SolarWMInput(1, None, np.eye(4)[None]))
    with pytest.raises(NoAnchor):
        asyncio.run(app.process_output(StepOutcome(error=caught.value)))
    assert app.state._chunk_index == 0


def test_upload_rejects_declared_type_mismatch() -> None:
    upload = _upload()
    wrong = UploadedFile(name="anchor.jpg", mime_type="image/jpeg", data=upload.data)
    with pytest.raises(CommandError):
        prepare_uploaded_image(wrong)


def test_backend_failure_after_success_does_not_count_or_acknowledge() -> None:
    app = SolarWM()
    app._engine = SolarWMModel()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=320, default_prompt="forest")
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))
    app._engine.backend = SimpleNamespace(
        reset=Mock(),
        generate_chunk=Mock(return_value=(np.zeros((9, 8, 8, 3), np.uint8), 1)),
    )

    async def drive():
        await app.set_image(_upload(), "forest")
        first = await app.process_input()
        assert first.anchor is not None
        assert (await app.process_input()).anchor is not None
        result = app.generate(first)
        await app.process_output(StepOutcome(result=result, elapsed=0.5))
        value = await app.process_input()
        assert value.anchor is None
        acknowledged = app.state._applied_world_id
        app._engine.backend.generate_chunk.side_effect = RuntimeError("backend failed")
        with pytest.raises(RuntimeError, match="backend failed") as caught:
            app.generate(value)
        with pytest.raises(RuntimeError, match="backend failed"):
            await app.process_output(StepOutcome(error=caught.value))
        assert app.state._chunk_index == 1
        assert app.state._applied_world_id == acknowledged
        assert app.state._last_chunk_seconds == 0.5

    asyncio.run(drive())


def test_model_graph_imports_without_runtime() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'reactor_runtime' or name.startswith('reactor_runtime.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import solarwm_stage2_model, solarwm_stage2_backend, solarwm_stage2_config
solarwm_stage2_model.SolarWMModel()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True
    )


def test_image_prepared_once_at_native_shape():
    value = prepare_uploaded_image(_upload())
    assert value.shape == (480, 864, 3)
    assert value.dtype == np.uint8
    app = SolarWM()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=320, default_prompt="forest")
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))
    asyncio.run(app.set_image(_upload(), "forest"))
    assert asyncio.run(app.process_input()).anchor.image is app.state._selected_image


def test_reset_only_queues_new_world():
    app = SolarWM()
    app.state = SolarWMState()
    app._engine = Mock()
    app._config = SimpleNamespace(max_chunks=320, default_prompt="forest")
    asyncio.run(app.set_image(_upload(), "forest"))
    previous = app.state._world_id
    asyncio.run(app.reset(42))
    app._engine.reset.assert_not_called()
    assert app.state._world_id == previous + 1


def test_config_paths_use_explicit_weights_root(tmp_path):
    from solarwm_stage2_config import read_config

    config = read_config(Path(__file__).parents[1] / "solarwm.yaml", tmp_path)
    assert config.base_path.is_relative_to(tmp_path)
    assert config.checkpoint_path.is_relative_to(tmp_path)
    assert config.runtime_root.is_relative_to(tmp_path)


def test_application_with_fake_model():
    from solarwm_stage2_model import SolarWMResult

    app = SolarWM()
    app.state = SolarWMState()
    app._config = SimpleNamespace(max_chunks=320, default_prompt="forest")
    app.state._planner = CameraMotionPlanner(MotionConfig(1, 8))
    app._engine = Mock()
    asyncio.run(app.set_image(_upload(), "forest"))
    value = asyncio.run(app.process_input())
    result = SolarWMResult(
        value.world_id, 1, np.zeros((9, 2, 2, 3), np.uint8), False, 320
    )
    app._engine.generate.return_value = result
    assert app.generate(value) is result
    asyncio.run(app.process_output(StepOutcome(result=result)))
    assert asyncio.run(app.process_input()).anchor is None
