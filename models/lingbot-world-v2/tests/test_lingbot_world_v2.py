"""Verify isolated native rollout bookkeeping and the application's step hooks."""

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from lingbot_world_v2 import LingBotWorldV2
from lingbot_world_v2_assets import BuiltInScene, read_config
from lingbot_world_v2_backend import _open_image
from lingbot_world_v2_camera import CameraMotionPlanner
from lingbot_world_v2_model import (
    LingbotV2Anchor,
    LingbotV2Input,
    LingbotV2Model,
    LingbotV2Result,
    NoAnchor,
    RolloutExhausted,
)
from lingbot_world_v2_types import LingBotWorldV2State
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile


def upload():
    data = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(data, format="PNG")
    return UploadedFile(name="anchor.png", mime_type="image/png", data=data.getvalue())


class FakeModel:
    def __init__(self):
        self.inputs = []
        self.world_id = None
        self.index = 0
        self.reset_count = 0
        self.error = None
        self.limit = 20

    def generate(self, input):
        self.inputs.append(input)
        if self.error:
            raise self.error
        if input.world_id != self.world_id:
            assert input.anchor is not None
            self.world_id = input.world_id
            self.index = 0
        self.index += 1
        return LingbotV2Result(
            np.zeros((13 if self.index == 1 else 16, 8, 8, 3), np.uint8),
            input.world_id,
            self.index,
            input.prompt,
            self.index >= self.limit,
        )

    def reset(self):
        self.reset_count += 1
        self.world_id = None
        self.index = 0


def make_model():
    app = LingBotWorldV2()
    app.state = LingBotWorldV2State()
    app.state.prompt = "A garden."
    app._config = SimpleNamespace(
        seed=42,
        max_chunks=20,
        chunk_latents=4,
        upload_intrinsics=(1, 1, 0.5, 0.5),
        upload_default_prompt="A peaceful landscape.",
        scenes=(),
    )
    app._planner = CameraMotionPlanner(16.0, 45.0)
    app._engine = FakeModel()
    app.send = AsyncMock()
    app.output.flush = Mock()
    app._selected_input = upload()
    return app


def step(app):
    input = asyncio.run(app.process_input())
    result = app.generate(input)
    output = asyncio.run(
        app.process_output(StepOutcome(result=result, elapsed=1.23456))
    )
    return input, result, output


@pytest.mark.parametrize("refusal", ["image", "limit", "prompt"])
def test_refusal_never_reaches_model(refusal):
    app = make_model()
    if refusal == "image":
        app._selected_input = None
    elif refusal == "limit":
        app._limit_reached = True
    else:
        app.state.prompt = " "
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    assert app._engine.inputs == []


def test_generate_reads_only_frozen_input():
    app = make_model()
    input = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "changed"
    app.state = app._config = app._planner = app._selected_input = None
    assert app.generate(input).world_id == input.world_id


def test_ten_actions_keep_one_world_and_send_anchor_once():
    app = make_model()
    axes = ["forward", "strafe", "vertical", "pitch", "yaw", "roll"]
    for index in range(10):
        controls = dict.fromkeys(axes, 0.0)
        controls[axes[index % 6]] = 0.1 if index < 6 else -0.1
        asyncio.run(app.set_camera(**controls))
        input, result, output = step(app)
        assert (input.anchor is not None) == (index == 0)
        assert input.poses.shape == (4, 4, 4)
        assert output.main_video.shape[0] == (13 if index == 0 else 16)
        assert result.chunk_index == index + 1
        completed = app.send.call_args_list[-2].args[0]
        assert completed.generation_seconds == 1.235
        assert completed.yaw == controls["yaw"]
    assert app._chunk_index == 10
    assert len({input.world_id for input in app._engine.inputs}) == 1


def test_failure_does_not_acknowledge_or_advance():
    app = make_model()
    app._engine.error = RuntimeError("failed")
    input = asyncio.run(app.process_input())
    with pytest.raises(RuntimeError, match="failed") as error:
        app.generate(input)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(app.process_output(StepOutcome(error=error.value)))
    assert app._chunk_index == 0
    assert app.state._applied_world_id is None
    assert asyncio.run(app.process_input()).anchor is not None
    app.send.assert_not_called()


def test_prompt_reset_upload_release_and_session_cleanup():
    app = make_model()
    first, _, _ = step(app)
    asyncio.run(app.set_prompt(prompt="  New garden.  "))
    second, result, _ = step(app)
    assert second.anchor is None and second.world_id == first.world_id
    assert result.prompt == "New garden."
    asyncio.run(app.reset(seed=123))
    third, _, _ = step(app)
    assert third.anchor.seed == 123
    assert third.world_id != first.world_id
    new_image = upload()
    asyncio.run(app.set_image(image=new_image, prompt="A forest."))
    fourth, _, _ = step(app)
    assert fourth.anchor.image == new_image.data
    assert fourth.world_id != third.world_id
    assert fourth.prompt == "A forest."
    asyncio.run(app.release_camera())
    assert all(value == 0 for value in app._camera_controls().values())
    app.on_session_ended()
    assert app._engine.reset_count == 1
    assert app._selected_input is None and app.state._applied_world_id is None
    app.on_session_started()
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())


def test_limit_reports_last_controls_and_requires_explicit_reset():
    app = make_model()
    app._engine.limit = 1
    app.state._yaw = 0.3
    step(app)
    assert app._limit_reached
    assert app.send.call_args_list[0].args[0].yaw == 0.3
    assert app.state._yaw == 0
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    with pytest.raises(CommandError):
        asyncio.run(app.set_prompt(prompt="New prompt"))
    asyncio.run(app.reset(seed=-1))
    assert asyncio.run(app.process_input()).anchor is not None


def test_model_owns_native_reset_count_limit_and_prompt_changes():
    engine = LingbotV2Model()
    backend = Mock()
    backend.generate_chunk.return_value = np.zeros((13, 8, 8, 3), np.uint8)
    engine._backend = backend
    engine._max_chunks = 2
    input = LingbotV2Input(1, None, "Garden", np.tile(np.eye(4), (4, 1, 1)))
    with pytest.raises(NoAnchor):
        engine.generate(input)
    anchor = LingbotV2Anchor(Path("anchor.png"), np.ones(4), 42)
    first = engine.generate(replace(input, anchor=anchor))
    second = engine.generate(replace(input, prompt="Forest"))
    assert (first.chunk_index, second.chunk_index) == (1, 2)
    assert second.complete and second.prompt == "Forest"
    backend.reset.assert_called_once_with(
        image=anchor.image,
        prompt="Garden",
        seed=42,
        intrinsics=anchor.intrinsics,
    )
    assert backend.generate_chunk.call_args.kwargs["relative_poses"] is input.poses
    with pytest.raises(RolloutExhausted):
        engine.generate(input)
    fresh = engine.generate(replace(input, world_id=2, anchor=anchor))
    assert fresh.chunk_index == 1 and not fresh.complete
    engine.reset()
    backend.end_session.assert_called_once()
    with pytest.raises(NoAnchor):
        engine.generate(input)


def test_model_backend_import_without_runtime():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('reactor_runtime'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import lingbot_world_v2_model
import lingbot_world_v2_backend
import lingbot_world_v2_assets
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_random_image_uses_selected_scene_calibration(tmp_path):
    app = make_model()
    image = tmp_path / "scene.png"
    image.write_bytes(upload().data)
    poses_path = tmp_path / "poses.npy"
    intrinsics_path = tmp_path / "intrinsics.npy"
    np.save(poses_path, np.eye(4, dtype=np.float32)[None])
    intrinsics = np.array([415, 415, 416, 240], np.float32)
    np.save(intrinsics_path, intrinsics)
    scene = BuiltInScene("scene", image, "Scene prompt", intrinsics_path, poses_path)
    app._config.scenes = (scene,)
    selected = asyncio.run(app.random_image())
    assert selected.filename == image.name
    input, _, _ = step(app)
    assert input.anchor.image == image and input.prompt == scene.prompt
    np.testing.assert_array_equal(input.anchor.intrinsics, intrinsics)
    assert asyncio.run(app.process_input()).anchor is None


def test_bad_upload_and_missing_examples_preserve_current_world():
    app = make_model()
    original = app._selected_input
    with pytest.raises(CommandError):
        asyncio.run(
            app.set_image(
                image=UploadedFile(name="bad.png", mime_type="image/png", data=b"bad"),
                prompt="Bad input",
            )
        )
    with pytest.raises(CommandError):
        asyncio.run(app.random_image())
    assert app._selected_input is original and app.state._world_id == 0


def test_backend_decodes_bytes_and_paths_equally(tmp_path):
    data = upload().data
    path = tmp_path / "anchor.png"
    path.write_bytes(data)
    np.testing.assert_array_equal(
        np.asarray(_open_image(data)), np.asarray(_open_image(path))
    )


def test_config_paths_are_explicit_and_native_settings_are_preserved(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LINGBOT_WORLD_V2_PATH", raising=False)
    monkeypatch.delenv("LINGBOT_WORLD_V2_CHECKPOINT_PATH", raising=False)
    path = Path(__file__).parents[1] / "lingbot_world_v2.yaml"
    config = read_config(path, tmp_path)
    assert config.source_path == tmp_path / "lingbot-world-v2"
    assert config.checkpoint_path == tmp_path / "lingbot-world-v2-14b-causal-fast"
    assert config.chunk_latents == 4
    assert config.timesteps == (0, 179, 358, 679)
    assert (config.local_attention_frames, config.attention_sink_frames) == (18, 6)
    assert config.max_chunks == 256
    standalone = read_config(path)
    assert standalone.source_path == path.resolve().parent / "lingbot-world-v2"
