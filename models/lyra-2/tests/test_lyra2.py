"""Drive the step boundary and native continuity without model weights."""

import asyncio
import io
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

import lyra2
from lyra2 import Lyra2
from lyra2_camera import Lyra2CameraPlanner
from lyra2_images import prepare_image
from lyra2_model import Lyra2Input, Lyra2Model, NoAnchor
from lyra2_types import ChunkCompleted, Lyra2State


def app():
    value = Lyra2()
    value.state = Lyra2State()
    value.send = AsyncMock()
    value.engine = Lyra2Model()
    value.state._planner = Lyra2CameraPlanner(
        translation_per_frame=0.0021875, rotation_degrees_per_frame=0.125
    )
    return value


def test_refused_step_never_reaches_model():
    value = app()
    value.engine = Mock()
    with pytest.raises(ApplicationError):
        asyncio.run(value.process_input())
    value.engine.generate.assert_not_called()


def test_generate_uses_only_frozen_input():
    value = app()
    value.engine = Mock()
    step = Lyra2Input(1, None, "scene", 1, None, None)
    value.state = None
    assert value.generate(step) is value.engine.generate.return_value
    value.engine.generate.assert_called_once_with(step)
    with pytest.raises(FrozenInstanceError):
        step.seed = 2


def test_every_step_yields_video_and_seeds_with_the_first_chunk():
    # Bench/Runtime stepping needs video from every step: the seed runs with chunk 1.
    value = app()
    value.state._image = Path("anchor.png")
    value.state._anchor = np.zeros((8, 8, 3), np.uint8)
    value.state.prompt = "scene"
    seed_k = np.diag([5.0, 6.0, 1.0]).astype(np.float32)
    backend = Mock()
    backend.reset.return_value = (np.eye(4, dtype=np.float32), seed_k)
    backend.generate_chunk.return_value = (np.zeros((80, 8, 8, 3), np.uint8), None)
    value.engine.backend = backend

    async def run():
        first = await value.process_input()
        assert first.anchor is value.state._anchor and first.intrinsics is None
        for index in range(1, 11):
            step = first if index == 1 else await value.process_input()
            assert (step.anchor is None) == (index > 1)
            result = value.generate(step)
            output = await value.process_output(
                StepOutcome(result=result, error=None, elapsed=1.25)
            )
            assert output is not None and len(output.main_video) == 80
            assert value.state._chunk == index
        backend.reset.assert_called_once()
        calls = backend.generate_chunk.call_args_list
        assert [call.kwargs["chunk"] for call in calls] == list(range(1, 11))
        assert all(np.array_equal(call.args[1][0], seed_k) for call in calls)
        np.testing.assert_array_equal(value.state._intrinsics, seed_k)
        messages = [
            call.args[0]
            for call in value.send.call_args_list
            if isinstance(call.args[0], ChunkCompleted)
        ]
        assert len(messages) == 10
        assert all(message.generation_seconds == 1.25 for message in messages)

    asyncio.run(run())


def test_failed_seed_does_not_commit_world():
    engine = Lyra2Model()
    engine.backend = Mock()
    engine.backend.reset.return_value = (np.eye(4, dtype=np.float32), np.eye(3))
    engine.backend.generate_chunk.side_effect = ValueError("failed")
    step = Lyra2Input(1, np.zeros((8, 8, 3), np.uint8), "scene", 1, np.zeros((80, 4, 4)), None)
    with pytest.raises(ValueError):
        engine.generate(step)
    assert engine.world_id is None and engine.chunk == 0


def test_failed_model_step_reaches_output_without_counting():
    value = app()
    value.engine.backend = Mock()
    step = Lyra2Input(1, None, "scene", 1, np.zeros((80, 4, 4)), None)
    with pytest.raises(NoAnchor) as caught:
        value.generate(step)
    with pytest.raises(NoAnchor):
        asyncio.run(
            value.process_output(
                StepOutcome(result=None, error=caught.value, elapsed=0.1)
            )
        )
    assert value.state._chunk == value.engine.chunk == 0
    value.send.assert_not_called()


def test_backend_failure_does_not_commit_chunk():
    engine = Lyra2Model()
    engine.backend = Mock()
    engine.world_id = 1
    engine.backend.generate_chunk.side_effect = ValueError("failed")
    with pytest.raises(ValueError):
        engine.generate(
            Lyra2Input(1, None, "scene", 1, np.zeros((80, 4, 4)), np.zeros((80, 3, 3)))
        )
    assert engine.chunk == 0


def planner():
    return Lyra2CameraPlanner(
        translation_per_frame=0.002, rotation_degrees_per_frame=0.125
    )


def test_native_chunk_shape_and_intrinsics():
    p = planner()
    k = np.array(((500, 0, 384), (0, 500, 224), (0, 0, 1)), np.float32)
    result = p.plan_chunk(
        forward=1,
        strafe=0,
        vertical=0,
        pitch=0,
        yaw=0,
        roll=0,
        frame_count=80,
        intrinsics=k,
    )
    assert result.w2c.shape == (80, 4, 4)
    assert result.intrinsics.shape == (80, 3, 3)
    np.testing.assert_array_equal(result.intrinsics[0], k)
    assert np.linalg.inv(result.w2c[-1])[2, 3] > 0


def test_six_axes_are_continuous_between_chunks():
    p = planner()
    k = np.eye(3, dtype=np.float32)
    first = p.plan_chunk(
        forward=0.5,
        strafe=1,
        vertical=0.25,
        pitch=0.2,
        yaw=-0.4,
        roll=0.1,
        frame_count=80,
        intrinsics=k,
    )
    second = p.plan_chunk(
        forward=0.5,
        strafe=1,
        vertical=0.25,
        pitch=0.2,
        yaw=-0.4,
        roll=0.1,
        frame_count=80,
        intrinsics=k,
    )
    boundary_step = np.linalg.norm(
        np.linalg.inv(second.w2c[0])[:3, 3] - np.linalg.inv(first.w2c[-1])[:3, 3]
    )
    assert 0 < boundary_step < 0.01
    assert np.linalg.norm(np.linalg.inv(second.w2c[-1])[:3, 3]) > np.linalg.norm(
        np.linalg.inv(first.w2c[-1])[:3, 3]
    )


def test_rejects_non_native_chunk_size():
    with pytest.raises(ValueError, match="exactly 80"):
        planner().plan_chunk(
            forward=0,
            strafe=0,
            vertical=0,
            pitch=0,
            yaw=0,
            roll=0,
            frame_count=79,
            intrinsics=np.eye(3),
        )


# Input preparation and session/worker ownership regression tests.


def upload(data=b"invalid"):
    return UploadedFile(name="scene.png", mime_type="image/png", data=data)


def test_native_image_preparation_and_bad_upload():
    data = io.BytesIO()
    Image.new("RGB", (120, 80), (10, 20, 30)).save(data, format="PNG")
    fitted = prepare_image(upload(data.getvalue()))
    assert fitted.shape == (480, 832, 3) and fitted.dtype == np.uint8
    with pytest.raises(CommandError):
        prepare_image(upload())
    with pytest.raises(CommandError):
        prepare_image(upload(b"x" * (25 * 1024 * 1024 + 1)))


def test_pixel_limit_and_decompression_bomb(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    data = io.BytesIO()
    Image.new("RGB", (8, 8)).save(data, format="PNG")
    with pytest.raises(CommandError):
        prepare_image(upload(data.getvalue()))


def test_reset_immediately_clears_progress_without_calling_model():
    value = app()
    value.output.flush = Mock()
    value.engine = Mock()
    value.state._anchor = np.zeros((8, 8, 3), np.uint8)
    value.state._chunk = 5
    value.state._active_prompt = "old"

    async def run():
        await value.reset(seed=2)
        assert value.state._chunk == 0
        queued = await value.set_prompt("new")
        assert queued.applies_to_chunk == 1
        camera = await value.set_camera_motion(
            forward=0, strafe=0, vertical=0, pitch=0, yaw=0, roll=0
        )
        assert camera.applies_to_chunk == 1

    asyncio.run(run())
    value.engine.reset.assert_not_called()


def test_empty_world_commands_and_missing_samples():
    value = app()
    value.config = {"source_path": "/nonexistent/lyra-test"}

    async def run():
        for command in (
            value.reset(seed=-1),
            value.set_prompt("scene"),
            value.release_camera(),
            value.random_image(),
        ):
            with pytest.raises(CommandError):
                await command

    asyncio.run(run())


def test_new_session_has_fresh_private_state_and_configured_seed():
    value = app()
    value.config = {
        "seed": 7,
        "translation_per_frame": 0.002,
        "rotation_degrees_per_frame": 0.125,
    }
    value.started()
    value.state._seed = 900
    value.state._chunk = 5
    value.state = Lyra2State()
    value.started()
    assert value.state._seed == 7 and value.state._chunk == 0
    assert value.state._anchor is None


def test_disconnect_preserves_world_and_broadcasts_released_axes():
    value = app()
    value.state.forward = 1
    value.state._chunk = 5
    value.engine = Mock()
    asyncio.run(value.on_disconnected())
    assert value.state.forward == 0 and value.state._chunk == 5
    value.engine.reset.assert_not_called()
    assert value.send.call_args.args[0].forward == 0


def test_load_uses_explicit_config_and_runtime_owned_worker(monkeypatch):
    config = {"source_path": "/source", "checkpoints_path": "/weights"}
    monkeypatch.setattr(lyra2, "prepare_config", Mock(return_value=config))
    monkeypatch.setattr(lyra2, "get_weights_path", lambda: Path("/weights"))
    runner = Mock()
    monkeypatch.setattr(lyra2, "DistributedRunner", runner)
    value = Lyra2()
    before = os.getcwd()
    value.load(Path("/config.yaml"))
    assert runner.call_args.kwargs["load_kwargs"] == {"config": config}
    assert runner.call_args.kwargs["world_size"] == 1
    runner.return_value.start.assert_called_once()
    assert os.getcwd() == before


def test_prompt_embedding_is_reused_until_text_changes():
    from lyra2_backend import Lyra2Backend

    backend = object.__new__(Lyra2Backend)
    backend.model = Mock(manual_t5="embedding")
    backend._prompt = "old"
    backend._set_prompt("old")
    backend.model._embed_prompt.assert_not_called()
    backend._set_prompt("new")
    backend.model._embed_prompt.assert_called_once_with("new")


def test_model_import_graph_is_runtime_free():
    import ast

    root = Path(__file__).parents[1]
    for name in ("lyra2_model.py", "lyra2_backend.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("reactor_runtime")
            elif isinstance(node, ast.Import):
                assert all(
                    not item.name.startswith("reactor_runtime") for item in node.names
                )


def test_image_preparation_matches_official_custom_trajectory_script():
    import io

    import cv2
    from lyra2_images import prepare_image
    from PIL import Image
    from reactor_runtime import UploadedFile

    pixels = np.random.default_rng(0).integers(0, 256, (300, 500, 3), np.uint8)
    for kind, extra in (("PNG", {}), ("JPEG", {"exif": Image.Exif()})):
        if kind == "JPEG":
            extra["exif"][0x0112] = 6  # rotated: OpenCV applies EXIF orientation
        data = io.BytesIO()
        Image.fromarray(pixels).save(data, kind, **extra)
        raw = data.getvalue()
        # lyra2_custom_traj_inference: cv2.imread -> RGB -> INTER_LINEAR resize.
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        expected = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (832, 480), interpolation=cv2.INTER_LINEAR
        )
        upload = UploadedFile(name="seed", mime_type=f"image/{kind.lower()}", data=raw)
        np.testing.assert_array_equal(prepare_image(upload), expected)
    assert bgr.shape[:2] == (500, 300)


def test_backend_serves_official_script_defaults():
    # lyra2_custom_traj_inference.py:138-157 (nv-tlabs/lyra@9fffc9a).
    import lyra2_backend
    from lyra2_camera import POSE_SCALE

    assert (lyra2_backend.HEIGHT, lyra2_backend.WIDTH) == (480, 832)
    assert lyra2_backend.DA3_FRAME_INTERVAL == 8
    assert lyra2_backend.DA3_MAX_HISTORY_FRAMES == 10
    assert POSE_SCALE == 1.1
    source = (Path(__file__).parents[1] / "lyra2_backend.py").read_text()
    for flag in (
        "da3_include_ar_chunk_last_frames = False",
        "da3_use_predicted_pose = False",
        "da3_predicted_pose_continuation = False",
        "load_moge_model(",
    ):
        assert flag in source


def test_seed_depth_rescales_to_moge_and_uses_path_intrinsics(monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    import types

    from lyra2_backend import Lyra2Backend

    da3_depth = torch.full((4, 6), 2.0)
    da3_mask = torch.ones((4, 6))
    da3_k = torch.eye(3)
    moge_depth = torch.full((4, 6), 4.0)
    moge_mask = torch.ones((4, 6))
    module = types.ModuleType("lyra_2._src.inference.depth_utils")
    module.moge_infer_depth_intrinsics = lambda *args, **kwargs: (
        None,
        moge_depth,
        None,
        moge_mask,
    )
    for name in ("lyra_2", "lyra_2._src", "lyra_2._src.inference"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    backend = object.__new__(Lyra2Backend)
    backend.moge_model = object()
    backend._da3_single = lambda *args: ("image", da3_depth, da3_k, da3_mask)
    backend._seed_intrinsics = None
    _, depth, k, mask = backend._seed_depth(None, torch.zeros((4, 6, 3)), (4, 6))
    # Least squares on inverse depth: scale = 0.5, so DA3 depth becomes 4.
    torch.testing.assert_close(depth, moge_depth, rtol=1e-5, atol=1e-5)
    assert k is da3_k and mask is da3_mask
    path_k = np.array(((420.2, 0, 416), (0, 431.0, 240), (0, 0, 1)), np.float32)
    backend._seed_intrinsics = path_k
    np.testing.assert_array_equal(
        backend._seed_depth(None, torch.zeros((4, 6, 3)), (4, 6))[2].numpy(), path_k
    )
