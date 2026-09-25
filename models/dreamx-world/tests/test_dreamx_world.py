"""Test DreamX-World's lightweight adapter contracts without loading weights."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch
from reactor_runtime import UploadedFile

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

dreamx_assets = importlib.import_module("dreamx_assets")
dreamx_camera = importlib.import_module("dreamx_camera")
dreamx_images = importlib.import_module("dreamx_images")
dreamx_types = importlib.import_module("dreamx_types")
dreamx_world = importlib.import_module("dreamx_world")


def test_camera_chunks_follow_native_latent_alignment() -> None:
    """Select pixel poses at DreamX's 1+4k latent alignment across chunks."""
    controller = dreamx_camera.DreamXCameraController(speed=1.5)

    first = controller.plan_chunk(frozenset({"w"}), first_chunk=True)
    second = controller.plan_chunk(frozenset({"w"}), first_chunk=False)

    np.testing.assert_array_equal(first.poses[:, 0], [0, 1, 5])
    np.testing.assert_array_equal(second.poses[:, 0], [9, 13, 17])
    assert first.reference_pose is None
    np.testing.assert_array_equal(second.reference_pose, first.poses[-1])
    np.testing.assert_allclose(first.poses[:, -1], [-0.075, -0.15, -0.45], atol=1e-6)


def test_camera_composes_native_movement_and_view_keys() -> None:
    """Apply translation and pan together while retaining an orthonormal pose."""
    controller = dreamx_camera.DreamXCameraController(speed=1.5)

    chunk = controller.plan_chunk(frozenset({"w", "j"}), first_chunk=True)

    rotation = chunk.poses[-1, 7:].reshape(3, 4)[:, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert not np.isclose(chunk.poses[-1, -1], -0.45)


def test_key_state_is_retained_for_next_chunk(monkeypatch: MonkeyPatch) -> None:
    """Retain pressed and released keys for the next chunk."""
    world = dreamx_world.DreamXWorld()
    world.state = dreamx_types.DreamXWorldState()
    world._selected_input = Path("selected.jpg")
    state_updates: list[Any] = []

    async def capture(message: Any) -> None:
        state_updates.append(message)

    monkeypatch.setattr(world, "send", capture)

    pressed = asyncio.run(world.set_key_state("w", True))
    assert pressed.pressed_keys == ["w"]
    assert world.state._pressed_keys == frozenset({"w"})

    released = asyncio.run(world.set_key_state("w", False))
    assert released.pressed_keys == []
    assert world.state._pressed_keys == frozenset()
    assert [message.pressed_keys for message in state_updates] == [["w"], []]


def test_image_selection_queues_fresh_rollout(monkeypatch: MonkeyPatch) -> None:
    """Start continuous generation after image selection."""
    world = dreamx_world.DreamXWorld()
    world.state = dreamx_types.DreamXWorldState()
    flushes: list[None] = []
    monkeypatch.setattr(world.output, "flush", lambda: flushes.append(None))

    world._select_image(Path("selected.jpg"), "uploaded", "A coherent world")

    assert flushes == [None]
    assert world.state._world_id != world.state._applied_world_id
    assert world._chunk_index == 0


def test_rollout_reset_flushes_pending_media(monkeypatch: MonkeyPatch) -> None:
    """Discard frames queued by the world that a reset replaces."""
    world = dreamx_world.DreamXWorld()
    world.state = dreamx_types.DreamXWorldState()
    flushes: list[None] = []
    monkeypatch.setattr(world.output, "flush", lambda: flushes.append(None))
    world._select_image(Path("selected.jpg"), "uploaded", "A coherent world")
    flushes.clear()

    asyncio.run(world.reset(-1))

    assert flushes == [None]


def test_inference_emits_one_complete_frame_batch(monkeypatch: MonkeyPatch) -> None:
    """Preserve chunk timing by emitting every decoded frame in one turn."""

    class Backend:
        def __init__(self) -> None:
            self.resets = 0
            self.prompts: list[str] = []

        def reset(self, _seed: int, _image: Path) -> None:
            self.resets += 1

        def generate_chunk(
            self, _prompt: str, _pressed_keys: frozenset[str]
        ) -> np.ndarray:
            self.prompts.append(_prompt)
            return np.zeros((9, 8, 8, 3), dtype=np.uint8)

    world = dreamx_world.DreamXWorld()
    world.state = dreamx_types.DreamXWorldState()
    world._config = SimpleNamespace(max_chunks_per_rollout=512)
    world._engine._backend = Backend()
    world._engine._config = world._config
    world._camera = dreamx_camera.DreamXCameraController(1.5)

    async def discard(_message: Any) -> None:
        return None

    monkeypatch.setattr(world, "send", discard)
    world._select_image(Path("selected.jpg"), "uploaded", "A coherent world")

    async def generate_first_chunk() -> Any:
        for index in range(10):
            world.state.prompt = f"Scene {index}"
            snapshot = await world.process_input()
            world.state.prompt = "A later input"
            result = world.generate(snapshot)
            output = await world.process_output(dreamx_world.StepOutcome(result=result))
            assert world._engine._backend.prompts[-1] == f"Scene {index}"
        assert world._chunk_index == 10
        assert world._engine._backend.resets == 1
        return output

    output = asyncio.run(generate_first_chunk())

    assert isinstance(output, dreamx_types.DreamXWorldOutput)
    assert output.main_video.shape == (9, 8, 8, 3)


def test_config_keeps_source_and_weights_under_runtime_root(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve source and checkpoints under the runtime-managed weights root."""
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(tmp_path))
    monkeypatch.delenv("DREAMX_WORLD_PATH", raising=False)

    config = dreamx_assets.read_config(MODEL_DIR / "dreamx_world.yaml")

    assert config.source_path == tmp_path / "DreamX-World"
    assert (
        config.dreamx.path == tmp_path / "checkpoints/DreamX-World-5B/model.safetensors"
    )
    assert config.wan.path == tmp_path / "checkpoints/Wan2.2-TI2V-5B"
    assert config.max_chunks_per_rollout == 512


def test_example_image_passes_upload_validation() -> None:
    """Accept a bundled JPEG through the same byte-upload contract as a client."""
    image_path = MODEL_DIR / "example_images/01_minecraft_sunset.jpg"
    upload = UploadedFile(
        name=image_path.name,
        mime_type="image/jpeg",
        data=image_path.read_bytes(),
    )

    dreamx_images.validate_uploaded_image(upload)


def make_app(monkeypatch):
    from dreamx_world_model import DreamXResult

    class Engine:
        def __init__(self):
            self.inputs = []
            self.resets = 0
            self.world_id = None
            self.index = 0

        def generate(self, input):
            self.inputs.append(input)
            if input.world_id != self.world_id:
                assert input.anchor is not None
                self.world_id = input.world_id
                self.index = 0
            self.index += 1
            return DreamXResult(
                np.zeros((9 if self.index == 1 else 12, 8, 8, 3), np.uint8),
                input.world_id,
                self.index,
                input.prompt,
                self.index >= 512,
            )

        def reset(self):
            self.resets += 1
            self.world_id = None

    app = dreamx_world.DreamXWorld()
    app.state = dreamx_types.DreamXWorldState()
    app._config = SimpleNamespace(
        seed=42, max_chunks_per_rollout=512, default_upload_prompt="Default scene"
    )
    app._camera = dreamx_camera.DreamXCameraController(1.5)
    app._engine = Engine()
    messages = []

    async def send(message):
        messages.append(message)

    monkeypatch.setattr(app, "send", send)
    return app, messages


def test_refused_steps_never_reach_model(monkeypatch):
    app, _ = make_app(monkeypatch)
    with pytest.raises(dreamx_world.ApplicationError, match="Select an image"):
        asyncio.run(app.process_input())
    app._selected_input = Path("image.jpg")
    app.state.prompt = " "
    with pytest.raises(dreamx_world.ApplicationError, match="non-empty prompt"):
        asyncio.run(app.process_input())
    assert app._engine.inputs == []


def test_generate_reads_only_frozen_input(monkeypatch):
    from dataclasses import FrozenInstanceError

    app, _ = make_app(monkeypatch)
    app._select_image(Path("image.jpg"), "built_in", "Scene")
    input = asyncio.run(app.process_input())
    with pytest.raises(FrozenInstanceError):
        input.prompt = "changed"
    app.state = None
    app._config = app._camera = app._selected_input = None
    assert app.generate(input).prompt == "Scene"


def test_ten_actions_acknowledge_anchor_once_and_keep_native_poses(monkeypatch):
    app, messages = make_app(monkeypatch)
    app._select_image(Path("image.jpg"), "built_in", "Scene")
    expected = dreamx_camera.DreamXCameraController(1.5)

    async def run():
        for index, key in enumerate(["w", "a", "s", "d", "i", "j", "k", "l", "w", "s"]):
            app._clear_controls()
            await app.set_key_state(key, True)
            await app.set_prompt(f"Scene {index}")
            input = await app.process_input()
            camera = expected.plan_chunk(frozenset({key}), first_chunk=index == 0)
            np.testing.assert_array_equal(input.poses, camera.poses)
            np.testing.assert_array_equal(input.reference_pose, camera.reference_pose)
            assert (input.anchor is not None) == (index == 0)
            result = app.generate(input)
            output = await app.process_output(
                dreamx_world.StepOutcome(result=result, elapsed=1.2345)
            )
            assert output.main_video.shape[0] == (9 if index == 0 else 12)
            assert app._chunk_index == index + 1
            assert messages[-2].inference_seconds == 1.234
            assert messages[-2].pressed_keys == [key]
            assert messages[-2].prompt == f"Scene {index}"

    asyncio.run(run())


def test_model_error_is_not_acknowledged_or_counted(monkeypatch):
    app, messages = make_app(monkeypatch)
    app._select_image(Path("image.jpg"), "built_in", "Scene")
    asyncio.run(app.process_input())
    error = RuntimeError("GPU failure")

    def fail(input):
        raise error

    monkeypatch.setattr(app._engine, "generate", fail)
    with pytest.raises(RuntimeError, match="GPU failure"):
        app.generate(asyncio.run(app.process_input()))
    with pytest.raises(RuntimeError, match="GPU failure"):
        asyncio.run(app.process_output(dreamx_world.StepOutcome(error=error)))
    assert app._chunk_index == 0
    assert app.state._applied_world_id is None
    assert not app._generating
    assert messages == []
    assert asyncio.run(app.process_input()).anchor is not None


def test_upload_reset_auto_reset_and_session_cleanup(monkeypatch):
    from dataclasses import replace

    app, messages = make_app(monkeypatch)
    image = MODEL_DIR / "example_images/01_minecraft_sunset.jpg"
    upload = UploadedFile(
        name=image.name, mime_type="image/jpeg", data=image.read_bytes()
    )

    async def run():
        await app.set_image(upload, "Uploaded scene")
        input = await app.process_input()
        assert input.anchor.image == upload.data
        result = app.generate(input)
        await app.process_output(dreamx_world.StepOutcome(result=result))
        assert (await app.process_input()).anchor is None
        await app.reset(99)
        input = await app.process_input()
        assert input.anchor.seed == 99
        assert input.world_id != result.world_id
        await app.set_key_state("w", True)
        result = app.generate(input)
        await app.process_output(
            dreamx_world.StepOutcome(result=replace(result, complete=True))
        )
        assert isinstance(messages[-2], dreamx_types.RolloutResetQueued)
        assert not app.state._pressed_keys
        assert (await app.process_input()).anchor is not None
        app.on_session_ended()
        assert app._engine.resets == 1
        assert app._selected_input is None
        assert app._chunk_index == 0
        assert app.state._applied_world_id is None

    asyncio.run(run())


def test_model_bookkeeping_guards_and_native_camera_forwarding():
    from dreamx_world_model import (
        DreamXAnchor,
        DreamXInput,
        DreamXModel,
        NoAnchor,
        RolloutExhausted,
    )

    class Backend:
        def __init__(self):
            self.resets = []
            self.calls = []
            self.ends = 0

        def reset(self, seed, image):
            self.resets.append((seed, image))

        def generate_chunk(self, prompt, camera):
            self.calls.append((prompt, camera))
            return np.zeros((9, 8, 8, 3), np.uint8)

        def end_session(self):
            self.ends += 1

    model = DreamXModel()
    backend = Backend()
    model._backend = backend
    model._config = SimpleNamespace(max_chunks_per_rollout=2)
    poses = np.zeros((3, 19), np.float32)
    input = DreamXInput(1, None, "Scene", poses, None)
    with pytest.raises(NoAnchor):
        model.generate(input)
    from dataclasses import replace

    first = model.generate(replace(input, anchor=DreamXAnchor(b"image", 42)))
    second = model.generate(replace(input, prompt="Changed prompt"))
    assert first.chunk_index == 1 and not first.complete
    assert second.chunk_index == 2 and second.complete
    assert backend.resets == [(42, b"image")]
    assert backend.calls[-1][0] == "Changed prompt"
    assert backend.calls[-1][1].poses is poses
    with pytest.raises(RolloutExhausted):
        model.generate(input)
    assert (
        model.generate(
            replace(input, world_id=2, anchor=DreamXAnchor(Path("image.png"), 77))
        ).chunk_index
        == 1
    )
    model.reset()
    assert backend.ends == 1
    with pytest.raises(NoAnchor):
        model.generate(input)


def test_model_imports_without_runtime():
    import subprocess

    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("reactor_runtime"):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import dreamx_world_model
assert dreamx_world_model.DreamXModel() is not None
"""
    subprocess.run([sys.executable, "-c", code], cwd=MODEL_DIR, check=True)


def test_model_dependency_graph_has_no_runtime_imports():
    import ast

    for filename in ["dreamx_world_model.py", "dreamx_backend.py", "dreamx_camera.py"]:
        tree = ast.parse((MODEL_DIR / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("reactor_runtime")
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("reactor_runtime") for alias in node.names
                )


def test_random_image_and_command_refusals(monkeypatch):
    app, _ = make_app(monkeypatch)
    from reactor_runtime import CommandError

    with pytest.raises(CommandError):
        asyncio.run(app.set_prompt("Scene"))
    with pytest.raises(CommandError):
        asyncio.run(app.reset(-1))
    with pytest.raises(CommandError):
        asyncio.run(app.set_key_state("w", True))
    image = (MODEL_DIR / "example_images/01_minecraft_sunset.jpg").resolve()
    app._config.random_images = (image,)
    app._scene_prompts = {image: "Demo scene"}
    result = asyncio.run(app.random_image())
    assert result.source == "built_in"
    assert result.prompt == "Demo scene"
    assert asyncio.run(app.process_input()).anchor.image == image
    with pytest.raises(CommandError):
        asyncio.run(app.set_prompt(" "))
