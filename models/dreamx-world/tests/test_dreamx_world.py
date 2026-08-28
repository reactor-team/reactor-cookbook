"""Test DreamX-World's lightweight adapter contracts without loading weights."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
    assert world.state._reset_requested is True
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
        def reset(self, _seed: int, _image: Path) -> None:
            return None

        def generate_chunk(
            self, _prompt: str, _pressed_keys: frozenset[str]
        ) -> np.ndarray:
            return np.zeros((9, 8, 8, 3), dtype=np.uint8)

    world = dreamx_world.DreamXWorld()
    world.state = dreamx_types.DreamXWorldState()
    world._backend = Backend()
    world._config = SimpleNamespace(max_chunks_per_rollout=512)

    async def discard(_message: Any) -> None:
        return None

    monkeypatch.setattr(world, "send", discard)
    world._select_image(Path("selected.jpg"), "uploaded", "A coherent world")

    async def generate_first_chunk() -> Any:
        return await anext(world.inference())

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
