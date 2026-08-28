"""Focused schema, control, and lifecycle tests for the HY-World 1.5 adapter."""

from __future__ import annotations

import asyncio
import importlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
from pytest import MonkeyPatch
from reactor_runtime import UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

RECIPE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(RECIPE_DIR))

pipeline_module = importlib.import_module("hy_world_1_5")
camera_module = importlib.import_module("hy_world_1_5_camera")
types_module = importlib.import_module("hy_world_1_5_types")
assets_module = importlib.import_module("hy_world_1_5_assets")

HYWorld15 = pipeline_module.HYWorld15
CameraControl = camera_module.CameraControl
NativeCameraPlanner = camera_module.NativeCameraPlanner
HYWorld15Output = types_module.HYWorld15Output
HYWorld15State = types_module.HYWorld15State
assemble_base_model = assets_module.assemble_base_model


class _Backend:
    def __init__(self) -> None:
        self.resets: list[tuple[str, int]] = []
        self.calls: list[tuple[Any, str]] = []

    def reset(self, *, image: Image.Image, prompt: str, seed: int) -> None:
        self.resets.append((prompt, seed))

    def generate_chunk(self, camera: Any, prompt: str) -> np.ndarray:
        self.calls.append((camera, prompt))
        count = 13 if len(self.calls) == 1 else 16
        return np.zeros((count, 8, 8, 3), dtype=np.uint8)

    def end_session(self) -> None:
        return None


def _image_upload() -> UploadedFile:
    stream = io.BytesIO()
    Image.new("RGB", (32, 18), color=(10, 20, 30)).save(stream, format="PNG")
    return UploadedFile(name="world.png", mime_type="image/png", data=stream.getvalue())


def _ready_model() -> tuple[Any, _Backend]:
    model = HYWorld15()
    model.state = HYWorld15State()
    model._config = SimpleNamespace(max_chunks=512, seed=1)
    backend = _Backend()
    model._backend = backend
    model._planner = NativeCameraPlanner()
    model._examples = ()
    model.on_session_started()

    async def discard(_message: Any) -> None:
        return None

    model.send = discard
    return model, backend


def test_contract_is_atomic_and_documents_command_results() -> None:
    """Expose one camera setter and complete success and failure semantics."""
    contract = ModelContract.of(HYWorld15)

    assert set(contract.commands) == {
        "set_prompt",
        "set_camera",
        "release_camera",
        "reset",
        "set_image",
        "random_image",
    }
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )
    assert "fps" not in HYWorld15.__dict__
    assert HYWorld15.buffer_size == 16


def test_native_camera_keeps_anchor_and_dual_action_alignment() -> None:
    """Match upstream latent cadence, motion scale, intrinsics, and action labels."""
    planner = NativeCameraPlanner()
    first = planner.plan(CameraControl(forward=1.0, strafe=0.0, pitch=0.0, yaw=0.0))

    np.testing.assert_allclose(first.viewmats[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(
        first.viewmats[:, 2, 3], [0.0, -0.08, -0.16, -0.24], atol=1e-6
    )
    np.testing.assert_array_equal(first.actions, [0, 9, 9, 9])
    np.testing.assert_allclose(first.intrinsics[0, 0, 0], 0.5050505)
    np.testing.assert_allclose(first.intrinsics[0, 1, 1], 0.89786756)

    diagonal = planner.plan(CameraControl(forward=1.0, strafe=1.0, pitch=1.0, yaw=1.0))
    np.testing.assert_array_equal(diagonal.actions, [50, 50, 50, 50])

    planner.reset()
    deadzone = planner.plan(
        CameraControl(forward=0.001, strafe=0.0, pitch=0.01, yaw=0.0)
    )
    np.testing.assert_array_equal(deadzone.actions, [0, 0, 0, 0])


def test_model_layout_links_remain_valid_after_weights_root_moves() -> None:
    """Keep assembled encoder paths portable across container bind mounts."""
    with tempfile.TemporaryDirectory() as directory:
        weights = Path(directory) / "weights"
        base_model = weights / "models/base"
        qwen = weights / "models/qwen"
        byt5 = weights / "models/byt5"
        glyph = weights / "models/glyph"
        vision = weights / "models/vision"
        for path in (qwen, byt5, glyph, vision):
            path.mkdir(parents=True)
        config = SimpleNamespace(
            base_model=SimpleNamespace(path=base_model),
            qwen=SimpleNamespace(path=qwen),
            byt5=SimpleNamespace(path=byt5),
            glyph=SimpleNamespace(path=glyph),
            flux_vision=SimpleNamespace(path=vision),
        )

        assemble_base_model(config)

        destinations = (
            base_model / "text_encoder/llm",
            base_model / "text_encoder/byt5-small",
            base_model / "text_encoder/Glyph-SDXL-v2",
            base_model / "vision_encoder/siglip",
        )
        assert all(path.is_symlink() for path in destinations)
        assert all(not Path(path.readlink()).is_absolute() for path in destinations)


def test_image_selection_queues_and_generates_first_chunk(
    monkeypatch: MonkeyPatch,
) -> None:
    """Generate continuously from the fresh world queued by image selection."""
    model, backend = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))

    reply = asyncio.run(model.set_image(_image_upload(), "A quiet road"))

    assert reply.source == "uploaded"
    assert reply.prompt == "A quiet road"
    assert model.state._restart_requested is True

    async def generate_first_chunk() -> Any:
        return await anext(model.inference())

    output = asyncio.run(generate_first_chunk())
    assert isinstance(output, HYWorld15Output)
    assert output.main_video.shape == (13, 8, 8, 3)
    assert flushes == [None]
    assert backend.resets == [("A quiet road", 1)]
    assert len(backend.calls) == 1
    assert model._chunk_index == 1


def test_rollout_reset_flushes_pending_media(monkeypatch: MonkeyPatch) -> None:
    """Discard frames queued by the world that a reset replaces."""
    model, _backend = _ready_model()
    flushes: list[None] = []
    monkeypatch.setattr(model.output, "flush", lambda: flushes.append(None))
    asyncio.run(model.set_image(_image_upload(), "A quiet road"))
    flushes.clear()

    asyncio.run(model.reset(-1))

    assert flushes == [None]


def test_prompt_applies_at_next_chunk_boundary() -> None:
    """Sample a queued prompt exactly at the next chunk boundary."""
    model, backend = _ready_model()
    asyncio.run(model.set_image(_image_upload(), "First prompt"))

    async def drain_first_chunk() -> None:
        stream = model.inference()
        await anext(stream)
        await stream.aclose()

    asyncio.run(drain_first_chunk())
    asyncio.run(model.set_prompt("Second prompt"))

    async def generate_next_chunk() -> Any:
        return await anext(model.inference())

    output = asyncio.run(generate_next_chunk())
    assert isinstance(output, HYWorld15Output)
    assert output.main_video.shape == (16, 8, 8, 3)
    assert [prompt for _, prompt in backend.calls] == ["First prompt", "Second prompt"]
    assert model._chunk_index == 2
