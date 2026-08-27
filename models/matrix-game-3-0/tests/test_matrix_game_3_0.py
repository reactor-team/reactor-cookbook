from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from PIL import Image
from reactor_runtime import UploadedFile

from matrix_game_3_0 import MatrixGame30
from matrix_game_3_0_backend import MatrixGame30Backend, action_from_controls
from matrix_game_3_0_images import normalize_output_frames
from matrix_game_3_0_types import MatrixGame30State


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_native_control_mapping_preserves_discrete_keys_and_continuous_camera() -> None:
    action = action_from_controls(frozenset(("w", "a")), -0.5, 1.0)

    assert action.keyboard == (1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert action.mouse == (-0.05, 0.1)


def test_native_chunk_frame_counts_are_enforced() -> None:
    first = np.zeros((57, 8, 8, 3), dtype=np.uint8)
    later = np.zeros((40, 8, 8, 3), dtype=np.uint8)

    assert normalize_output_frames(first, 0).shape[0] == 57
    assert normalize_output_frames(later, 1).shape[0] == 40


def test_session_waits_for_explicit_image_selection() -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State()
    model._config = SimpleNamespace(seed=42, max_chunks=12)

    class Backend:
        def generate_chunk(self, _action: object) -> np.ndarray:
            raise AssertionError("generation must wait for image selection")

    model._backend = Backend()

    model.on_session_started()
    message = model._state_update()

    async def first_turn() -> object:
        generator = model.inference()
        output = await anext(generator)
        await generator.aclose()
        return output

    assert model._selected_input is None
    assert model.state._restart_requested is False
    assert message.image_source == "none"
    assert message.next_chunk is None
    assert message.next_chunk_frames is None
    assert asyncio.run(first_turn()) is None


def test_uploaded_image_and_prompt_start_a_fresh_rollout(tmp_path: Path) -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="original")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = tmp_path / "original.png"

    message = model.set_image(
        UploadedFile(name="anchor.png", mime_type="image/png", data=_png_bytes()),
        "replacement",
    )

    assert model.state._restart_requested is True
    assert message.prompt == "replacement"
    assert message.image_source == "uploaded"
    assert message.next_chunk_frames == 57


def test_uploaded_image_without_prompt_starts_a_fresh_rollout(
    tmp_path: Path,
) -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="original")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = tmp_path / "original.png"

    message = model.set_image(
        UploadedFile(name="anchor.png", mime_type="image/png", data=_png_bytes()),
        "",
    )

    assert model.state._restart_requested is True
    assert message.prompt == ""
    assert message.image_source == "uploaded"
    assert message.next_chunk_frames == 57


def test_control_events_hold_discrete_keys_and_continuous_camera() -> None:
    model = MatrixGame30()
    model.state = MatrixGame30State(prompt="scene")
    model._config = SimpleNamespace(max_chunks=12)
    model._selected_input = Path("anchor.png")
    model.send = AsyncMock()

    async def apply_controls() -> None:
        key_message = await model.set_key_state("w", True)
        pitch_message = await model.set_pitch(0.5)
        yaw_message = await model.set_yaw(-1.0)
        assert key_message.pressed_keys == ["w"]
        assert pitch_message.pitch == 0.5
        assert yaw_message.yaw == -1.0

    asyncio.run(apply_controls())

    action = action_from_controls(
        model.state._pressed_keys,
        model.state.pitch,
        model.state.yaw,
    )
    assert action.keyboard == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert action.mouse == (0.05, -0.1)
    assert model.send.await_count == 3


def test_backend_bridges_one_action_to_each_unmodified_iteration(
    tmp_path: Path,
) -> None:
    original_action = object()
    original_video = object()
    module = SimpleNamespace(
        get_current_action=original_action,
        process_video=original_video,
    )
    seen_actions: list[dict[str, object]] = []

    class FakePipeline:
        def generate(self, *_args: object, **_kwargs: object) -> None:
            for index, frame_count in enumerate((57, 40)):
                seen_actions.append(module.get_current_action())
                module.process_video(
                    np.full((frame_count, 4, 6, 3), index, dtype=np.uint8),
                    str(tmp_path / f"reactor_current_iteration_{index}.mp4"),
                    None,
                    None,
                )

    config = SimpleNamespace(
        chunk_timeout_seconds=10.0,
        size="704*1280",
        sample_shift=5.0,
        num_inference_steps=3,
        guide_scale=5.0,
    )
    backend = MatrixGame30Backend(config)
    backend._module = module
    backend._pipeline = FakePipeline()
    backend._args = SimpleNamespace()

    image = tmp_path / "anchor.png"
    Image.new("RGB", (16, 16)).save(image)
    backend.reset("scene", 42, image)
    first = backend.generate_chunk(action_from_controls(frozenset(("w",)), 0.0, 0.0))
    second = backend.generate_chunk(action_from_controls(frozenset(("d",)), 0.0, -1.0))
    backend.end_session()

    assert first.shape == (57, 4, 6, 3)
    assert second.shape == (40, 4, 6, 3)
    assert len(seen_actions) == 2
    assert module.get_current_action is original_action
    assert module.process_video is original_video
