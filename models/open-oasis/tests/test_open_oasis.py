"""Contract, native actions, uploads, and one-frame boundaries for Open-Oasis."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from reactor_runtime.interface.model.contract import ModelContract

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

from open_oasis import OpenOasis
from open_oasis_assets import decode_image
from open_oasis_types import OpenOasisState


def ready_model() -> OpenOasis:
    model = OpenOasis()
    model.state = OpenOasisState()
    model._config = SimpleNamespace(seed=0)
    model._source = Path("/unused")
    model._conditioning = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    model.send = lambda _message: _awaitable()  # type: ignore[method-assign]
    model.output = SimpleNamespace(flush=lambda: None)
    return model


async def _awaitable() -> None:
    return None


def test_schema_is_complete_and_precise() -> None:
    contract = ModelContract.of(OpenOasis)
    assert set(contract.commands) == {
        "mouse_move",
        "random_scene",
        "release_controls",
        "reset",
        "set_image",
        "set_video",
        "set_key_state",
        "set_mouse_button_state",
    }
    assert all("Emits" in command.description for command in contract.commands.values())
    assert all(
        field.info.description
        for command in contract.commands.values()
        for field in command.command.__command_fields__.values()
    )
    document = contract.render_schema().to_openapi()
    assert document["x-reactor"]["tracks"] == [
        {"name": "main_video", "kind": "video", "direction": "out"}
    ]
    assert set(document["webhooks"]) == {
        "action_changed",
        "conditioning_changed",
        "rollout_reset",
        "state_update",
    }
    assert (
        document["paths"]["/events/set_video"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["properties"]["prompt_frames"]["maximum"]
        == 32
    )


def test_native_actions_cover_all_twenty_five_dimensions() -> None:
    model = ready_model()
    model.state._pressed_keys = frozenset({"w", "space", "e", "9"})
    model.state._pressed_mouse_buttons = frozenset({"left", "right", "middle"})
    model.state._camera_x = -0.5
    model.state._camera_y = 0.75
    action = model._build_action()
    assert action.shape == (25,)
    assert np.count_nonzero(action) == 9
    assert action[15] == -0.5 and action[16] == 0.75


def test_camera_accumulates_then_release_clears_everything() -> None:
    model = ready_model()
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.mouse_move(0.75, -0.8))
    reply = asyncio.run(model.mouse_move(0.75, -0.8))
    assert reply.camera_x == 1 and reply.camera_y == -1
    released = asyncio.run(model.release_controls())
    assert released.pressed_keys == [] and released.camera_x == 0


def test_uploaded_image_is_rgb_native_resolution() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (40, 20), (2, 4, 8, 128)).save(buffer, format="PNG")
    frame = decode_image(buffer.getvalue())
    assert frame.shape == (1, 360, 640, 3)
    assert frame.dtype == np.uint8


def test_conditioning_upload_is_moderated() -> None:
    fields = ModelContract.of(OpenOasis).commands
    assert fields["set_image"].command.__command_fields__["image"].info.moderate
    assert fields["set_video"].command.__command_fields__["video"].info.moderate


def test_new_connection_waits_and_disconnect_discards_upload() -> None:
    model = ready_model()
    model._backend = object()  # type: ignore[assignment]
    model.on_session_started()
    assert model._conditioning is None
    assert model._conditioning_name == "none"

    model._conditioning = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    model._conditioning_name = "uploaded.png"
    asyncio.run(model.on_disconnected())

    assert model._conditioning is None
    assert model._conditioning_name == "none"
    assert next(model.inference()) is None


def test_playback_contract_uses_one_frame_chunks_without_explicit_fps() -> None:
    assert "fps" not in OpenOasis.__dict__
    assert OpenOasis.buffer_size == 1


def test_press_and_release_before_sampling_still_produces_one_frame_pulse() -> None:
    model = ready_model()
    asyncio.run(model.set_key_state("w", True))
    asyncio.run(model.set_key_state("w", False))

    assert model.state._pressed_keys == frozenset()
    assert model._build_action()[11] == 1

    model.state._pending_key_pulses = frozenset()
    assert model._build_action()[11] == 0
