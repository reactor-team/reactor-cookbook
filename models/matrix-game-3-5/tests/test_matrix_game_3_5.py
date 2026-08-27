"""Check Matrix's upload-gated session startup contract."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from reactor_runtime import CommandError, UploadedFile
from reactor_runtime.interface.model.contract import ModelContract

from matrix_game_3_5 import MatrixGame35
from matrix_game_3_5_types import MatrixGame35State

MODEL_DIR = Path(__file__).parents[1]

GENERIC_PROMPT = (
    "An immersive first-person view that faithfully continues the input scene, "
    "preserving its existing environment, objects, geometry, materials, lighting, "
    "and visual style as the camera moves naturally through it."
)


def _upload() -> UploadedFile:
    """Return a small valid anchor upload."""
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(payload, format="PNG")
    return UploadedFile(
        name="anchor.png",
        mime_type="image/png",
        data=payload.getvalue(),
    )


def _model() -> MatrixGame35:
    """Return a loaded-enough model for lifecycle and command checks."""
    model = MatrixGame35()
    model.state = MatrixGame35State()
    model._config = SimpleNamespace(seed=3407, max_chunks=512)
    model._default_prompt = GENERIC_PROMPT
    return model


def test_session_waits_for_an_image_selection() -> None:
    """Wait for the viewer's anchor instead of generating from the demo image."""
    model = _model()

    model.on_session_started()
    state = model._state_update()

    assert model._selected_input is None
    assert state.image_source == "none"
    assert state.image_name == ""
    assert state.completed_chunks == 0
    assert state.next_chunk is None
    assert set(ModelContract.of(MatrixGame35).commands) == {
        "reset",
        "set_forward",
        "set_image",
        "set_pitch",
        "set_prompt",
        "set_roll",
        "set_strafe",
        "set_vertical",
        "set_yaw",
    }


def test_first_upload_starts_continuous_generation() -> None:
    """Start continuous generation from the uploaded anchor."""
    model = _model()
    model.on_session_started()

    state = model.set_image(_upload(), "")

    assert state.image_source == "uploaded"
    assert state.image_name == "anchor.png"
    assert state.completed_chunks == 0
    assert state.next_chunk == 1
    assert state.prompt == GENERIC_PROMPT
    assert model.state._restart_requested is True


def test_generation_controls_require_an_uploaded_image() -> None:
    """Reject controls that cannot produce a chunk before anchor selection."""
    model = _model()
    model.on_session_started()

    with pytest.raises(CommandError) as error:
        model.set_forward(1.0)

    assert error.value.code == "image_required"
