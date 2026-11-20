"""Finite conditioning preserves every native latent and successful boundary."""

import asyncio
import json

import numpy as np
import pytest
from hy_world_1_5_camera import load_camera_path
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from test_hy_world_1_5 import _image_upload, _ready_model


def payload(count=48):
    views = np.tile(np.eye(4), (count, 1, 1))
    views[:, 0, 3] = np.arange(count) * 0.08
    return {
        "viewmats": views.tolist(),
        "intrinsics": np.tile(np.eye(3), (count, 1, 1)).tolist(),
        "actions": (np.arange(count) % 81).tolist(),
    }


def upload(value):
    return UploadedFile(
        name="path.json", mime_type="application/json", data=json.dumps(value).encode()
    )


def test_twelve_chunks_match_all_conditions_and_finite_stop():
    async def run():
        app, backend = _ready_model()
        await app.set_image(_image_upload(), prompt="world")
        data = payload()
        selected = await app.set_camera_path(upload(data))
        assert (selected.chunks, selected.frames) == (12, 189)
        assert app._state_update().max_chunks == 12
        with pytest.raises(CommandError):
            await app.set_camera(forward=1, strafe=0, yaw=0, pitch=0)
        for index in range(12):
            step = await app.process_input()
            repeated = await app.process_input()
            np.testing.assert_array_equal(step.actions, repeated.actions)
            np.testing.assert_allclose(
                step.viewmats,
                np.asarray(data["viewmats"])[index * 4 : index * 4 + 4],
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                step.actions, data["actions"][index * 4 : index * 4 + 4]
            )
            result = app.generate(step)
            await app.process_output(StepOutcome(result=result, elapsed=0.1))
            assert len(result.frames) == (13 if index == 0 else 16)
        assert len(backend.resets) == 1
        # Upstream draws the whole path's noise at once; the backend needs its length.
        assert backend.latent_frames == 48
        with pytest.raises(ApplicationError):
            await app.process_input()
        await app.reset(seed=3)
        assert (await app.process_input()).actions[0] == 0
        await app.set_image(_image_upload(), prompt="another")
        assert app._camera_path is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    ["short", "tail", "nan", "rotation", "action", "float_action", "intrinsics"],
)
def test_invalid_path(change):
    data = payload(8)
    if change in ("short", "tail"):
        data = payload(3 if change == "short" else 7)
    elif change == "nan":
        data["viewmats"][0][0][0] = float("nan")
    elif change == "rotation":
        data["viewmats"][0][0][0] = 2
    elif change == "action":
        data["actions"][0] = 81
    elif change == "float_action":
        data["actions"][0] = 1.2
    else:
        data["intrinsics"][0][0][0] = -1
    with pytest.raises(ValueError):
        load_camera_path(json.dumps(data).encode(), 512)


def test_invalid_upload_preserves_existing_path_and_failure_does_not_advance():
    async def run():
        app, _ = _ready_model()
        await app.set_image(_image_upload(), prompt="world")
        await app.set_camera_path(upload(payload(8)))
        before = app._camera_path
        with pytest.raises(CommandError):
            await app.set_camera_path(upload(payload(7)))
        assert app._camera_path is before
        await app.process_input()
        with pytest.raises(RuntimeError):
            await app.process_output(
                StepOutcome(error=RuntimeError("test"), elapsed=0.1)
            )
        assert app._chunk_index == 0
        np.testing.assert_array_equal(
            (await app.process_input()).actions, before[0].actions
        )

    asyncio.run(run())


def test_native_reflected_third_person_basis_is_preserved():
    data = payload(8)
    for view in data["viewmats"]:
        view[0][0] = -1
    path = load_camera_path(json.dumps(data).encode(), 512)
    np.testing.assert_array_equal(path[0].viewmats[:, 0, 0], [-1] * 4)
    data["viewmats"][-1][0][0] = 1
    with pytest.raises(ValueError):
        load_camera_path(json.dumps(data).encode(), 512)


def test_path_keeps_float64_values_for_memory_selection():
    data = payload(8)
    data["viewmats"][1][0][3] = 0.1 + 1e-12
    path = load_camera_path(json.dumps(data).encode(), 512)
    assert path[0].viewmats.dtype == np.float64
    assert path[0].intrinsics.dtype == np.float64
    assert path[0].viewmats[1, 0, 3] == 0.1 + 1e-12


def test_live_controls_draw_noise_per_chunk():
    async def run():
        app, backend = _ready_model()
        await app.set_image(_image_upload(), prompt="world")
        result = app.generate(await app.process_input())
        await app.process_output(StepOutcome(result=result, elapsed=0.1))
        assert backend.latent_frames is None

    asyncio.run(run())
