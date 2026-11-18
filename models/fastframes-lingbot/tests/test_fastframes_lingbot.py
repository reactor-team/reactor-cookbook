import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from fastframes_lingbot import FastFramesLingBotApp
from fastframes_lingbot_images import prepare_image
from fastframes_lingbot_model import CHUNKS, FastFramesLingBotModel, LingBotResult, WorldStateError
from fastframes_lingbot_types import WorldState

CONTROLLER = SimpleNamespace(id="controller")
VIEWER = SimpleNamespace(id="viewer")
FRAMES = np.zeros((16, 2, 3, 3), dtype=np.uint8)


def upload() -> UploadedFile:
    data = BytesIO()
    Image.new("RGB", (80, 24), (10, 50, 80)).save(data, "PNG")
    return UploadedFile(name="scene.png", mime_type="image/png", data=data.getvalue())


def make_app() -> FastFramesLingBotApp:
    app = FastFramesLingBotApp()
    app.state = WorldState()
    app.engine = Mock()
    app.send = AsyncMock()
    return app


async def start(app: FastFramesLingBotApp) -> int:
    return (await app.start_world(CONTROLLER, upload(), "A village", 7)).world_id


async def deliver(app, step, chunk):
    result = LingBotResult(step.world_id, step.prompt_revision, step.action, chunk, FRAMES)
    return await app.process_output(StepOutcome(result=result, elapsed=0.21, error=None))


def test_anchor_motion_and_automatic_stationary_steps():
    async def exercise():
        app = make_app()
        with pytest.raises(ApplicationError):
            await app.process_input()
        world = await start(app)
        await app.move_camera(CONTROLLER, world, "forward", False)
        first = await app.process_input()
        assert first.anchor.seed == 7
        assert first.anchor.image.shape == (464, 832, 3)
        assert not first.anchor.image.flags.writeable
        assert first.action == "idle"
        await deliver(app, first, 1)
        second = await app.process_input()
        assert second.anchor is None and second.action == "forward"
        await deliver(app, second, 2)
        third = await app.process_input()
        assert third.anchor is None and third.action == "idle"
        assert app._status().max_chunks == 256
    asyncio.run(exercise())


def test_hold_release_blur_and_latest_tap():
    async def exercise():
        app = make_app()
        world = await start(app)
        await deliver(app, await app.process_input(), 1)
        await app.move_camera(CONTROLLER, world, "left", False)
        await app.move_camera(CONTROLLER, world, "yaw_right", False)
        await app.release_camera(CONTROLLER, world, False)
        assert (await app.process_input()).action == "yaw_right"
        await app.move_camera(CONTROLLER, world, "pitch_up", True)
        assert (await app.process_input()).action == "pitch_up"
        assert (await app.process_input()).action == "pitch_up"
        await app.release_camera(CONTROLLER, world, False)
        assert (await app.process_input()).action == "idle"
        await app.move_camera(CONTROLLER, world, "pitch_down", True)
        await app.release_camera(CONTROLLER, world, False)
        assert (await app.process_input()).action == "pitch_down"
        assert (await app.process_input()).action == "idle"
        await app.move_camera(CONTROLLER, world, "down", False)
        await app.release_camera(CONTROLLER, world, True)
        assert (await app.process_input()).action == "idle"
    asyncio.run(exercise())


def test_owner_refusals_and_disconnect():
    async def exercise():
        app = make_app()
        world = await start(app)
        app.send.reset_mock()
        with pytest.raises(CommandError) as error:
            await app.move_camera(VIEWER, world, "down", True)
        assert error.value.code == "controller_required"
        with pytest.raises(CommandError) as error:
            await app.start_world(VIEWER, upload(), "A village", 0)
        assert error.value.code == "world_in_use"
        app.send.assert_not_awaited()
        await app.move_camera(CONTROLLER, world, "down", True)
        await app.on_disconnected(VIEWER)
        assert app.state._held == "down"
        await app.on_disconnected(CONTROLLER)
        app.engine.reset.assert_called_once()
        assert app.state._held is None and app.state._anchor is None
        with pytest.raises(CommandError) as error:
            await app.move_camera(CONTROLLER, world, "down", False)
        assert error.value.code == "stale_world"
        await app.start_world(VIEWER, upload(), "Another village", 0)
    asyncio.run(exercise())


def test_prompt_applies_during_idle_and_horizon_ends_generation():
    async def exercise():
        app = make_app()
        world = await start(app)
        await deliver(app, await app.process_input(), 1)
        accepted = await app.update_prompt(CONTROLLER, world, "A dark village")
        assert accepted.revision == 1 and app._status().prompt_pending
        idle = await app.process_input()
        assert idle.action == "idle"
        assert idle.prompt == "A dark village" and idle.prompt_revision == 1
        await deliver(app, idle, 2)
        assert not app._status().prompt_pending
        await app.move_camera(CONTROLLER, world, "up", True)
        step = await app.process_input()
        assert step.prompt == "A dark village" and step.prompt_revision == 1
        await deliver(app, step, CHUNKS)
        assert app._status().phase == "complete"
        assert app._status().held_action is None and not app._status().prompt_pending
        with pytest.raises(ApplicationError):
            await app.process_input()
    asyncio.run(exercise())


def test_stale_result_and_model_error():
    async def exercise():
        app = make_app()
        await start(app)
        step = await app.process_input()
        await start(app)
        assert await deliver(app, step, 1) is None
        assert app._status().output_frames == 0
        app.send.reset_mock()
        with pytest.raises(RuntimeError, match="GPU step failed"):
            await app.process_output(StepOutcome(result=None, elapsed=0.2, error=RuntimeError("GPU step failed")))
        app.send.assert_not_awaited()
    asyncio.run(exercise())


def test_invalid_image():
    with pytest.raises(ValueError, match="readable"):
        prepare_image(b"not an image")


def test_model_starts_once_and_propagates_native_failure():
    async def make_steps():
        app = make_app()
        world = await start(app)
        first = await app.process_input()
        await deliver(app, first, 1)
        await app.move_camera(CONTROLLER, world, "left", False)
        return first, await app.process_input()

    first, second = asyncio.run(make_steps())
    model = FastFramesLingBotModel()
    model.backend = Mock()
    model.backend.step.return_value = FRAMES
    model.backend.chunk = 1
    model.world_id = None
    with pytest.raises(WorldStateError):
        model.generate(second)
    model.generate(first)
    model.backend.start.assert_called_once_with(first.anchor, first.prompt)
    model.backend.step.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        model.generate(second)
    model.backend.start.assert_called_once()
    model.backend.step.assert_called_with("left")
