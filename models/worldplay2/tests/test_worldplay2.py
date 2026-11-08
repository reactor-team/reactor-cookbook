"""CPU regression tests for the Cookbook entrypoint and native chunk boundary."""

import asyncio
import inspect
import unittest
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

import worldplay2_app
from worldplay2_app import WorldPlay2
from worldplay2_model import (
    MAX_CHUNKS,
    EncodePrompt,
    Prepared,
    PrepareWorld,
    WorldPlay2Input,
    WorldPlay2Model,
    WorldPlay2Result,
    load_settings,
)
from worldplay2_types import ChunkGenerated, RolloutCompleted, WorldPlay2State

ROWS = ((0.0, 0.0, 0.0, 0.0, 1.0, 0.0),) * 4


def upload():
    content = BytesIO()
    Image.new("RGB", (32, 32), "green").save(content, format="PNG")
    return SimpleNamespace(name="garden.png", data=content.getvalue())


class FakeEngine:
    def __init__(self):
        self.inputs = []
        self.world_id = None
        self.index = 0
        self.error = None
        self.reset = Mock()

    def generate(self, value):
        self.inputs.append(value)
        if self.error:
            raise self.error
        if isinstance(value, EncodePrompt):
            return Prepared(world_id=None)
        if isinstance(value, PrepareWorld):
            self.world_id, self.index = value.world_id, 0
            return Prepared(world_id=value.world_id)
        if value.world_id != self.world_id:
            self.world_id, self.index = value.world_id, 0
        self.index += 1
        return WorldPlay2Result(
            world_id=value.world_id,
            chunk_index=self.index,
            frames=np.zeros((13 if self.index == 1 else 16, 8, 8, 3), np.uint8),
        )


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = WorldPlay2()
        self.app.state = WorldPlay2State()
        self.engine = FakeEngine()
        self.app._engine = self.engine
        self.app.send = AsyncMock()
        self.app.on_session_started()
        self.planner = patch.object(worldplay2_app, "plan_action", return_value=ROWS)
        self.planner.start()

    async def asyncTearDown(self):
        self.planner.stop()
        self.app._loop.close()

    async def select_image(self):
        return await self.app.set_image(
            upload(), "The scene in the video is: a garden."
        )

    async def step(self):
        value = await self.app.process_input()
        result = self.app.generate(value)
        await self.app.process_output(StepOutcome(result=result, elapsed=0.25))
        return value, result

    async def test_refusals_do_not_call_the_engine(self):
        with self.assertRaises(ApplicationError):
            await self.app.process_input()
        self.assertEqual(self.engine.inputs, [])
        await self.select_image()
        await self.app.set_prompt("")
        before = len(self.engine.inputs)
        with self.assertRaises(ApplicationError):
            await self.app.process_input()
        self.assertEqual(len(self.engine.inputs), before)

    async def test_invalid_upload_and_reset_raise_without_broadcasting(self):
        for call in (
            self.app.set_image(SimpleNamespace(name="bad.png", data=b"invalid"), ""),
            self.app.reset(42),
        ):
            with self.assertRaises(CommandError):
                await call
        self.assertEqual(self.engine.inputs, [])
        self.app.send.assert_not_awaited()

    async def test_anchor_until_acknowledged_and_live_prompt_keeps_world(self):
        await self.select_image()
        first = await self.app.process_input()
        self.assertIsNotNone(first.anchor)
        self.assertIsNotNone((await self.app.process_input()).anchor)
        await self.step()
        self.assertIsNone((await self.app.process_input()).anchor)
        await self.app.set_prompt("The scene in the video is: a rainy garden.")
        value, result = await self.step()
        self.assertEqual(value.world_id, first.world_id)
        self.assertIsNone(value.anchor)
        self.assertIn("rainy", value.prompt)
        self.assertEqual(result.chunk_index, 2)

    async def test_generate_reads_only_the_snapshot(self):
        await self.select_image()
        value = await self.app.process_input()
        self.app.state = None
        self.app.generate(value)
        self.assertIs(self.engine.inputs[-1], value)

    async def test_failed_prepare_reanchors_with_real_model(self):
        model = WorldPlay2Model()
        fail = False
        frames = np.zeros((16, 8, 8, 3), np.uint8)

        def stream_world(**kwargs):
            nonlocal fail
            if fail:
                fail = False
                raise RuntimeError("preparation failed once")

            def stream():
                yield frames
                yield frames

            return stream()

        model._pipeline = SimpleNamespace(
            stream_world=stream_world,
            clear_conditioning_cache=Mock(),
            vae=SimpleNamespace(model=SimpleNamespace(clear_cache=Mock())),
        )
        model._host_frames = lambda rgb: rgb
        self.app._engine = model
        await self.select_image()
        await self.step()
        original_image = self.app._image
        original_prompt = self.app.state.prompt
        for command in ("set_image", "reset"):
            fail = True
            with self.assertRaises(CommandError) as error:
                if command == "set_image":
                    await self.app.set_image(upload(), "new scene")
                else:
                    await self.app.reset(77)
            self.assertEqual(error.exception.code, "prepare_failed")
            self.assertIsNone(self.app._active_world)
            value, result = await self.step()
            self.assertEqual(value.anchor.image, original_image)
            self.assertEqual(value.anchor.seed, 42)
            self.assertEqual(value.prompt, original_prompt)
            self.assertEqual(result.chunk_index, 1)

    async def test_replaced_world_result_does_not_emit_or_mutate(self):
        await self.select_image()
        old = self.app.generate(await self.app.process_input())
        await self.app.reset(77)
        before = self.app.send.await_count
        output = await self.app.process_output(StepOutcome(result=old, elapsed=0.1))
        self.assertIsNone(output)
        self.assertEqual(self.app.send.await_count, before)
        self.assertEqual(self.app._completed, 0)

    async def test_failure_does_not_complete_or_acknowledge_a_chunk(self):
        await self.select_image()
        value = await self.app.process_input()
        error = RuntimeError("fake compute failed")
        self.engine.error = error
        with self.assertRaises(RuntimeError):
            self.app.generate(value)
        with self.assertRaises(RuntimeError):
            await self.app.process_output(StepOutcome(error=error, elapsed=0.1))
        self.assertEqual(self.app._completed, 0)
        self.assertIsNotNone((await self.app.process_input()).anchor)

    async def test_reset_and_new_session_start_clean(self):
        await self.select_image()
        first, _ = await self.step()
        await self.app.reset(7)
        value = await self.app.process_input()
        self.assertGreater(value.world_id, first.world_id)
        self.assertEqual(value.anchor.seed, 7)
        self.assertEqual(self.app._completed, 0)
        self.app.on_session_ended()
        self.engine.reset.assert_called_once()
        self.app.state = WorldPlay2State()
        self.app.on_session_started()
        self.assertIsNone(self.app._image)
        self.assertEqual(self.app._completed, 0)
        with self.assertRaises(ApplicationError):
            await self.app.process_input()

    async def test_horizon_reports_final_chunk_then_completion(self):
        await self.select_image()
        self.engine.index = MAX_CHUNKS - 1
        _, result = await self.step()
        messages = [call.args[0] for call in self.app.send.await_args_list]
        self.assertEqual(result.chunk_index, MAX_CHUNKS)
        self.assertIsInstance(messages[-3], ChunkGenerated)
        self.assertIsInstance(messages[-1], RolloutCompleted)
        self.assertEqual(messages[-3].elapsed_seconds, 0.25)
        with self.assertRaises(ApplicationError):
            await self.app.process_input()

    async def test_runtime_loop_drives_ten_chunks_with_fake_compute(self):
        self.assertTrue(inspect.iscoroutinefunction(self.app.process_input))
        self.assertTrue(inspect.iscoroutinefunction(self.app.process_output))
        await self.select_image()
        outputs = []
        done = asyncio.Event()

        async def emit(media, **kwargs):
            outputs.append(media)
            if len(outputs) == 10:
                self.app._live.clear()
                done.set()

        self.app.emit = emit
        self.app._on_loop_ready()
        self.app._live.set()
        task = asyncio.create_task(self.app.run())
        try:
            await asyncio.wait_for(done.wait(), 5)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        steps = [
            value for value in self.engine.inputs if isinstance(value, WorldPlay2Input)
        ]
        self.assertEqual(len(steps), 10)
        self.assertIsNotNone(steps[0].anchor)
        self.assertTrue(all(value.anchor is None for value in steps[1:]))
        self.assertEqual(self.app._completed, 10)


class ModelTests(unittest.TestCase):
    def test_flat_import_schema_and_config(self):
        contract = ModelContract.of(WorldPlay2)
        self.assertEqual(
            set(contract.commands),
            {
                "set_image",
                "reset",
                "set_prompt",
                "set_action",
                "set_perspective",
                "set_yaw_degrees",
                "set_pitch_degrees",
            },
        )
        config = load_settings(Path(__file__).resolve().parents[1] / "worldplay2.yaml")
        self.assertEqual(
            (config.world_size, config.height, config.width), (4, 448, 832)
        )
        self.assertEqual(MAX_CHUNKS, 256)

    def test_frozen_contract_and_model_chunk_bookkeeping(self):
        model = WorldPlay2Model()
        model._pipeline = SimpleNamespace(
            clear_conditioning_cache=Mock(),
            vae=SimpleNamespace(model=SimpleNamespace(clear_cache=Mock())),
        )
        model._world_id = 1
        frames = np.zeros((16, 8, 8, 3), np.uint8)
        model._stream = iter([frames, frames])
        model._host_frames = lambda rgb: rgb
        value = WorldPlay2Input(1, None, "garden", "none", ROWS)
        with self.assertRaises(FrozenInstanceError):
            value.world_id = 2
        self.assertEqual(model.generate(value).chunk_index, 1)
        self.assertEqual(model.generate(value).chunk_index, 2)

    def test_model_failure_clears_world_and_propagates(self):
        model = WorldPlay2Model()
        clear = Mock()
        model._pipeline = SimpleNamespace(
            clear_conditioning_cache=clear,
            vae=SimpleNamespace(model=SimpleNamespace(clear_cache=Mock())),
        )
        model._world_id = 1

        def broken():
            raise RuntimeError("native failure")
            yield

        model._stream = broken()
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            model.generate(WorldPlay2Input(1, None, "garden", "none", ROWS))
        self.assertIsNone(model._stream)
        self.assertIsNone(model._world_id)
        self.assertEqual(model._chunk_index, 0)
        clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
