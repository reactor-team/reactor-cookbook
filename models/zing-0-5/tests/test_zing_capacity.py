"""Capacity never authorizes an implicit world reset."""

import asyncio
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from reactor_runtime import CommandError
from zing import Zing
from zing_assets import read_config
from zing_types import ZingState


def test_limit_retains_world_and_releases_controls():
    """Exhaustion freezes the backend until an explicit reset creates a new epoch."""

    class Backend:
        resets = 0
        calls = 0

        def reset(self, **kwargs):
            self.resets += 1

        def generate_chunk(self, **kwargs):
            self.calls += 1
            return np.zeros((16, 8, 8, 3), dtype=np.uint8)

        def cache_frames(self):
            return self.calls * 4

    model = Zing()
    model.state = ZingState()
    model._config = replace(
        read_config(Path(__file__).parents[1] / "zing.yaml"), max_chunks=2
    )
    model._backend = backend = Backend()
    model.on_session_started()
    messages = []

    async def record(message):
        messages.append(message)

    model.send = record

    async def run():
        await model.set_prompt("A courtyard")
        epoch = model._world_epoch
        output = model.inference()
        assert await anext(output) is not None
        await model.set_key("w", True)
        assert await anext(output) is not None
        assert model._state_update().limit_reached
        assert model._world_epoch == epoch
        assert model._completed_chunks == 2
        assert not model.state._pressed_keys
        for _ in range(3):
            assert await anext(output) is None
        assert (backend.resets, backend.calls) == (1, 2)
        assert (
            sum(type(message).__name__ == "RolloutLimitReached" for message in messages)
            == 1
        )
        with pytest.raises(CommandError):
            await model.set_key("w", True)
        await model.set_key("w", False)
        await model.release_controls()
        await model.reset(42)
        assert model._world_epoch == epoch + 1
        assert not model._limit_reached
        assert await anext(output) is not None
        assert (backend.resets, backend.calls) == (2, 3)
        await output.aclose()

    asyncio.run(run())
