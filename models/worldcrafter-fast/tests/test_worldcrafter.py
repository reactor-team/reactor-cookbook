"""Exercise the Reactor contract and world transitions without GPU weights."""
import asyncio
from dataclasses import FrozenInstanceError
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

import worldcrafter
from upstream_backend import InferenceWorkerError
from worldcrafter import WorldCrafter
from worldcrafter_model import (
    InvalidPoses, NoAnchor, RolloutExhausted, WorldCrafterInput, WorldCrafterModel,
    WorldCrafterResult,
)
from worldcrafter_types import WorldCrafterState, ChunkGenerated
from worldcrafter_camera import FRAMES, CameraPlanner


def run(awaitable):
    return asyncio.run(awaitable)


def ready():
    app = WorldCrafter()
    app.state = WorldCrafterState()
    messages = []

    async def send(message):
        messages.append(message)

    app.send = send
    return app, messages


def image():
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (20, 80, 120)).save(buffer, format="PNG")
    return SimpleNamespace(data=buffer.getvalue(), name="scene.png")


def start(app):
    return run(app.start_world(image(), "A stone courtyard, first person.", 42, True))


def result(world=1, chunk=0):
    return WorldCrafterResult(world_id=world, chunk_index=chunk,
        frames=np.zeros((33, 384, 640, 3), dtype=np.uint8),
        camera_end=np.eye(4)[:3].tolist(), gravity_aligned=True, gpu_count=8,
        world_up=[0., -1., 0.])


def test_schema_surface():
    contract = ModelContract.of(WorldCrafter)
    assert set(contract.commands) == {"start_world", "move", "set_camera_path", "pause", "resume", "reset"}


def test_playout_follows_generation_time():
    assert "fps" not in vars(WorldCrafter)
    assert WorldCrafter.buffer_size == 2 * FRAMES


def test_refused_step_never_calls_model():
    app, messages = ready()
    with pytest.raises(ApplicationError):
        run(app.process_input())
    assert not messages and app._engine is None


def test_refused_command_has_no_success_message():
    app, messages = ready()
    with pytest.raises(CommandError) as error:
        run(app.move("forward", 1, 1))
    assert error.value.code == "image_required"
    assert messages == []


def test_anchor_only_until_ack_and_runtime_elapsed():
    app, messages = ready()
    start(app)
    first = run(app.process_input())
    assert first.image and first.world_id == 1 and np.allclose(first.world_poses, np.eye(4)[:3])
    with pytest.raises(FrozenInstanceError):
        first.seed = 12
    output = run(app.process_output(StepOutcome(result=result(), elapsed=1.25)))
    assert output.main_video.shape == (33, 384, 640, 3)
    event = next(message for message in messages if isinstance(message, ChunkGenerated))
    assert event.generation_seconds == 1.25 and event.gpu_count == 8
    run(app.move("yaw_left", 20, 1))
    second = run(app.process_input())
    assert second.image is None and second.world_poses.shape == (66, 3, 4) and second.world_poses[-1, 0, 2] < 0


def test_reset_rejects_stale_result_and_restarts_camera():
    app, messages = ready()
    start(app)
    run(app.process_input())
    run(app.reset())
    count = len(messages)
    assert run(app.process_output(StepOutcome(result=result()))) is None
    assert len(messages) == count and app.state._completed == 0
    fresh = run(app.process_input())
    assert fresh.world_id == 2 and fresh.image and np.allclose(fresh.world_poses, np.eye(4)[:3])


def test_model_failure_is_not_a_completed_chunk():
    app, messages = ready()
    start(app)
    run(app.process_input())
    count = len(messages)
    with pytest.raises(RuntimeError, match="GPU failed"):
        run(app.process_output(StepOutcome(error=RuntimeError("GPU failed"))))
    assert len(messages) == count and app.state._completed == 0


@pytest.mark.parametrize("action,amount,reason", [
    ("forward", 6, "Translation must be 0.1 to 5 units"),
    ("yaw_left", float("nan"), "Rotation must be 0.1 to 45 degrees"),
    ("roll", 30, "supported camera direction"),
])
def test_invalid_motion_does_not_enter_queue(action, amount, reason):
    app, messages = ready()
    start(app)
    before, sent = list(app.state._queue), len(messages)
    with pytest.raises(CommandError, match=reason) as error:
        run(app.move(action, amount, 1))
    assert error.value.code == "invalid_motion"
    assert list(app.state._queue) == before and len(messages) == sent


def test_queue_limit_pause_resume_disconnect():
    app, _ = ready()
    start(app)
    run(app.move("forward", 1, 8))
    with pytest.raises(CommandError) as error:
        run(app.move("forward", 1, 8))
    assert error.value.code == "queue_full"
    run(app.pause())
    with pytest.raises(ApplicationError):
        run(app.process_input())
    run(app.resume())
    assert np.allclose(run(app.process_input()).world_poses, np.eye(4)[:3])
    before = list(app.state._queue)
    run(app.on_disconnected())
    assert list(app.state._queue) == before and not app.state._paused


def test_lifecycle_clears_image_and_resets_model():
    app, _ = ready()
    calls = []
    app._engine = SimpleNamespace(healthy=True, reset=lambda: calls.append("reset"))
    start(app)
    app.on_session_ended()
    app.state = WorldCrafterState()
    assert calls == ["reset"]
    assert app.state._image is None and not app.state._queue and app.state._completed == 0


class FakeEngine:
    def __init__(self, healthy=True, fail_reset=False):
        self.healthy, self.fail_reset, self.calls = healthy, fail_reset, []

    def reset(self):
        self.calls.append("reset")
        if self.fail_reset:
            self.healthy = False
            raise InferenceWorkerError("Inference worker did not answer")

    def close(self):
        self.calls.append("close")


@pytest.mark.parametrize("engine,expected", [
    (FakeEngine(fail_reset=True), ["reset", "close"]),
    (FakeEngine(healthy=False), ["close"]),
])
def test_session_end_replaces_an_unhealthy_worker(monkeypatch, engine, expected):
    app, _ = ready()
    app._engine = engine
    started = []
    monkeypatch.setattr(app, "_start_engine", lambda: started.append(True))
    app.on_session_ended()
    assert engine.calls == expected and started == [True]


def test_healthy_worker_is_kept_across_sessions(monkeypatch):
    app, _ = ready()
    app._engine = engine = FakeEngine()
    monkeypatch.setattr(app, "_start_engine", lambda: pytest.fail("worker restarted"))
    app.on_session_ended()
    assert engine.calls == ["reset"]


def test_generate_uses_only_snapshot():
    model = WorldCrafterModel()
    model._max_chunks = 50
    calls = []
    expected = result()
    model.backend = SimpleNamespace(generate=lambda value: calls.append(value) or expected)
    local, world = CameraPlanner().plan("static", 0)
    snapshot = WorldCrafterInput(world_id=1, image=b"image", prompt="scene", seed=0,
                               gravity_align=True, local_poses=local, world_poses=world)
    assert model.generate(snapshot) is expected and calls == [snapshot]


def test_invalid_image_does_not_replace_world():
    app, _ = ready()
    start(app)
    old = app.state._image
    with pytest.raises(CommandError) as error:
        run(app.start_world(SimpleNamespace(data=b"not image"), "scene", 0, True))
    assert error.value.code == "invalid_image"
    assert app.state._image is old and app.state._world_id == 1


def test_internal_image_bug_is_not_reported_as_invalid_upload(monkeypatch):
    app, _ = ready()

    def broken(data):
        raise KeyError("bug")

    monkeypatch.setattr(worldcrafter, "validate_image", broken)
    with pytest.raises(KeyError):
        run(app.start_world(image(), "scene", 0, True))


class FakeBackend:
    def __init__(self):
        self.healthy, self.steps, self.resets = True, [], 0

    def generate(self, input):
        index = len(self.steps)
        self.steps.append(input)
        return result(world=input.world_id, chunk=index)

    def reset(self):
        self.resets += 1
        self.steps.clear()


@pytest.fixture
def model():
    instance = WorldCrafterModel()
    instance.backend, instance._max_chunks = FakeBackend(), 2
    return instance


def step(world_id, planner, image=b"png", action="static", amount=0.0):
    local, world = planner.plan(action, amount)
    return WorldCrafterInput(world_id=world_id, image=image, prompt="scene", seed=3,
                             gravity_align=True, local_poses=local, world_poses=world)


def test_model_refuses_a_new_world_without_anchor(model):
    with pytest.raises(NoAnchor):
        model.generate(step(1, CameraPlanner(), image=None))
    assert model.backend.steps == []


def test_model_stops_at_its_chunk_limit(model):
    planner = CameraPlanner()
    model.generate(step(1, planner))
    model.generate(step(1, planner, image=None, action="forward", amount=1.0))
    with pytest.raises(RolloutExhausted):
        model.generate(step(1, planner, image=None, action="forward", amount=1.0))
    assert len(model.backend.steps) == 2


def test_model_rejects_poses_for_another_chunk(model):
    planner = CameraPlanner()
    planner.plan("static", 0.0)
    with pytest.raises(InvalidPoses):
        model.generate(step(1, planner))
    assert model.backend.steps == []


def test_model_reset_drops_the_world(model):
    model.generate(step(1, CameraPlanner()))
    model.reset()
    assert model.backend.resets == 1
    with pytest.raises(NoAnchor):
        model.generate(step(1, CameraPlanner(), image=None))


def test_failed_worker_restarts_once_and_pauses_a_fresh_world(monkeypatch):
    app, messages = ready()
    start(app)
    run(app.process_input())
    run(app.process_output(StepOutcome(result=result())))
    run(app.move("forward", 1, 4))
    run(app.process_input())
    app._engine = engine = FakeEngine(healthy=False)
    replacements = []
    monkeypatch.setattr(app, "_start_engine", lambda: replacements.append(True))
    messages.clear()
    output = run(app.process_output(StepOutcome(error=InferenceWorkerError("worker died"))))
    assert output is None and engine.calls == ["close"] and replacements == [True]
    assert app.state._inflight is None and app.state._paused
    assert app.state._completed == 0 and app.state._world_id == 2
    assert list(app.state._queue) == [("static", 0.0)]
    assert len(messages) == 1 and messages[0].paused and messages[0].completed_chunks == 0
    with pytest.raises(ApplicationError):
        run(app.process_input())
    run(app.resume())
    fresh = run(app.process_input())
    assert fresh.image == image().data and fresh.seed == 42
    assert fresh.world_id == 2 and np.allclose(fresh.world_poses, np.eye(4)[:3])


def test_replacement_failure_is_not_retried(monkeypatch):
    app, messages = ready()
    start(app)
    run(app.process_input())
    app._engine = FakeEngine(healthy=False)
    attempts = []

    def fail():
        attempts.append(True)
        raise InferenceWorkerError("replacement failed")

    monkeypatch.setattr(app, "_start_engine", fail)
    with pytest.raises(InferenceWorkerError, match="replacement failed"):
        run(app.process_output(StepOutcome(error=InferenceWorkerError("worker died"))))
    assert attempts == [True] and app.state._paused
    assert not any(isinstance(message, ChunkGenerated) for message in messages)


def test_runtime_loop_survives_worker_failure_and_emits_after_resume(monkeypatch):
    import threading

    async def scenario():
        app, messages = ready()
        app._on_loop_ready()
        app._live.set()
        await app.start_world(image(), "A stone courtyard, first person.", 42, True)
        app._engine = FakeEngine(healthy=False)
        recovered = threading.Event()
        emitted = asyncio.Event()
        calls = []

        def replace():
            recovered.set()

        def generate(snapshot):
            calls.append(snapshot)
            if len(calls) == 1:
                raise InferenceWorkerError("worker died")
            return result(world=snapshot.world_id)

        async def emit(output, **kwargs):
            assert output.main_video.shape[0] == 33
            emitted.set()

        # Exercise the installed Runtime's real step loop; stub only GPU and wire.
        monkeypatch.setattr(app, "_start_engine", replace)
        monkeypatch.setattr(app, "generate", generate)
        monkeypatch.setattr(app, "emit", emit)
        monkeypatch.setattr(app, "_report_step", lambda *args: None)
        task = asyncio.create_task(app.run())
        try:
            # Recovery runs on a thread; keep its completion flag thread-safe.
            async def recovery_finished():
                while not recovered.is_set():
                    await asyncio.sleep(.01)
            await asyncio.wait_for(recovery_finished(), timeout=5)
            async with app._step_lock:
                await app.resume()
            await asyncio.wait_for(emitted.wait(), timeout=5)
            assert not task.done()
            assert len(calls) == 2 and calls[1].image is not None
            completed = [m for m in messages if isinstance(m, ChunkGenerated)]
            assert len(completed) == 1 and completed[0].world_id == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    run(scenario())
