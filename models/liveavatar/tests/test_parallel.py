"""CPU tests for bounded native continuation and Runtime process ownership."""

import ast
import queue
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from liveavatar_parallel import ParallelBackend


def backend():
    value = object.__new__(ParallelBackend)
    value.rank = value.output_rank = 2
    value._dist = Mock()
    value._dist.get_world_size.return_value = 3

    def gather(statuses, status, **kwargs):
        statuses[:] = [("wait", None), ("wait", None), status]

    value._dist.all_gather_object.side_effect = gather
    value._group = object()
    value._results = queue.Queue(maxsize=1)
    value._ack = queue.Queue(maxsize=1)
    value._thread = Mock()
    value._thread.is_alive.return_value = False
    value.active = True
    value.pending_ack = False
    value.audio = np.zeros(48000 * 4, np.float32)
    value.offset = 0
    return value


def test_one_clip_demand_and_audio_alignment():
    value = backend()
    value._results.put(("chunk", np.zeros((45, 2, 2, 3), np.uint8)))
    video, audio = value.next()
    assert len(video) == 45 and audio.shape == (1, 86400)
    assert value._ack.empty() and value.pending_ack
    value._results.put(("end", None))
    assert value.next() is None
    assert value._ack.get_nowait() is True
    assert not value.active
    value._thread.join.assert_called_once()


def test_active_cancel_belongs_to_runtime():
    value = backend()
    with pytest.raises(RuntimeError, match="Runtime"):
        value.close()
    value.active = False
    value.close()
    value.close()
    assert value.audio.size == 0 and value.offset == 0


def test_remote_rank_receives_same_cpu_video():
    value = backend()
    value.rank = 0

    def gather(statuses, status, **kwargs):
        statuses[:] = [status, ("wait", None), ("chunk", None)]

    value._dist.all_gather_object.side_effect = gather
    video = np.zeros((48, 2, 2, 3), np.uint8)

    def broadcast(payload, **kwargs):
        payload[0] = ("chunk", video)

    value._dist.broadcast_object_list.side_effect = broadcast
    frames, audio = value.next()
    assert frames is video and audio.shape == (1, 92160)
    assert not value.pending_ack


def test_native_error_is_not_a_successful_clip():
    value = backend()
    value._results.put(("error", "decode failed"))
    with pytest.raises(RuntimeError, match="decode failed"):
        value.next()
    assert value.offset == 0


def test_failure_on_non_output_rank_propagates_without_waiting_for_video():
    value = backend()

    def gather(statuses, status, **kwargs):
        statuses[:] = [("error", "DiT failed"), ("wait", None), status]

    value._dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError, match="DiT failed"):
        value.next()
    value._dist.broadcast_object_list.assert_not_called()


def test_model_never_creates_processes_or_temp_media():
    root = Path(__file__).parents[1]
    for name in (
        "liveavatar_model.py",
        "liveavatar_parallel.py",
        "liveavatar_parallel_worker.py",
        "liveavatar_streaming.py",
    ):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    item.name not in {"multiprocessing", "tempfile", "subprocess"}
                    for item in node.names
                )
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in {
                    "np.save",
                    "np.load",
                    "dist.init_process_group",
                }


def test_application_owns_runner_and_restart(monkeypatch):
    import liveavatar_pipeline as application
    from liveavatar_model import LiveAvatarSettings
    from liveavatar_types import LiveAvatarState

    runner = Mock()
    monkeypatch.setattr(application, "DistributedRunner", runner)
    value = application.LiveAvatar()
    value._settings = LiveAvatarSettings(
        Path("/source"), Path("/base"), Path("/lora"), Path("/work")
    )
    value.state = LiveAvatarState()
    value._start_engine()
    assert runner.call_args.kwargs["world_size"] == 3
    runner.return_value.start.assert_called_once()
    value._release_engine()
    value._release_engine()
    runner.return_value.shutdown.assert_called_once()
    assert value._engine is None
    value._start_engine()
    assert runner.call_count == 2


class CPUWorker:
    """Exercise the real Runtime IPC without torch, weights or GPU access."""

    def load(self):
        self.count = 0

    def generate(self, input):
        if input < 0:
            raise ValueError("deliberate failure")
        self.count += 1
        return {
            "rank": self.rank,
            "count": self.count,
            "frames": np.zeros((2, 2, 2, 3), np.uint8),
        }

    def reset(self):
        self.count = 0


@pytest.mark.parametrize("world_size", [1, 3])
def test_runtime_process_start_failure_reset_and_shutdown(world_size):
    from reactor_runtime.distributed import DistributedRunner

    runner = DistributedRunner(
        CPUWorker,
        world_size=world_size,
        init_process_group=False,
        call_timeout=10,
        start_timeout=30,
    )
    try:
        runner.start()
        assert runner.generate(1)["count"] == 1
        with pytest.raises(ValueError, match="deliberate"):
            runner.generate(-1)
        assert runner.generate(1)["count"] == 2
        runner.reset()
        result = runner.generate(1)
        assert result["rank"] == 0 and result["count"] == 1
        assert result["frames"].dtype == np.uint8
    finally:
        runner.shutdown()
        runner.shutdown()
    assert not runner.healthy


def test_native_continuation_thread_pauses_at_each_delivered_clip(
    tmp_path, monkeypatch
):
    import soundfile as sf
    import liveavatar_parallel_worker

    value = backend()
    value.active = False
    value._model = object()
    value._plan = {}
    delivered = []

    def generate(model, plan, rank, job, deliver):
        for count in (45, 48):
            delivered.append(count)
            deliver(np.zeros((count, 2, 2, 3), np.uint8))

    monkeypatch.setattr(liveavatar_parallel_worker, "generate_take", generate)
    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.zeros(16000 * 4, np.float32), 16000)
    value.start(
        image=tmp_path / "avatar.png",
        audio=audio_path,
        pose=None,
        prompt="",
        negative_prompt="",
        seed=420,
        max_chunks=2,
    )
    assert len(value.next()[0]) == 45
    assert delivered == [45]
    assert len(value.next()[0]) == 48
    assert delivered == [45, 48]
    assert value.next() is None
    assert not value._thread.is_alive()
    value.close()
