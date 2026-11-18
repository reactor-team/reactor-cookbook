import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome

from apple_fastframes import FastFramesApp
from apple_fastframes_model import (
    LATENT_STEPS,
    FastFramesInput,
    FastFramesModel,
    FastFramesResult,
    RolloutStateError,
)
from apple_fastframes_types import FastFramesState


def app():
    service = FastFramesApp()
    service.state = FastFramesState()
    service.engine = Mock()
    service.send = AsyncMock()
    service.output = Mock()
    return service


def run(coroutine):
    return asyncio.run(coroutine)


def test_refused_start_and_step_do_not_call_model():
    service = app()
    with pytest.raises(ApplicationError):
        run(service.process_input())
    with pytest.raises(CommandError) as error:
        run(service.start_video(" ", 12))
    assert error.value.code == "invalid_prompt"
    service.engine.generate.assert_not_called()
    service.send.assert_not_awaited()


def test_anchor_acknowledgment_and_stale_final_result():
    service = app()
    old = run(service.start_video("Old scene", 91))
    first = run(service.process_input())
    assert first.seed == 91
    assert run(service.process_input()) == first
    result = FastFramesResult(
        world_id=old.world_id,
        chunk=1,
        prompt_revision=0,
        complete=False,
        frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
    )
    service.engine.generate.return_value = result
    assert service.generate(first) is result
    service.engine.generate.assert_called_once_with(first)
    assert (
        run(service.process_output(StepOutcome(result=result, elapsed=0.28)))
        is not None
    )
    assert service.state._last_chunk_seconds == 0.28
    assert run(service.process_input()).seed is None

    new = run(service.start_video("New scene", 92))
    service.send.reset_mock()
    stale = FastFramesResult(
        world_id=old.world_id,
        chunk=LATENT_STEPS,
        prompt_revision=0,
        complete=True,
        frames=np.zeros((4, 1, 1, 3), dtype=np.uint8),
    )
    assert run(service.process_output(StepOutcome(result=stale, elapsed=1))) is None
    service.send.assert_not_awaited()
    assert run(service.process_input()).world_id == new.world_id
    assert run(service.process_input()).seed == 92
    assert service.state._chunks == 0


def test_prompt_is_applied_on_resume_and_stop_invalidates_progress():
    service = app()
    started = run(service.start_video("Scene one", 0))
    run(service.set_paused(True))
    with pytest.raises(ApplicationError):
        run(service.process_input())
    run(service.update_prompt("Scene two"))
    pending = service.send.await_args.args[0]
    assert pending.phase == "paused" and pending.prompt_pending
    assert pending.world_id == started.world_id
    run(service.set_paused(False))
    step = run(service.process_input())
    assert step.prompt == "Scene two" and step.prompt_revision == 1
    result = FastFramesResult(
        world_id=step.world_id,
        chunk=1,
        prompt_revision=1,
        complete=False,
        frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
    )
    run(service.process_output(StepOutcome(result=result, elapsed=0.4)))
    assert not service.send.await_args.args[0].prompt_pending
    stopped = run(service.stop_video())
    assert stopped.world_id != started.world_id
    assert service.send.await_args.args[0].output_frames == 0
    service.engine.reset.assert_called_once()
    with pytest.raises(CommandError) as error:
        run(service.update_prompt("Not running"))
    assert error.value.code == "no_active_clip"


def test_model_failure_propagates_and_session_teardown_resets():
    service = app()
    run(service.start_video("Scene", 0))
    error = RolloutStateError("Sample failure")
    service.engine.generate.side_effect = error
    with pytest.raises(RolloutStateError) as raised:
        service.generate(run(service.process_input()))
    with pytest.raises(RolloutStateError) as received:
        run(service.process_output(StepOutcome(error=raised.value, elapsed=0.01)))
    assert received.value is error and service.state._chunks == 0
    service.on_session_ended()
    service.engine.reset.assert_called_once()
    service.engine.healthy = False
    service.on_session_ended()
    service.engine.reset.assert_called_once()


def test_one_backend_rollout_with_two_prompt_updates_and_native_horizon():
    model = FastFramesModel()
    model._backend = Mock()
    model._backend.step.side_effect = lambda i: np.zeros(
        (1 if i == 0 else 4, 1, 1, 3), dtype=np.uint8
    )
    results = []
    for index in range(LATENT_STEPS):
        revision = int(index >= 3) + int(index >= 13)
        results.append(
            model.generate(
                FastFramesInput(
                    world_id=9,
                    prompt=f"Scene {revision}",
                    prompt_revision=revision,
                    seed=7 if index == 0 else None,
                )
            )
        )
    assert [r.chunk for r in results] == list(range(1, LATENT_STEPS + 1))
    assert sum(len(r.frames) for r in results) == 81
    assert [r.complete for r in results] == [False] * 20 + [True]
    assert [c.args for c in model._backend.set_prompt.call_args_list] == [
        ("Scene 0",),
        ("Scene 1",),
        ("Scene 2",),
    ]
    model._backend.start.assert_called_once_with(7)
    with pytest.raises(RolloutStateError, match="complete"):
        model.generate(
            FastFramesInput(world_id=9, prompt="Scene 2", prompt_revision=2, seed=None)
        )
    model.reset()
    with pytest.raises(RolloutStateError, match="seed"):
        model.generate(
            FastFramesInput(world_id=9, prompt="Scene 2", prompt_revision=2, seed=None)
        )
    result = model.generate(
        FastFramesInput(world_id=1, prompt="Fresh session", prompt_revision=0, seed=0)
    )
    assert result.chunk == 1
    model._backend.start.assert_called_with(0)


def test_join_only_replies_to_joining_viewer():
    service = app()
    run(service.start_video("Scene", 0))
    run(service.set_paused(True))
    service.send.reset_mock()
    viewer = Mock(send=AsyncMock())
    run(service.on_connected(viewer))
    assert viewer.send.await_args.args[0].phase == "paused"
    service.send.assert_not_awaited()


def test_prepared_snapshot_reuses_relative_links(tmp_path, monkeypatch):
    import huggingface_hub

    from apple_fastframes_assets import snapshot

    root = tmp_path / "old" / "models--owner--weights"
    blob = root / "blobs" / "data"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"test")
    model = root / "snapshots" / "revision"
    model.mkdir(parents=True)
    (model / "weight.safetensors").symlink_to("../../blobs/data")
    new = tmp_path / "relocated"
    (tmp_path / "old").rename(new)
    forbid_download = Mock(
        side_effect=AssertionError("Prepared assets must be read-only")
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbid_download)
    pinned = snapshot("owner/weights", "revision", ("weight.safetensors",), new)
    assert (pinned / "weight.safetensors").read_bytes() == b"test"
    forbid_download.assert_not_called()


@pytest.mark.parametrize("variant", [None, "ff-s", "ff-l"])
def test_variant_selects_only_its_inference_assets(tmp_path, monkeypatch, variant):
    import apple_fastframes_assets as assets

    config = tmp_path / "apple_fastframes.yaml"
    config.write_text(
        "source_path: upstream\n" + (f"variant: {variant}\n" if variant else "")
    )
    monkeypatch.setattr(
        assets.subprocess,
        "check_output",
        lambda *args, **kwargs: assets.SOURCE_REVISION,
    )
    monkeypatch.setattr(assets.subprocess, "run", Mock())
    fetch = Mock(side_effect=lambda repo, revision, files, weights: weights / repo)
    monkeypatch.setattr(assets, "snapshot", fetch)
    prepared = assets.prepare(config, tmp_path)
    assert prepared.variant == (variant or "ff-s")
    repos = [call.args[0] for call in fetch.call_args_list]
    if variant == "ff-l":
        assert repos == ["apple/FastFrames-L", "Wan-AI/Wan2.1-T2V-14B"]
        assert prepared.generator == prepared.apple / "fastframes-l/dit.safetensors"
    else:
        assert repos == ["apple/FastFrames-S", assets.CF_REPO, "Wan-AI/Wan2.1-T2V-1.3B"]
        assert prepared.generator == tmp_path / assets.CF_REPO / assets.CF_GENERATOR
