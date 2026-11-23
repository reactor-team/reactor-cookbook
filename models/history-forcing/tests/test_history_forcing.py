"""Exercise the real application/model contracts without CUDA or checkpoints."""

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.interface.model.contract import ContractError, ModelContract

from history_forcing import HistoryForcing
from history_forcing_model import (
    Anchor,
    HistoryForcingInput,
    HistoryForcingModel,
    HistoryForcingResult,
    MissingAnchorError,
)
from history_forcing_types import HistoryForcingState


def test_model_failure_does_not_increment_and_new_world_requires_anchor():
    model = HistoryForcingModel()
    model._backend = Mock()
    model._backend.step.return_value = result().frames
    with pytest.raises(MissingAnchorError, match="seed"):
        model.generate(HistoryForcingInput(world_id=1, prompt="garden", anchor=None))
    request = HistoryForcingInput(world_id=1, prompt="garden", anchor=Anchor(seed=42))
    assert model.generate(request).chunk_index == 1
    model._backend.begin.assert_called_once_with(42)
    request = HistoryForcingInput(world_id=1, prompt="rain", anchor=None)
    assert model.generate(request).chunk_index == 2
    model._backend.begin.assert_called_once()
    model._backend.step.side_effect = RuntimeError("native failed")
    with pytest.raises(RuntimeError, match="native failed"):
        model.generate(request)
    assert model._chunk_index == 2
    model.reset()
    assert model._world_id is None and model._chunk_index == 0


def test_text_change_invalidates_all_cross_attention_without_resetting_history():
    from history_forcing_backend import NativeBackend, Run

    backend = NativeBackend.__new__(NativeBackend)
    encoder = Mock(side_effect=lambda **kw: {"text": kw["text_prompts"][0]})
    visual, decoder, noise = object(), object(), object()
    backend.pipe = SimpleNamespace(
        text_encoder=encoder,
        crossattn_cache=[{"is_init": True} for _ in range(30)],
        kv_cache1=visual,
        vae=decoder,
    )
    backend.run = Run(
        noise=noise,
        noise_offset=12,
        position=264,
        prompt="garden",
        conditioning={"text": "garden"},
    )
    backend._update_prompt("garden")
    encoder.assert_not_called()
    for text in ["rain", "sunshine"]:
        for c in backend.pipe.crossattn_cache:
            c["is_init"] = True
        backend._update_prompt(text)
        assert backend.run.conditioning == {"text": text}
        assert all(not c["is_init"] for c in backend.pipe.crossattn_cache)
        assert backend.pipe.kv_cache1 is visual and backend.pipe.vae is decoder
        assert (
            backend.run.noise is noise
            and backend.run.noise_offset == 12
            and backend.run.position == 264
        )
    assert encoder.call_count == 2


def test_prepared_assets_need_no_network_or_writes(tmp_path, monkeypatch):
    import json

    import huggingface_hub

    from history_forcing_assets import BASE_FILES, PROVENANCE, prepare_weights

    for name in (
        *BASE_FILES,
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
    ):
        p = tmp_path / "models/Wan2.1-T2V-1.3B" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"prepared asset")
    p = tmp_path / "models/History-Forcing/dmd/model.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"prepared checkpoint")
    (tmp_path / "history-forcing-assets.json").write_text(json.dumps(PROVENANCE))
    forbidden = Mock(side_effect=AssertionError("prepared read-only assets"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    prepare_weights(tmp_path)
    forbidden.assert_not_called()


def app():
    model = HistoryForcing()
    model.state = HistoryForcingState()
    model.engine = Mock()
    model.output = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    return model, messages


def result(world=1, index=1):
    return HistoryForcingResult(
        world_id=world, chunk_index=index, frames=np.zeros((2, 480, 832, 3), np.uint8)
    )


def test_refused_step_never_reaches_compute():
    model, _ = app()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.set_paused(True))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.engine.generate.assert_not_called()


def test_runtime_loop_routes_compute_failure_without_acknowledging_output():
    import threading

    from reactor_runtime.core.model import ClientConnected, SessionStarted
    from reactor_runtime.core.values import ConnId

    model, _ = app()
    failure = RuntimeError("failed native compute")
    model.engine.generate.side_effect = failure
    finished = threading.Event()
    failures, outcomes, media = [], [], []
    process_output = model.process_output

    async def record(outcome):
        outcomes.append(outcome)
        return await process_output(outcome)

    def failed(error):
        failures.append(error)
        finished.set()

    model.process_output = record
    model.bind_output(
        broadcast=lambda *args: None,
        addressed=lambda *args: None,
        media=lambda *args: media.append(args),
    )
    model.bind_failure(failed)
    model.start_thread()
    try:
        model.post_reactor_event(SessionStarted("history_forcing-cpu-test"))
        model.post_reactor_event(ClientConnected(ConnId(1), total=1))
        command = ModelContract.of(HistoryForcing).validate(
            "start", {"prompt": "A garden", "seed": 42}
        )
        model.submit_command(command, ConnId(1), None)
        assert finished.wait(timeout=5), "Runtime did not report the failed step"
        assert failures == [failure] and not media
        assert len(outcomes) == 1 and outcomes[0].error is failure
        assert model.state._completed == 0 and model.state._applied_world_id is None
        model.engine.generate.assert_called_once()
    finally:
        model.stop()


def test_forward_reads_only_input():
    model, _ = app()
    value = HistoryForcingInput(world_id=7, prompt="old", anchor=None)
    model.state = None
    model.generate(value)
    model.engine.generate.assert_called_once_with(value)


def test_anchor_repeats_until_acknowledged_then_is_omitted():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    first = asyncio.run(model.process_input())
    assert first.anchor == Anchor(seed=42)
    assert asyncio.run(model.process_input()).anchor is first.anchor
    out = asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=0.3)))
    assert out.main_video.shape == (2, 480, 832, 3)
    assert asyncio.run(model.process_input()).anchor is None
    assert messages[-1].completed_chunks == 1 and messages[-1].last_chunk_seconds == 0.3


@pytest.mark.parametrize("transition", ["start", "reset", "stop"])
def test_stale_output_cannot_change_replaced_or_stopped_world(transition):
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    if transition == "start":
        asyncio.run(model.start("A beach", 7))
    elif transition == "reset":
        asyncio.run(model.reset(None))
    else:
        asyncio.run(model.stop_video())
    before = model._status()
    messages.clear()
    assert (
        asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=1)))
        is None
    )
    assert model._status() == before and not messages


def test_two_prompt_updates_preserve_history_and_pause():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.process_output(StepOutcome(result=result(index=3), elapsed=0.3)))
    model.output.flush.reset_mock()
    for text in ("A cat arrives", "The cat walks away"):
        asyncio.run(model.set_paused(True))
        state = asyncio.run(model.set_prompt(text))
        assert state.world_id == 1 and state.completed_chunks == 3 and state.paused
        with pytest.raises(ApplicationError):
            asyncio.run(model.process_input())
        asyncio.run(model.set_paused(False))
        request = asyncio.run(model.process_input())
        assert request.prompt == text and request.anchor is None
    model.output.flush.assert_not_called()
    model.engine.reset.assert_not_called()


@pytest.mark.parametrize(
    "command,args,code",
    [
        ("start", (" ", 0), "invalid_prompt"),
        ("set_prompt", (" ",), "invalid_prompt"),
        ("set_paused", (True,), "no_video"),
        ("reset", (), "no_video"),
    ],
)
def test_refusal_changes_nothing_and_broadcasts_nothing(command, args, code):
    model, messages = app()
    before = model._status()
    with pytest.raises(CommandError) as refused:
        asyncio.run(getattr(model, command)(*args))
    assert refused.value.code == code
    assert model._status() == before and not messages
    model.engine.generate.assert_not_called()


def test_reset_keeps_current_seed_unless_explicitly_changed():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    for seed, expected in ((None, 42), (77, 77), (None, 77)):
        asyncio.run(model.reset(seed))
        assert asyncio.run(model.process_input()).anchor.seed == expected
        assert model.state._completed == 0


def test_stop_flushes_resets_and_preserves_staged_prompt():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    state = asyncio.run(model.stop_video())
    assert not state.active and state.prompt == "A garden" and state.output_frames == 0
    model.engine.reset.assert_called_once_with()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert asyncio.run(model.start("", 0)).active


def test_join_is_private_and_does_not_reset_shared_world():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    messages.clear()
    sent = []

    async def send(message):
        sent.append(message)

    client = Mock()
    client.send = send
    asyncio.run(model.on_connected(client))
    assert sent == [model._status()] and not messages
    model.engine.reset.assert_not_called()
    assert "on_disconnected" not in HistoryForcing.__dict__


def test_failure_never_acknowledges_or_increments():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    failure = RuntimeError("native failed")
    with pytest.raises(RuntimeError, match="native failed"):
        asyncio.run(model.process_output(StepOutcome(error=failure, elapsed=1)))
    assert model.state._completed == 0 and model.state._applied_world_id is None


def test_end_only_resets_healthy_runner_and_new_session_is_empty():
    model, _ = app()
    model.engine.healthy = False
    model.on_session_ended()
    model.engine.reset.assert_not_called()
    model.engine.healthy = True
    model.on_session_ended()
    model.engine.reset.assert_called_once_with()
    model.state = HistoryForcingState()
    assert model._status().seed is None and not model._status().active


def test_input_and_result_are_frozen():
    value = HistoryForcingInput(world_id=1, prompt="garden", anchor=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.prompt = "new"
    output = result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        output.chunk_index = 0


def test_schema_exposes_only_real_commands_without_rollout_cap():
    from reactor_runtime import ReactorApp

    contract = ModelContract.of(HistoryForcing)
    assert set(contract.commands) == {
        "start",
        "set_prompt",
        "set_paused",
        "reset",
        "stop",
        "get_state",
    }
    assert not HistoryForcing.__dict__.get("fps")
    assert HistoryForcing.stop is ReactorApp.stop


def test_model_imports_without_runtime_or_torch():
    import subprocess
    import sys

    directory = Path(__file__).resolve().parents[1]
    check = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','reactor_runtime','naive_rt'):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import history_forcing_model
"""
    subprocess.run(
        [sys.executable, "-c", check], cwd=directory, check=True, capture_output=True
    )


def test_public_prompt_commands_keep_moderation_and_schema_limits():
    contract = ModelContract.of(HistoryForcing)
    for command in ("start", "set_prompt"):
        field = contract.commands[command].command.__command_fields__["prompt"].info
        assert field.moderate
    for seed in (-1, 2**32):
        with pytest.raises(ContractError):
            contract.validate("start", {"prompt": "A garden", "seed": seed})
    with pytest.raises(ContractError):
        contract.validate("set_prompt", {"prompt": "a" * 4097})


def test_downloaded_cache_survives_relocation(tmp_path, monkeypatch):
    import os

    import huggingface_hub

    from history_forcing_assets import BASE_FILES, prepare_weights

    weights = tmp_path / "original"
    base = weights / "hub/base"
    for name in (
        *BASE_FILES,
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
    ):
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"asset")
    checkpoint = weights / "hub/checkpoint"
    (checkpoint / "dmd").mkdir(parents=True)
    (checkpoint / "dmd/model.pt").write_bytes(b"checkpoint")
    download = Mock(side_effect=[str(base), str(checkpoint)])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepare_weights(weights)
    assert download.call_count == 2
    for link in (
        weights / "models/Wan2.1-T2V-1.3B",
        weights / "models/History-Forcing",
    ):
        assert not Path(os.readlink(link)).is_absolute()
    moved = tmp_path / "container-mount"
    weights.rename(moved)
    download.side_effect = AssertionError("A moved prepared cache must remain usable")
    prepare_weights(moved)
    assert download.call_count == 2


def test_backend_reset_drops_all_run_and_native_cache_state():
    from history_forcing_backend import NativeBackend, Run

    backend = NativeBackend.__new__(NativeBackend)
    clear = Mock()
    backend.pipe = SimpleNamespace(
        kv_cache1=object(),
        crossattn_cache=object(),
        vae=SimpleNamespace(model=SimpleNamespace(clear_cache=clear)),
    )
    backend.run = Run(noise=object(), position=270, prompt="garden")
    backend.reset()
    assert backend.run is None
    assert backend.pipe.kv_cache1 is None and backend.pipe.crossattn_cache is None
    clear.assert_called_once_with()


def test_read_only_snapshot_keeps_paused_shared_world_unchanged():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.set_paused(True))
    messages.clear()
    before = model._status()
    assert asyncio.run(model.get_state()) == before
    assert model._status() == before and not messages
    model.engine.reset.assert_not_called()
    model.engine.generate.assert_not_called()
