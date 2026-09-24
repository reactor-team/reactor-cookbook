"""Test the Cosmos3-Policy-DROID contract, gates, and model-half bookkeeping without a GPU."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

import cosmos3_policy_droid_assets as assets
import cosmos3_policy_droid_model
from cosmos3_policy_droid import Cosmos3PolicyDroid, parse_executed_step, parse_proprio
from cosmos3_policy_droid_assets import (
    DEFAULT_SOURCE,
    SOURCE_ENV,
    Repository,
    ensure_source_checkout,
    read_config,
    resolve_source_path,
    route_checkpoint_downloads,
)
from cosmos3_policy_droid_model import VIEWS, Cosmos3PolicyModel, PolicyInput, PolicyResult
from cosmos3_policy_droid_types import ActionPrediction, PolicyState

MODEL_DIR = Path(__file__).resolve().parent.parent
WRIST = np.full((360, 640, 3), 7, dtype=np.uint8)
EXTERIOR = np.full((180, 320, 3), 3, dtype=np.uint8)
PROPRIO = json.dumps({"joint_position": [[0.1] * 7], "gripper_position": [[0.5]]})


def _echo(step: int) -> str:
    return json.dumps({"step": step, "action": [[0.0] * 8]})


class FakeModel:
    """The model half by shape: records inputs and returns scripted chunks."""

    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []
        self.resets = 0

    def generate(self, input: PolicyInput) -> PolicyResult:
        self.inputs.append(input)
        return PolicyResult(actions=np.full((32, 8), len(self.inputs), np.float32), horizon=32, dof=8)

    def reset(self) -> None:
        self.resets += 1


class _Frame:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _Track:
    def __init__(self) -> None:
        self.pending: list[np.ndarray] = []

    def push(self, frame: np.ndarray) -> None:
        self.pending.append(frame)

    def try_read(self, n: int = 1) -> list[_Frame] | None:
        if len(self.pending) < n:
            return None
        frames, self.pending = self.pending[-n:], []
        return [_Frame(f) for f in frames]


class _Media:
    def __init__(self) -> None:
        self.wrist_view = _Track()
        self.exterior_view_1 = _Track()
        self.exterior_view_2 = _Track()

    def push_all(self) -> None:
        self.wrist_view.push(WRIST)
        self.exterior_view_1.push(EXTERIOR)
        self.exterior_view_2.push(EXTERIOR)


def _app() -> tuple[Any, FakeModel, _Media, list[Any]]:
    app = Cosmos3PolicyDroid()
    model = FakeModel()
    app._engine = model
    app.state = PolicyState()
    media = _Media()
    app.media = media
    messages: list[Any] = []

    async def record(message: Any) -> None:
        messages.append(message)

    app.send = record
    app.on_session_started()
    return app, model, media, messages


def _ready(app: Any, media: _Media) -> None:
    media.push_all()
    app.state.proprio_json = PROPRIO
    app.state.task_description = "put the banana in the bowl"


async def _step(app: Any) -> Any:
    input = await app.process_input()
    result = app.generate(input)
    return await app.process_output(StepOutcome(result=result, elapsed=0.2))


# -- the client contract ---------------------------------------------------------


def test_contract_is_the_fleet_models() -> None:
    contract = ModelContract.of(Cosmos3PolicyDroid)
    assert set(contract.commands) == {
        "reset",
        "set_task_description",
        "set_proprio_json",
        "set_executed_step_json",
    }
    assert list(contract.tracks) == list(VIEWS)
    assert set(contract.messages) == {"action_prediction"}


def test_state_starts_with_both_gates_closed() -> None:
    state = PolicyState()
    assert state._last_executed == -1
    assert state._predicted == -1
    assert state.task_description == ""


# -- process_input -----------------------------------------------------------------


def test_refuses_until_every_view_has_delivered_a_frame() -> None:
    app, model, media, _ = _app()
    app.state.proprio_json = PROPRIO
    media.wrist_view.push(WRIST)
    with pytest.raises(ApplicationError, match="every camera view"):
        asyncio.run(app.process_input())
    # The wrist frame is retained; the two exterior views arriving later complete the set.
    media.exterior_view_1.push(EXTERIOR)
    media.exterior_view_2.push(EXTERIOR)
    input = asyncio.run(app.process_input())
    assert input.wrist_view is WRIST
    assert model.inputs == []


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        json.dumps({"joint_position": [[0.1] * 6], "gripper_position": [[0.5]]}),
        json.dumps({"joint_position": [[0.1] * 7]}),
        json.dumps({"joint_position": [0.1] * 7, "gripper_position": [0.5]}),
        json.dumps({"joint_position": [[float("nan")] * 7], "gripper_position": [[0.5]]}),
    ],
)
def test_refuses_without_valid_proprio(raw: str) -> None:
    app, _, media, _ = _app()
    media.push_all()
    app.state.proprio_json = raw
    with pytest.raises(ApplicationError, match="proprio_json"):
        asyncio.run(app.process_input())


def test_first_prediction_needs_no_echo_and_carries_the_state() -> None:
    app, _, media, _ = _app()
    _ready(app, media)
    input = asyncio.run(app.process_input())
    assert input.task == "put the banana in the bowl"
    assert input.joint_position.shape == (1, 7)
    assert input.gripper_position.shape == (1, 1)
    assert input.joint_position.dtype == np.float32
    with pytest.raises(FrozenInstanceError):
        input.task = "mutated"


def test_second_prediction_waits_for_the_echo_to_advance() -> None:
    app, model, media, _ = _app()
    _ready(app, media)
    asyncio.run(_step(app))
    assert app.state._predicted == 0

    with pytest.raises(ApplicationError, match="executed_step_json"):
        asyncio.run(app.process_input())
    app.state.executed_step_json = _echo(-1)
    with pytest.raises(ApplicationError, match="executed_step_json"):
        asyncio.run(app.process_input())

    app.state.executed_step_json = _echo(0)
    asyncio.run(_step(app))
    assert app.state._predicted == 1
    assert app.state._last_executed == 0
    # The same echo again does not open the gate a second time.
    with pytest.raises(ApplicationError):
        asyncio.run(app.process_input())
    assert len(model.inputs) == 2


def test_frames_are_retained_between_steps() -> None:
    app, model, media, _ = _app()
    _ready(app, media)
    asyncio.run(_step(app))
    app.state.executed_step_json = _echo(0)
    fresh = np.full((360, 640, 3), 9, dtype=np.uint8)
    media.wrist_view.push(fresh)
    asyncio.run(_step(app))
    assert model.inputs[1].wrist_view is fresh
    assert model.inputs[1].exterior_view_1 is EXTERIOR


# -- generate and process_output -------------------------------------------------------


def test_generate_forwards_to_the_model_half() -> None:
    app, model, _, _ = _app()
    input = PolicyInput(
        WRIST, EXTERIOR, EXTERIOR, np.zeros((1, 7), np.float32), np.zeros((1, 1), np.float32), "t"
    )
    result = app.generate(input)
    assert model.inputs == [input]
    assert result.actions.shape == (32, 8)


def test_process_output_sends_one_chunk_per_step_and_no_media() -> None:
    app, _, media, messages = _app()
    _ready(app, media)
    assert asyncio.run(_step(app)) is None
    app.state.executed_step_json = _echo(0)
    asyncio.run(_step(app))
    assert [m.step for m in messages] == [0, 1]
    assert all(isinstance(m, ActionPrediction) for m in messages)
    assert np.asarray(messages[0].action).shape == (32, 8)
    assert messages[1].action[0][0] == 2.0


def test_model_failure_propagates_without_counting_a_prediction() -> None:
    app, _, _, messages = _app()
    with pytest.raises(RuntimeError, match="cuda"):
        asyncio.run(app.process_output(StepOutcome(error=RuntimeError("cuda"), elapsed=0.1)))
    assert app.state._predicted == -1
    assert messages == []


# -- commands and lifecycle -------------------------------------------------------------


def test_reset_reopens_the_gate_and_restarts_the_count() -> None:
    app, model, media, messages = _app()
    _ready(app, media)
    asyncio.run(_step(app))
    asyncio.run(app.reset())
    assert app.state._predicted == -1
    assert app.state._last_executed == -1
    assert app._frames == {}
    with pytest.raises(ApplicationError, match="every camera view"):
        asyncio.run(app.process_input())
    media.push_all()
    asyncio.run(_step(app))
    assert messages[-1].step == 0
    assert model.resets == 0  # the policy holds nothing to reset


def test_session_end_resets_the_model_half_and_drops_the_frames() -> None:
    app, model, media, _ = _app()
    _ready(app, media)
    asyncio.run(_step(app))
    app.on_session_ended()
    assert model.resets == 1
    assert app._frames == {}


# -- parsers ------------------------------------------------------------------------------


def test_parse_proprio_reads_row_lists() -> None:
    raw = json.dumps({"joint_position": [[0.0] * 7, [1.0] * 7], "gripper_position": [[0.0], [1.0]]})
    joints, gripper = parse_proprio(raw)  # type: ignore[misc]
    assert joints.shape == (2, 7)
    assert gripper.shape == (2, 1)


def test_parse_executed_step() -> None:
    assert parse_executed_step("") is None
    assert parse_executed_step("{}") is None
    assert parse_executed_step('{"step": "x"}') is None
    assert parse_executed_step(_echo(7)) == 7


# -- config ---------------------------------------------------------------------------------


def test_config_defaults_to_the_edge_checkpoint(tmp_path: Path) -> None:
    config = read_config(MODEL_DIR / "cosmos3_policy_droid.yaml")
    assert config.checkpoint == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert config.format_prompt_as_json is True
    assert config.guidance_interval == (960, 1001)
    assert (config.num_steps, config.guidance, config.action_chunk_size) == (4, 3.0, 32)

    nano = tmp_path / "nano.yaml"
    nano.write_text(
        "checkpoint: nvidia/Cosmos3-Nano-Policy-DROID\nformat_prompt_as_json: null\nguidance_interval: null\n"
    )
    config = read_config(nano)
    assert config.checkpoint == "nvidia/Cosmos3-Nano-Policy-DROID"
    assert config.format_prompt_as_json is None
    assert config.guidance_interval is None


def test_config_pins_the_upstream_source(tmp_path: Path) -> None:
    config = read_config(MODEL_DIR / "cosmos3_policy_droid.yaml")
    assert config.source == DEFAULT_SOURCE
    assert len(config.source.revision) == 40

    short = tmp_path / "short.yaml"
    short.write_text("source:\n  revision: cf5d68c\n")
    with pytest.raises(ValueError, match="40-hex"):
        read_config(short)


def test_source_path_resolves_under_the_weights_root_unless_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    assert resolve_source_path(DEFAULT_SOURCE, Path("/weights")) == Path("/weights/source/cosmos-framework")
    absolute = Repository(path=Path("/src/cf"), url=DEFAULT_SOURCE.url, revision=DEFAULT_SOURCE.revision)
    assert resolve_source_path(absolute, Path("/weights")) == Path("/src/cf")
    monkeypatch.setenv(SOURCE_ENV, "/elsewhere/cosmos-framework")
    assert resolve_source_path(DEFAULT_SOURCE, Path("/weights")) == Path("/elsewhere/cosmos-framework")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _local_upstream(tmp_path: Path) -> tuple[Repository, Path]:
    """A local Git repository standing in for github.com/NVIDIA/cosmos-framework."""
    upstream = tmp_path / "upstream"
    (upstream / "cosmos_framework").mkdir(parents=True)
    (upstream / "cosmos_framework" / "__init__.py").write_text("")
    _git("init", "-q", cwd=upstream)
    _git("add", ".", cwd=upstream)
    _git("commit", "-q", "-m", "pinned", cwd=upstream)
    pinned = _git("rev-parse", "HEAD", cwd=upstream)
    (upstream / "later.txt").write_text("after the pin")
    _git("add", ".", cwd=upstream)
    _git("commit", "-q", "-m", "later", cwd=upstream)
    return Repository(path=Path("source/cf"), url=str(upstream), revision=pinned), tmp_path / "weights"


def test_source_checkout_clones_the_pinned_revision_once(tmp_path: Path) -> None:
    source, weights = _local_upstream(tmp_path)
    path = resolve_source_path(source, weights)
    ensure_source_checkout(source, path)
    assert _git("rev-parse", "HEAD", cwd=path) == source.revision
    assert not (path / "later.txt").exists()  # detached at the pin, not at the branch head
    assert (path / "cosmos_framework" / "__init__.py").is_file()
    # A second call verifies and leaves the checkout alone.
    ensure_source_checkout(source, path)


def test_source_checkout_refuses_a_drifted_or_modified_checkout(tmp_path: Path) -> None:
    source, weights = _local_upstream(tmp_path)
    path = resolve_source_path(source, weights)
    ensure_source_checkout(source, path)

    (path / "cosmos_framework" / "__init__.py").write_text("# edited\n")
    with pytest.raises(RuntimeError, match="local changes"):
        ensure_source_checkout(source, path)
    _git("checkout", "--", ".", cwd=path)

    _git("checkout", "-q", "--detach", "HEAD~0", cwd=path)
    wrong = Repository(path=source.path, url=source.url, revision="0" * 40)
    with pytest.raises(RuntimeError, match="revision is"):
        ensure_source_checkout(wrong, path)

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="Git checkout"):
        ensure_source_checkout(source, plain)


def test_activate_source_puts_the_checkout_first_on_sys_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    assets.activate_source(tmp_path)
    assets.activate_source(tmp_path)
    assert sys.path[0] == str(tmp_path)
    assert sys.path.count(str(tmp_path)) == 1


# -- the model half, with the framework stubbed ----------------------------------------------


class _FakeService:
    """RobolabPolicyService by shape: records observations and returns a chunk."""

    instances: list[_FakeService] = []

    def __init__(self, args: Any) -> None:
        self.args = args
        self.cfg = types.SimpleNamespace(action_chunk_size=args.action_chunk_size, action_dim=8)
        self.observations: list[dict[str, Any]] = []
        self.shape = (32, 8)
        _FakeService.instances.append(self)

    def _build_setup_args(self, args: Any, overrides: Any) -> Any:
        return types.SimpleNamespace(guardrails=True, args=args, overrides=overrides)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(obs)
        return {"action": np.zeros(self.shape, np.float32)}


def _stub_framework(
    monkeypatch: pytest.MonkeyPatch, cache_calls: list[dict[str, Any]]
) -> list[tuple[Repository, Path]]:
    def server_args(**kwargs: Any) -> Any:
        return types.SimpleNamespace(**kwargs)

    server = types.ModuleType("cosmos_framework.scripts.action_policy_server_robolab")
    server.RobolabPolicyService = _FakeService  # type: ignore[attr-defined]
    server.RobolabServerArgs = server_args  # type: ignore[attr-defined]
    scripts = types.ModuleType("cosmos_framework.scripts")
    scripts.action_policy_server_robolab = server  # type: ignore[attr-defined]

    class _Hf:
        repository_type = types.SimpleNamespace(value="model")
        include: tuple[str, ...] = ()
        exclude: tuple[str, ...] = ()
        subdirectory = ""

        def __init__(self, repository: str, revision: str, filename: str = "") -> None:
            self.repository, self.revision, self.filename = repository, revision, filename

    class CheckpointDirHf(_Hf):
        def _download(self) -> str:
            raise AssertionError("upstream downloader must be routed")

    class CheckpointFileHf(_Hf):
        def _download(self) -> str:
            raise AssertionError("upstream downloader must be routed")

    db = types.ModuleType("cosmos_framework.utils.checkpoint_db")
    db.CheckpointDirHf = CheckpointDirHf  # type: ignore[attr-defined]
    db.CheckpointFileHf = CheckpointFileHf  # type: ignore[attr-defined]
    utils = types.ModuleType("cosmos_framework.utils")
    utils.checkpoint_db = db  # type: ignore[attr-defined]
    package = types.ModuleType("cosmos_framework")
    package.scripts = scripts  # type: ignore[attr-defined]
    package.utils = utils  # type: ignore[attr-defined]

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id: str, **kwargs: Any) -> str:
        cache_calls.append({"repo": repo_id, **kwargs})
        return f"/weights/{repo_id}"

    def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        cache_calls.append({"repo": repo_id, "filename": filename, **kwargs})
        return f"/weights/{repo_id}/{filename}"

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(set_device=lambda _i: None)  # type: ignore[attr-defined]

    for name, module in {
        "cosmos_framework": package,
        "cosmos_framework.scripts": scripts,
        "cosmos_framework.scripts.action_policy_server_robolab": server,
        "cosmos_framework.utils": utils,
        "cosmos_framework.utils.checkpoint_db": db,
        "huggingface_hub": hub,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    _FakeService.instances.clear()

    # The source checkout is the model half's first step; record it instead of cloning.
    checkouts: list[tuple[Repository, Path]] = []
    monkeypatch.setattr(
        cosmos3_policy_droid_model,
        "ensure_source_checkout",
        lambda source, path: checkouts.append((source, path)),
    )
    monkeypatch.setattr(cosmos3_policy_droid_model, "activate_source", lambda path: None)
    return checkouts


def test_model_load_clones_the_source_then_builds_the_edge_service_and_warms_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    checkouts = _stub_framework(monkeypatch, calls)
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    model = Cosmos3PolicyModel()
    model.load(MODEL_DIR / "cosmos3_policy_droid.yaml", Path("/weights"))

    assert checkouts == [(DEFAULT_SOURCE, Path("/weights/source/cosmos-framework"))]
    service = _FakeService.instances[-1]
    assert service.args.checkpoint_path == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert service.args.format_prompt_as_json is True
    assert service.args.guidance_interval == (960, 1001)
    assert (service.args.num_steps, service.args.guidance) == (4, 3.0)
    assert (model.horizon, model.dof) == (32, 8)
    assert len(service.observations) == 2  # warmup: first call compiles, second confirms
    assert service._build_setup_args(None, None).guardrails is False


def test_model_generate_maps_the_input_to_the_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_framework(monkeypatch, [])
    model = Cosmos3PolicyModel()
    model.load(None, Path("/weights"))
    service = _FakeService.instances[-1]
    joints = np.zeros((1, 7), np.float32)
    gripper = np.zeros((1, 1), np.float32)
    result = model.generate(PolicyInput(WRIST, EXTERIOR, EXTERIOR, joints, gripper, "stack the cups"))
    obs = service.observations[-1]
    assert obs["prompt"] == "stack the cups"
    assert obs["observation/wrist_image_left"] is WRIST
    assert obs["observation/exterior_image_1_left"] is EXTERIOR
    assert obs["observation/joint_position"] is joints
    assert result.actions.shape == (32, 8)
    assert result.actions.dtype == np.float32

    service.shape = (16, 8)
    with pytest.raises(RuntimeError, match="expected \\(32, 8\\)"):
        model.generate(PolicyInput(WRIST, EXTERIOR, EXTERIOR, joints, gripper, "t"))


def test_model_generate_before_load_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        Cosmos3PolicyModel().generate(
            PolicyInput(WRIST, EXTERIOR, EXTERIOR, np.zeros((1, 7)), np.zeros((1, 1)), "t")
        )


def test_checkpoint_downloads_route_through_huggingface_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _stub_framework(monkeypatch, calls)
    route_checkpoint_downloads(Path("/weights"))
    route_checkpoint_downloads(Path("/elsewhere"))  # idempotent: the first routing stays
    from cosmos_framework.utils import checkpoint_db as db

    assert db.CheckpointDirHf("nvidia/Cosmos3-Edge-Policy-DROID", "main")._download() == (
        "/weights/nvidia/Cosmos3-Edge-Policy-DROID"
    )
    assert (
        db.CheckpointFileHf("Wan-AI/Wan2.2-TI2V-5B", "main", "Wan2.2_VAE.pth")
        ._download()
        .endswith("Wan2.2_VAE.pth")
    )
    assert all(call["cache_dir"] == "/weights" for call in calls)
    assert calls[0]["revision"] == "main"


def test_disable_guardrails_is_idempotent() -> None:
    class Service:
        def _build_setup_args(self, args: Any, overrides: Any) -> Any:
            return types.SimpleNamespace(guardrails=True, args=args)

    cosmos3_policy_droid_model._disable_guardrails(Service)
    first = Service._build_setup_args
    cosmos3_policy_droid_model._disable_guardrails(Service)
    assert Service._build_setup_args is first
    assert Service()._build_setup_args("a", "b").guardrails is False
