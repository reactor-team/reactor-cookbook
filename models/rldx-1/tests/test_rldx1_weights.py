# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import prepare_weights
import pytest
import yaml
from rldx1_assets import (
    BASE_MODEL_FILES,
    CHECKPOINT_FILES,
    PREPARE_COMMAND,
    check_weights,
    resolve_weights,
)
from rldx1_model import RLDXModel

MODEL_DIR = Path(__file__).parents[1]
CONFIG = yaml.safe_load((MODEL_DIR / "config.yml").read_text())
PINS = {
    (CONFIG[key]["repo_id"], CONFIG[key]["revision"])
    for key in ("checkpoint", "base_model")
}


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace huggingface_hub with a fake whose hf_hub_download writes the file."""
    calls: list[dict] = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def _stub_modules(monkeypatch: pytest.MonkeyPatch, modules: dict[str, object]) -> None:
    """Serve stub modules in place of the GPU stack, with the hub blocked."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    for name in ("rldx", "rldx.data", "rldx.policy"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def test_manifest_uses_the_prepared_folder_with_the_hub_off() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    dockerfile = (MODEL_DIR / "Dockerfile").read_text()
    assert re.search(r"^ENV HF_HUB_OFFLINE=1$", dockerfile, re.MULTILINE)
    assert "weights/" in (MODEL_DIR / ".dockerignore").read_text().split()
    readme = (MODEL_DIR / "README.md").read_text()
    assert PREPARE_COMMAND in readme
    assert "reactor deploy production --weights ./weights" in readme
    for repo_id, revision in PINS:
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert repo_id in readme and revision in readme


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    assert {(call["repo_id"], call["revision"]) for call in calls} == PINS
    weights = resolve_weights(CONFIG, tmp_path)
    for source in (weights.checkpoint, weights.base_model):
        asked = [c["filename"] for c in calls if c["repo_id"] == source.repo_id]
        assert asked == list(source.files)
        assert {
            Path(c["local_dir"]) for c in calls if c["repo_id"] == source.repo_id
        } == {source.path}
    assert weights.checkpoint.path == tmp_path / "RLDX-1-FT-ROBOCASA"
    assert weights.base_model.path == tmp_path / "RLDX-1-VLM"
    check_weights(weights)
    assert not [path for path in tmp_path.rglob("*") if path.is_symlink()]


def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match="RLDX-1-FT-ROBOCASA") as raised:
        RLDXModel().load(CONFIG, tmp_path)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_one_missing_shard_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    shard = tmp_path / "RLDX-1-FT-ROBOCASA" / "model-00002-of-00003.safetensors"
    shard.unlink()
    before = sorted(tmp_path.rglob("*"))
    with pytest.raises(FileNotFoundError, match=re.escape(str(shard))) as raised:
        RLDXModel().load(CONFIG, tmp_path)
    assert PREPARE_COMMAND in str(raised.value)
    assert sorted(tmp_path.rglob("*")) == before


def test_load_hands_the_policy_both_local_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    received: dict = {}

    class Stop(Exception):
        pass

    def policy(**kwargs):
        received.update(kwargs)
        raise Stop

    _stub_modules(
        monkeypatch,
        {
            "rldx.data.embodiment_tags": {
                "EmbodimentTag": SimpleNamespace(GENERAL_EMBODIMENT="general")
            },
            "rldx.policy.rldx_policy": {
                "RLDXPolicy": policy,
                "RLDXSimPolicyWrapper": None,
            },
        },
    )
    with pytest.raises(Stop):
        RLDXModel().load(CONFIG, tmp_path)
    assert received["model_path"] == str(tmp_path / "RLDX-1-FT-ROBOCASA")
    assert received["base_model_path"] == str(tmp_path / "RLDX-1-VLM")


def _import_policy_loader(
    monkeypatch: pytest.MonkeyPatch, calls: list
) -> types.ModuleType:
    """Import the vendored policy loader with fake transformers and the hub blocked."""

    def recorder(name, result):
        def from_pretrained(path, **kwargs):
            calls.append((name, Path(path), kwargs))
            return result

        return SimpleNamespace(from_pretrained=from_pretrained)

    model_config = SimpleNamespace(physics_keys=[], use_memory=False, action_horizon=16)
    model = SimpleNamespace(config=model_config, eval=lambda: None)
    language = SimpleNamespace(modality_keys=["annotation.task"], delta_indices=[0])
    processor = SimpleNamespace(
        physics_keys=[],
        collator=object(),
        eval=lambda: None,
        get_modality_configs=lambda: {"general": {"language": language}},
    )
    _stub_modules(
        monkeypatch,
        {
            "torch": {"bfloat16": "bfloat16"},
            "transformers": {
                "AutoConfig": recorder("AutoConfig", SimpleNamespace(use_memory=False)),
                "AutoModel": recorder("AutoModel", model),
                "AutoProcessor": recorder("AutoProcessor", processor),
            },
            "rldx.data.embodiment_tags": {"EmbodimentTag": object},
            "rldx.data.interfaces": {"BaseProcessor": object},
        },
    )
    path = MODEL_DIR / "rldx" / "policy" / "policy_loader.py"
    spec = importlib.util.spec_from_file_location("rldx_policy_loader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_policy_loader_reads_local_folders_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "RLDX-1-FT-ROBOCASA"
    base = tmp_path / "RLDX-1-VLM"
    (checkpoint / "processor").mkdir(parents=True)
    base.mkdir()
    calls: list = []
    loader = _import_policy_loader(monkeypatch, calls)
    assert not hasattr(loader, "snapshot_download")

    result = loader.PolicyLoader.load(
        embodiment_tag=SimpleNamespace(value="general"),
        model_path=str(checkpoint),
        device="cpu",
        base_model_path=str(base),
    )

    assert result.language_key == "annotation.task"
    assert [(name, path) for name, path, _ in calls] == [
        ("AutoConfig", checkpoint),
        ("AutoModel", checkpoint),
        ("AutoProcessor", checkpoint / "processor"),
    ]
    assert calls[1][2]["model_name"] == str(base)
    assert calls[2][2] == {"model_name": str(base)}


def test_policy_loader_refuses_a_missing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    loader = _import_policy_loader(monkeypatch, calls)
    missing = tmp_path / "RLDX-1-FT-ROBOCASA"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        loader.PolicyLoader.load(
            embodiment_tag=SimpleNamespace(value="general"),
            model_path=str(missing),
            device="cpu",
        )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_processor_accepts_the_local_base_model_override() -> None:
    source = (MODEL_DIR / "rldx" / "model" / "core" / "processing_rldx.py").read_text()
    overrides = next(
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "override_keys" for t in node.targets)
    )
    assert "model_name" in ast.literal_eval(overrides)


def test_file_lists_cover_what_the_loaders_read() -> None:
    assert {"config.json", "processor/processor_config.json"} <= set(CHECKPOINT_FILES)
    assert len([f for f in CHECKPOINT_FILES if f.endswith(".safetensors")]) == 3
    assert {"config.json", "tokenizer_config.json", "vocab.json", "merges.txt"} <= set(
        BASE_MODEL_FILES
    )
    assert not [f for f in BASE_MODEL_FILES if f.endswith(".safetensors")]
