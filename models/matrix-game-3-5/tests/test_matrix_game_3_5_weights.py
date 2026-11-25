# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weight check never downloads, and prepare_weights.py builds what it checks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

config_module = importlib.import_module("matrix_game_3_5_config")
prepare_weights = importlib.import_module("prepare_weights")


def _config(root: Path):
    return config_module.read_config(MODEL_DIR / "matrix_game_3_5.yaml", root)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    run = " ".join(manifest["build"]["run"])
    assert "Matrix-Game-3.5.git" in run and "prepare_worker.sh" in run


def test_layout_follows_weights_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.wan.path == tmp_path / "checkpoints/Wan2.2-TI2V-5B"
    assert config.tokenizer_dir == config.wan.path / "google/umt5-xxl"
    assert config.checkpoint.path.parents[1] == tmp_path / "checkpoints"
    assert config.source_path == Path("/opt/Matrix-Game-3.5")


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        config_module.check_weights(_config(tmp_path))


def test_check_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the check must not download")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=refuse)
    )
    with pytest.raises(FileNotFoundError, match="rebuild the image"):
        config_module.check_source(_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        config_module.check_weights(_config(tmp_path))


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_snapshot(repo_id, revision, local_dir, **_kwargs):
        calls.append((repo_id, revision))
        files = {
            "Wan-AI/Wan2.2-TI2V-5B": config_module.WAN_REQUIRED_FILES,
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1": config_module.DEPTH_REQUIRED_FILES,
            "RiemannDynamics/Matrix-Game-3.5-Distilled": ("first-person.safetensors",),
        }[repo_id]
        for name in files:
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot)
    )
    out = tmp_path / "weights"
    prepare_weights.main(["--output", str(out)])
    config_module.check_weights(_config(out))
    assert len(calls) == 3
    assert all(len(c[1]) == 40 for c in calls)
