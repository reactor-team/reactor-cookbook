# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weight check never downloads, and prepare_weights.py builds what it checks."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

dreamx_assets = importlib.import_module("dreamx_assets")
prepare_weights = importlib.import_module("prepare_weights")


def _config(root: Path):
    return dreamx_assets.read_config(MODEL_DIR / "dreamx_world.yaml", root)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert "AMAP-ML/DreamX-World.git" in " ".join(manifest["build"]["run"])


def test_config_reads_layout_from_weights_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert (
        config.dreamx.path == tmp_path / "checkpoints/DreamX-World-5B/model.safetensors"
    )
    assert config.wan.path == tmp_path / "checkpoints/Wan2.2-TI2V-5B"
    assert config.source_path == Path("/opt/DreamX-World")


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        dreamx_assets.check_weights(_config(tmp_path))


def test_check_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the check must not download")

    fake = SimpleNamespace(hf_hub_download=refuse, snapshot_download=refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    config = replace(_config(tmp_path), source_path=tmp_path / "no-source")
    with pytest.raises(FileNotFoundError, match="rebuild the image"):
        dreamx_assets.check_runtime_assets(config)
    with pytest.raises(FileNotFoundError):
        dreamx_assets.check_weights(config)


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id, filename, revision, local_dir, **_kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        calls.append((repo_id, revision, filename))

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download)
    )
    out = tmp_path / "weights"
    prepare_weights.main(["--output", str(out)])
    config = _config(out)
    dreamx_assets.check_weights(config)
    assert {c[0] for c in calls} == {config.dreamx.repo_id, config.wan.repo_id}
    assert all(len(c[1]) == 40 for c in calls)
