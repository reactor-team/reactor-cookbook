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

evoke_config = importlib.import_module("evoke_config")
prepare_weights = importlib.import_module("prepare_weights")


def _config(root: Path):
    return evoke_config.read_config(MODEL_DIR / "evoke.yaml", root)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    run = " ".join(manifest["build"]["run"])
    assert "AlayaLab/Evoke.git" in run and "prepare_worker.sh" in run


def test_layout_follows_weights_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.base_model == tmp_path / "evoke-base"
    assert config.vigeo_path == tmp_path / "ViGeo1.1"
    assert config.source_path == Path("/opt/Evoke")


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        evoke_config.check_weights(_config(tmp_path))


def test_check_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the check must not download")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=refuse)
    )
    with pytest.raises(FileNotFoundError, match="rebuild the image"):
        evoke_config.check_source(_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        evoke_config.check_weights(_config(tmp_path))


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns, **_kwargs):
        calls.append((repo_id, revision))
        files = {
            "AlayaLab/Evoke": evoke_config._EVOKE_REQUIRED,
            "pkqbajng/ViGeo": ("vigeo.pt",),
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
    evoke_config.check_weights(_config(out))
    assert [c[0] for c in calls] == ["AlayaLab/Evoke", "pkqbajng/ViGeo"]
    assert all(len(c[1]) == 40 for c in calls)
