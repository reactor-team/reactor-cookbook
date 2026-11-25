# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weight check never downloads, and prepare_weights.py builds what it checks."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

worldplay2_model = importlib.import_module("worldplay2_model")
prepare_weights = importlib.import_module("prepare_weights")


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    for revision in (worldplay2_model.BASE_REVISION, worldplay2_model.FAST_REVISION):
        assert len(revision) == 40


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        worldplay2_model.check_weights(tmp_path)


def test_load_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("load must not download")

    fake = SimpleNamespace(hf_hub_download=refuse, snapshot_download=refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    model = worldplay2_model.WorldPlay2Model.__new__(worldplay2_model.WorldPlay2Model)
    with pytest.raises(FileNotFoundError, match="prepare_weights.py"):
        model.load(worldplay2_model.Settings(), tmp_path)


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(repo_id, filename, revision, local_dir, **_kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        calls.append((repo_id, revision))

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download)
    )
    out = tmp_path / "weights"
    prepare_weights.main(["--output", str(out)])
    worldplay2_model.check_weights(out)
    assert {repo for repo, _ in calls} == {
        worldplay2_model.BASE_REPO,
        worldplay2_model.FAST_REPO,
    }
    assert all(len(revision) == 40 for _, revision in calls)
