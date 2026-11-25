# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weight check never downloads, and prepare_weights.py builds what it checks."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

streamrender_h3_assets = importlib.import_module("streamrender_h3_assets")
prepare_weights = importlib.import_module("prepare_weights")

ASSETS = {
    "assets": {
        "tokenizer": "tokenizer",
        "tae": "tae/taeh3_official.pth",
        "prompt": "free text, not a path",
        "backbone": streamrender_h3_assets.MERGED_BACKBONE,
    }
}


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert len(streamrender_h3_assets.REVISION) == 40


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        streamrender_h3_assets.check_weights(tmp_path / "assets.json")
    (tmp_path / "assets.json").write_text(json.dumps(ASSETS))
    with pytest.raises(FileNotFoundError, match="taeh3_official"):
        streamrender_h3_assets.check_weights(tmp_path / "assets.json")


def test_check_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the check must not download")

    fake = SimpleNamespace(hf_hub_download=refuse, snapshot_download=refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    with pytest.raises(FileNotFoundError):
        streamrender_h3_assets.check_weights(tmp_path / "assets.json")


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_snapshot(repo_id, revision, local_dir, **_kwargs):
        root = Path(local_dir)
        (root / "tokenizer").mkdir(parents=True)
        (root / "tae").mkdir()
        (root / "tae/taeh3_official.pth").write_bytes(b"x")
        (root / "assets.json").write_text(json.dumps(ASSETS))
        calls.append((repo_id, revision))

    def fake_run(command, check):
        merged = tmp_path / "weights" / streamrender_h3_assets.MERGED_BACKBONE
        merged.parent.mkdir(parents=True, exist_ok=True)
        merged.write_bytes(b"x")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot)
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "weights"
    prepare_weights.main(["--output", str(out)])
    streamrender_h3_assets.check_weights(out / "assets.json")
    assert calls == [(streamrender_h3_assets.REPO_ID, streamrender_h3_assets.REVISION)]
