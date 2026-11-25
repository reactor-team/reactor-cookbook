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

sana_wm_assets = importlib.import_module("sana_wm_assets")
prepare_weights = importlib.import_module("prepare_weights")


def _config(root: Path):
    return sana_wm_assets.read_config(MODEL_DIR / "sana_wm.yaml", root)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    build = " ".join(manifest["build"]["run"])
    config = _config(Path("/w"))
    assert config.source_revision in build
    assert config.pi3x_source_revision in build
    assert str(config.source_path) in build
    assert str(config.pi3x_source_path) in build


def test_config_reads_layout_from_weights_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.streaming_path == tmp_path / "SANA-WM_streaming"
    assert config.text_encoder_path == tmp_path / "gemma-2-2b-it"
    assert config.pi3x_path == tmp_path / "Pi3X"
    assert config.source_path == Path("/opt/SANA")


def test_missing_weights_name_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        sana_wm_assets.check_weights(_config(tmp_path))


def test_check_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the check must not download")

    fake = SimpleNamespace(hf_hub_download=refuse, snapshot_download=refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    config = replace(_config(tmp_path), source_path=tmp_path / "no-source")
    with pytest.raises(FileNotFoundError, match="rebuild the image"):
        sana_wm_assets.check_runtime(config)
    with pytest.raises(FileNotFoundError):
        sana_wm_assets.check_weights(config)


def test_caches_leave_the_weights_folder_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    root = sana_wm_assets.configure_runtime_caches()
    assert root == tmp_path / ".cache" / "sana-wm"
    assert (root / "huggingface").is_dir()


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
    sana_wm_assets.check_weights(config)
    assert {c[0] for c in calls} == {
        config.streaming.repo_id,
        config.stage1_text_encoder.repo_id,
        config.pi3x_model.repo_id,
    }
    assert all(len(c[1]) == 40 for c in calls)
