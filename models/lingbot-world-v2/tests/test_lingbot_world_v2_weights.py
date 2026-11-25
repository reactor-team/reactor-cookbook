"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import prepare_weights
import pytest
import yaml
from lingbot_world_v2_assets import (
    _CHECKPOINT_FILES,
    PREPARE_COMMAND,
    _check_source,
    check_checkpoint,
    read_config,
)

MODEL_DIR = Path(__file__).parents[1]
CONFIG = MODEL_DIR / "lingbot_world_v2.yaml"


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace huggingface_hub with a fake whose snapshot_download writes the repository files."""
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for name in _CHECKPOINT_FILES:
            (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (local_dir / name).write_bytes(b"x")
        return str(local_dir)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_manifest_uses_the_prepared_folder_and_clones_the_pinned_source() -> None:
    config = read_config(CONFIG, Path("/weights"))
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    clone = manifest["build"]["run"][0]
    assert config.source_revision in clone and config.source_url in clone
    assert str(config.source_path) in clone
    assert config.checkpoint_path == Path("/weights")


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    (call,) = calls
    config = read_config(CONFIG, tmp_path)
    assert call["revision"] == config.checkpoint_revision
    assert call["repo_id"] == config.checkpoint_repo_id
    assert Path(call["local_dir"]) == tmp_path
    check_checkpoint(config.checkpoint_path)
    assert (tmp_path / "transformers" / "config.json").is_file()


def test_missing_checkpoint_names_the_fix_and_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match="Wan2.1_VAE.pth") as raised:
        check_checkpoint(tmp_path)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_without_the_transformer_config_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    (tmp_path / "transformers" / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"transformers/config\.json"):
        check_checkpoint(tmp_path)
    assert not (tmp_path / "transformers" / "config.json").exists()


def test_missing_source_asks_for_a_rebuild_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("git call")

    monkeypatch.setattr(subprocess, "run", refuse)
    config = replace(read_config(CONFIG, tmp_path), source_path=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="reactor build"):
        _check_source(config)
    assert not (tmp_path / "absent").exists()
