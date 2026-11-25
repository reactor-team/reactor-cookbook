"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

import sys
import types
from dataclasses import replace
from pathlib import Path

import prepare_weights
import pytest
import yaml
from matrix_game_2_assets import (
    MODEL_ALLOW_PATTERNS,
    PREPARE_COMMAND,
    check_checkpoint,
    check_runtime_assets,
    read_config,
)

MODEL_DIR = Path(__file__).parents[1]
CONFIG = MODEL_DIR / "matrix_game_2.yaml"
FILES = (
    "base_distilled_model/base_distill.safetensors",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "xlm-roberta-large/tokenizer.json",
)


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace huggingface_hub with a fake whose snapshot_download writes the model files."""
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for name in FILES:
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
    clone = " ".join(manifest["build"]["run"])
    assert config.source_revision in clone and config.source_url in clone
    assert str(config.source_path.parent) in clone
    assert config.weights_root == Path("/weights")


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    (call,) = calls
    config = read_config(CONFIG, tmp_path)
    assert call["repo_id"] == config.model_repo_id
    assert call["revision"] == config.model_revision
    assert call["allow_patterns"] == list(MODEL_ALLOW_PATTERNS)
    assert Path(call["local_dir"]) == tmp_path
    check_checkpoint(config)


def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = read_config(CONFIG, tmp_path)
    with pytest.raises(FileNotFoundError, match="Wan2.1_VAE.pth") as raised:
        check_checkpoint(config)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_missing_source_names_the_rebuild_and_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = replace(read_config(CONFIG, tmp_path), source_path=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="reactor build"):
        check_runtime_assets(config)
