# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Checkpoint resolution in the weights folder, and the folder that ``prepare_weights.py`` builds."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import prepare_weights
import pytest
import yaml
from joy_video_edit_assets import read_config, resolve_checkpoints

_MODEL_DIR = Path(__file__).resolve().parent.parent
_CONFIG = _MODEL_DIR / "joy_video_edit.yaml"


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace huggingface_hub with a fake whose snapshot_download writes each repository's files."""
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        if kwargs["repo_id"] == "jdopensource/JoyAI-Video-Edit":
            (local_dir / "dit").mkdir(parents=True, exist_ok=True)
            (local_dir / "dit" / "joyai_video_edit_dit_0811.pth").write_bytes(b"dit")
            (local_dir / "vae").mkdir(exist_ok=True)
        else:
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_the_shipped_config_pins_full_revisions() -> None:
    section = read_config(_CONFIG)["checkpoints"]
    for name in ("joyai", "text_encoder"):
        assert len(section[name]["revision"]) == 40


def test_prepare_builds_the_layout_that_load_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    joyai, text_encoder = calls
    assert joyai["revision"] == "39491dda89fd2b535777fd8644a6631ea89b857c"
    assert joyai["allow_patterns"] == ["dit/joyai_video_edit_dit_0811.pth", "vae/*"]
    assert text_encoder["revision"] == "4bfb270765825d2fa059011deb4c96fdd579be6f"
    assert Path(text_encoder["local_dir"]) == tmp_path / "MiMo-VL-7B-RL-2508"

    checkpoints = resolve_checkpoints(read_config(_CONFIG), tmp_path)
    assert checkpoints.dit == tmp_path / "dit" / "joyai_video_edit_dit_0811.pth"
    assert checkpoints.vae == tmp_path / "vae"
    assert checkpoints.text_encoder == tmp_path / "MiMo-VL-7B-RL-2508"


def test_resolve_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoints(read_config(_CONFIG), tmp_path)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("missing", "name"),
    [
        ("dit/joyai_video_edit_dit_0811.pth", "DiT"),
        ("vae", "VAE"),
        ("MiMo-VL-7B-RL-2508", "text encoder"),
    ],
)
def test_a_missing_checkpoint_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str, name: str
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    target = tmp_path / missing
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    with pytest.raises(FileNotFoundError, match=f"{name} checkpoint not found") as raised:
        resolve_checkpoints(read_config(_CONFIG), tmp_path)
    assert "python prepare_weights.py --output ./weights" in str(raised.value)


def test_read_config_refuses_a_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="mapping"):
        read_config(path)
