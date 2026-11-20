# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Checkpoint resolution: pinned downloads into the weights root, the revision markers, and ``checkpoint_dir``."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from joy_video_edit_assets import read_config, resolve_checkpoints

_MODEL_DIR = Path(__file__).resolve().parent.parent


def _config(**overrides: Any) -> dict[str, Any]:
    config = read_config(_MODEL_DIR / "joy_video_edit.yaml")
    config.update(overrides)
    return config


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace huggingface_hub with a fake whose snapshot_download writes the requested layout."""
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        if kwargs["repo_id"] == "jdopensource/JoyAI-Video-Edit":
            (local_dir / "dit").mkdir(parents=True, exist_ok=True)
            (local_dir / "dit" / "joyai_video_edit_dit_0811.pth").write_bytes(b"dit")
            (local_dir / "vae").mkdir(exist_ok=True)
        else:
            (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_the_shipped_config_pins_full_revisions() -> None:
    section = _config()["checkpoints"]
    for name in ("joyai", "text_encoder"):
        assert len(section[name]["revision"]) == 40


def test_first_load_downloads_the_pinned_snapshots_and_later_loads_reuse_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    checkpoints = resolve_checkpoints(_config(), tmp_path)
    assert checkpoints.dit == tmp_path / "dit" / "joyai_video_edit_dit_0811.pth"
    assert checkpoints.vae == tmp_path / "vae"
    assert checkpoints.text_encoder == tmp_path / "MiMo-VL-7B-RL-2508"
    joyai, text_encoder = calls
    assert joyai["revision"] == "39491dda89fd2b535777fd8644a6631ea89b857c"
    assert joyai["allow_patterns"] == ["dit/joyai_video_edit_dit_0811.pth", "vae/*"]
    assert Path(text_encoder["local_dir"]) == tmp_path / "MiMo-VL-7B-RL-2508"

    resolve_checkpoints(_config(), tmp_path)
    assert len(calls) == 2


def test_a_new_revision_downloads_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch)
    resolve_checkpoints(_config(), tmp_path)
    config = _config()
    config["checkpoints"]["joyai"]["revision"] = "f" * 40
    resolve_checkpoints(config, tmp_path)
    assert [c["revision"] for c in calls][-1] == "f" * 40


def test_checkpoint_dir_is_used_as_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch)
    staged = tmp_path / "staged"
    (staged / "dit").mkdir(parents=True)
    (staged / "dit" / "joyai_video_edit_dit_0811.pth").write_bytes(b"dit")
    (staged / "vae").mkdir()
    (staged / "MiMo-VL-7B-RL-2508").mkdir()
    checkpoints = resolve_checkpoints(_config(checkpoint_dir=str(staged)), tmp_path / "unused")
    assert checkpoints.dit.parent.parent == staged
    assert calls == []


def test_a_relative_checkpoint_dir_or_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        resolve_checkpoints(_config(checkpoint_dir="weights"), tmp_path)
    with pytest.raises(FileNotFoundError, match="DiT"):
        resolve_checkpoints(_config(checkpoint_dir=str(tmp_path)), tmp_path)


def test_read_config_refuses_a_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="mapping"):
        read_config(path)
