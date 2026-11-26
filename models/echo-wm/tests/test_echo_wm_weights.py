# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

from __future__ import annotations

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import echo_wm
import echo_wm_assets
import prepare_weights
import pytest
import yaml
from echo_wm_assets import (
    GEMMA_FILES,
    PREPARE_COMMAND,
    check_assets,
    check_source,
    check_weights,
    read_config,
)

MODEL_DIR = Path(__file__).parents[1]
CONFIG = MODEL_DIR / "echo_wm.yaml"


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace huggingface_hub with a fake that writes the requested files."""
    calls: list[dict[str, Any]] = []

    def hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in kwargs["allow_patterns"]:
            (local_dir / name).write_bytes(b"x")
        return str(local_dir)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def _record_subprocess(
    monkeypatch: pytest.MonkeyPatch, stdout: str = ""
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(echo_wm_assets.subprocess, "run", run)
    return calls


def test_manifest_uses_the_prepared_folder_and_clones_the_pinned_source() -> None:
    config = read_config(CONFIG, Path("/weights"))
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert "git" in manifest["build"]["system_packages"]
    steps = manifest["build"]["run"]
    assert not [step for step in steps if step.lstrip().startswith("--")]
    clone = " ".join(steps)
    assert f"git init {config.source_path}" in clone
    assert config.source_url in clone and config.source_revision in clone
    assert "weights/" in (MODEL_DIR / ".dockerignore").read_text().splitlines()
    assert config.checkpoint.path.is_relative_to("/weights")
    assert config.gemma.path.is_relative_to("/weights")
    assert not config.runtime_dir.is_relative_to("/weights")


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    config = read_config(CONFIG, tmp_path)
    checkpoint, gemma = calls
    assert checkpoint["repo_id"] == config.checkpoint.repo_id == "Echo-Team/Echo-WM"
    assert checkpoint["revision"] == config.checkpoint.revision
    assert checkpoint["filename"] == "echo-wm-flash.safetensors"
    assert gemma["repo_id"] == config.gemma.repo_id
    assert gemma["revision"] == config.gemma.revision
    assert gemma["allow_patterns"] == list(GEMMA_FILES)
    assert all(len(call["revision"]) == 40 for call in calls)
    check_weights(config)
    assert (tmp_path / "echo-wm" / "echo-wm-flash.safetensors").is_file()
    assert not [path for path in tmp_path.rglob("*") if path.is_symlink()]
    # Upstream loads every safetensors file below the Gemma folder.
    assert not list(config.gemma.path.rglob("echo-wm*.safetensors"))


def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = read_config(CONFIG, tmp_path)
    with pytest.raises(FileNotFoundError, match="echo-wm-flash.safetensors") as raised:
        check_weights(config)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_missing_gemma_file_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    config = read_config(CONFIG, tmp_path)
    (config.gemma.path / "tokenizer.model").unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match="tokenizer.model") as raised:
        check_weights(config)
    assert PREPARE_COMMAND in str(raised.value)


def test_load_checks_the_folder_and_writes_nothing_into_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(echo_wm, "get_weights_path", lambda: tmp_path)
    monkeypatch.setattr(echo_wm_assets, "check_source", lambda config: None)
    with pytest.raises(FileNotFoundError, match=r"prepare_weights\.py"):
        echo_wm.EchoWM().load(CONFIG)
    assert list(tmp_path.iterdir()) == []


def test_missing_source_asks_for_a_rebuild_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    calls = _record_subprocess(monkeypatch)
    config = replace(read_config(CONFIG, tmp_path), source_path=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="reactor build"):
        check_assets(config)
    assert calls == []
    assert not (tmp_path / "absent").exists()


def test_source_check_only_reads_the_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "JoyAI-Echo"
    (source / ".git").mkdir(parents=True)
    config = replace(read_config(CONFIG, tmp_path), source_path=source)
    calls = _record_subprocess(monkeypatch, stdout=config.source_revision + "\n")
    check_source(config)
    (command,) = calls
    assert command[-2:] == ["rev-parse", "HEAD"]
    assert f"safe.directory={source}" in command
    assert not {"clone", "fetch", "pull", "checkout"} & set(command)

    _record_subprocess(monkeypatch, stdout="0" * 40)
    with pytest.raises(RuntimeError, match="reactor build"):
        check_source(config)


def test_config_pins_every_source_by_full_commit(tmp_path: Path) -> None:
    values = yaml.safe_load(CONFIG.read_text())
    values["assets"]["gemma"]["revision"] = "main"
    config = tmp_path / "echo_wm.yaml"
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="assets.gemma.revision"):
        read_config(config)
