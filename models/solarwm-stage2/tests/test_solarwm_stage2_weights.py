# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

from __future__ import annotations

import fnmatch
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import prepare_weights
from solarwm_stage2_config import (
    ALLOW_PATTERNS,
    PREPARE_COMMAND,
    REQUIRED_FILES,
    check_runtime,
    check_source,
    check_weights,
    read_config,
)

MODEL_DIR = Path(__file__).parents[1]
CONFIG = MODEL_DIR / "solarwm_stage2.yaml"
# The repository tree at the pinned revision, below the two folders in use.
REPO_TREE = (
    "SolarWM-5B-base/LICENSE.txt",
    "SolarWM-5B-base/UPSTREAM_README.md",
    "SolarWM-5B-base/assets-manifest.json",
    "SolarWM-5B-base/conditioning/wan_negemb_cn.pth",
    "SolarWM-5B-base/release.json",
    "SolarWM-5B-base/text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
    "SolarWM-5B-base/tokenizer/special_tokens_map.json",
    "SolarWM-5B-base/tokenizer/spiece.model",
    "SolarWM-5B-base/tokenizer/tokenizer.json",
    "SolarWM-5B-base/tokenizer/tokenizer_config.json",
    "SolarWM-5B-base/transformer/config.json",
    "SolarWM-5B-base/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "SolarWM-5B-base/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "SolarWM-5B-base/transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "SolarWM-5B-base/transformer/diffusion_pytorch_model.safetensors.index.json",
    "SolarWM-5B-base/vae/Wan2.2_VAE.pth",
    "SolarWM-5B-sgf-stage2-81f/COMPLETE.json",
    "SolarWM-5B-sgf-stage2-81f/LICENSE.txt",
    "SolarWM-5B-sgf-stage2-81f/NOTICE.md",
    "SolarWM-5B-sgf-stage2-81f/checkpoint-manifest.json",
    "SolarWM-5B-sgf-stage2-81f/model.pt",
    "SolarWM-5B-sgf-stage2-81f/release-manifest.json",
    "SolarWM-5B-sgf-stage2-81f/release.json",
)


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace huggingface_hub with a fake that writes the matching, missing tree files."""
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for name in REPO_TREE:
            target = local_dir / name
            if target.exists() or not any(
                fnmatch.fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]
            ):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return str(local_dir)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    import solarwm_stage2_config

    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(solarwm_stage2_config.subprocess, "run", refuse)


def test_manifest_uses_the_prepared_folder_and_clones_the_pinned_source() -> None:
    config = read_config(CONFIG, Path("/weights"))
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    build = manifest["build"]
    assert build["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert "git" in build["system_packages"]
    assert not any(step.lstrip().startswith("--") for step in build["run"])
    clone = " ".join(build["run"])
    assert f"git init {config.source_path}" in clone
    assert f"{config.source_url} {config.source_revision}" in clone
    assert "checkout --detach FETCH_HEAD" in clone
    assert len(config.source_revision) == 40
    assert len(config.checkpoint_revision) == 40
    ignored = (MODEL_DIR / ".dockerignore").read_text().split()
    assert "weights/" in ignored
    readme = (MODEL_DIR / "README.md").read_text()
    assert "python prepare_weights.py --output ./weights" in readme
    assert "reactor deploy production --weights ./weights" in readme
    assert config.source_revision in readme and config.checkpoint_revision in readme


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    (call,) = calls
    config = read_config(CONFIG, tmp_path)
    assert call["repo_id"] == config.repo_id
    assert call["revision"] == config.checkpoint_revision
    assert call["allow_patterns"] == list(ALLOW_PATTERNS)
    assert Path(call["local_dir"]) == tmp_path
    check_weights(config)
    assert not (tmp_path / "SolarWM-5B-base/transformer").exists()


def test_prepare_fetches_an_empty_file_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    damaged = tmp_path / REQUIRED_FILES[0]
    damaged.write_bytes(b"")
    prepare_weights.main(["--output", str(tmp_path)])
    assert len(calls) == 2
    assert damaged.read_bytes() == b"x"


@pytest.mark.parametrize("name", REQUIRED_FILES)
@pytest.mark.parametrize("empty", [False, True])
def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, empty: bool
) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    damaged = tmp_path / name
    if empty:
        damaged.write_bytes(b"")
    else:
        damaged.unlink()
    before = sorted(tmp_path.rglob("*"))

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = read_config(CONFIG, tmp_path)
    with pytest.raises(FileNotFoundError, match=name) as raised:
        check_weights(config)
    assert PREPARE_COMMAND in str(raised.value)
    assert sorted(tmp_path.rglob("*")) == before


def test_empty_weights_folder_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = read_config(CONFIG, tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_weights.py"):
        check_weights(config)
    assert list(tmp_path.iterdir()) == []


def test_missing_source_asks_for_a_rebuild_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    _no_git(monkeypatch)
    config = replace(read_config(CONFIG, tmp_path), source_path=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="reactor build"):
        check_source(config)
    assert not (tmp_path / "absent").exists()


def test_startup_reads_the_revision_and_writes_outside_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import solarwm_stage2_config

    _fake_hub(monkeypatch)
    weights = tmp_path / "weights"
    prepare_weights.main(["--output", str(weights)])
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = replace(
        read_config(CONFIG, weights),
        source_path=tmp_path / "source",
        runtime_root=tmp_path / "cache",
    )
    (config.source_path / ".git").mkdir(parents=True)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(stdout=config.source_revision + "\n")

    monkeypatch.setattr(solarwm_stage2_config.subprocess, "run", run)
    before = sorted(weights.rglob("*"))
    check_runtime(config)
    (command,) = commands
    assert command[:3] == ["git", "-c", f"safe.directory={config.source_path}"]
    assert command[-2:] == ["rev-parse", "HEAD"]
    assert config.runtime_root.is_dir()
    assert sorted(weights.rglob("*")) == before

    commands.clear()
    monkeypatch.setattr(
        solarwm_stage2_config.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(stdout="0" * 40),
    )
    with pytest.raises(RuntimeError, match=config.source_revision):
        check_runtime(config)


def test_runtime_scratch_folder_is_outside_the_weights(tmp_path: Path) -> None:
    config = read_config(CONFIG, tmp_path)
    assert config.runtime_root == Path("~/.cache/solarwm-stage2").expanduser()
    assert not config.runtime_root.is_relative_to(tmp_path)
