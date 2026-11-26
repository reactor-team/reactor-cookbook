# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import alayaworld_assets
import prepare_weights
import pytest
import yaml
from alayaworld_assets import (
    PREPARE_COMMAND,
    AlayaWorldConfig,
    check_source,
    check_weights,
    read_config,
)
from alayaworld_model import AlayaWorldModel

MODEL_DIR = Path(__file__).parents[1]
CONFIG = MODEL_DIR / "alayaworld.yaml"


def _assets(config: AlayaWorldConfig):
    return (config.model, config.gemma, config.da3_model)


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace huggingface_hub with a fake whose hf_hub_download writes the file."""
    calls: list[dict] = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def _record_git(monkeypatch: pytest.MonkeyPatch, config: AlayaWorldConfig) -> list:
    """Answer `git rev-parse` with each checkout's pinned revision and record every call."""
    revisions = {
        str(checkout.path): checkout.revision
        for checkout in (config.source, config.da3_source)
    }
    calls: list[list[str]] = []

    def run(arguments, **_):
        calls.append(list(arguments))
        path = arguments[arguments.index("-C") + 1]
        return subprocess.CompletedProcess(arguments, 0, f"{revisions[path]}\n", "")

    monkeypatch.setattr(alayaworld_assets.subprocess, "run", run)
    return calls


def _fake_sources(tmp_path: Path, config: AlayaWorldConfig) -> AlayaWorldConfig:
    """Point both checkouts at folders holding the upstream files that load() checks."""
    source = tmp_path / "AlayaWorld"
    da3 = tmp_path / "Depth-Anything-3"
    da3.mkdir(parents=True)
    moved = replace(
        config,
        source=replace(config.source, path=source),
        da3_source=replace(config.da3_source, path=da3),
        upstream_config=source / config.upstream_config.relative_to(config.source.path),
        upload_template=source / config.upload_template.relative_to(config.source.path),
        random_inputs=tuple(
            source / path.relative_to(config.source.path)
            for path in config.random_inputs
        ),
    )
    moved.upstream_config.parent.mkdir(parents=True)
    moved.upstream_config.write_text("paths: {}\n")
    for scene in {moved.upload_template, *moved.random_inputs}:
        scene.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("_image.png", "_camera.pt", "_prompt.txt"):
            Path(f"{scene}{suffix}").write_bytes(b"x")
    return moved


def test_manifest_uses_the_prepared_folder_and_clones_the_pinned_sources() -> None:
    config = read_config(CONFIG, Path("/weights"))
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert "git" in manifest["build"]["system_packages"]
    steps = manifest["build"]["run"]
    assert not any(step.lstrip().startswith("--") for step in steps)
    for checkout in (config.source, config.da3_source):
        (step,) = [step for step in steps if checkout.url in step]
        assert checkout.revision in step and f"git init {checkout.path} " in step
        assert "checkout --detach FETCH_HEAD" in step
    assert "weights/" in (MODEL_DIR / ".dockerignore").read_text().split()
    readme = (MODEL_DIR / "README.md").read_text()
    assert "reactor deploy production --weights ./weights" in readme
    assert PREPARE_COMMAND in readme
    for asset in _assets(config):
        assert asset.path.is_relative_to(Path("/weights"))
        assert f"{asset.repo_id}@{asset.revision}" in readme
    for checkout in (config.source, config.da3_source):
        assert checkout.revision in readme


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    config = read_config(CONFIG, tmp_path)
    expected = {
        (asset.repo_id, asset.revision, name, asset.path)
        for asset in _assets(config)
        for name in asset.files
    }
    requested = {
        (call["repo_id"], call["revision"], call["filename"], Path(call["local_dir"]))
        for call in calls
    }
    assert requested == expected and len(calls) == len(expected)
    assert config.checkpoint_path.is_file()
    assert (config.da3_model.path / "model.safetensors").is_file()
    check_weights(config)


def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    weights = tmp_path / "weights"
    weights.mkdir()
    config = _fake_sources(tmp_path / "opt", read_config(CONFIG, weights))
    git = _record_git(monkeypatch, config)
    with pytest.raises(FileNotFoundError, match="merged_infer.safetensors") as raised:
        AlayaWorldModel().load(config)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(weights.iterdir()) == []
    assert all("rev-parse" in call for call in git) and len(git) == 2


def test_present_sources_are_only_read_at_their_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fake_sources(tmp_path, read_config(CONFIG, tmp_path / "weights"))
    git = _record_git(monkeypatch, config)
    check_source(config)
    assert [call[call.index("-C") + 2 :] for call in git] == [["rev-parse", "HEAD"]] * 2
    assert all(call[1:3] == ["-c", f"safe.directory={call[4]}"] for call in git)


@pytest.mark.parametrize("checkout", ["source", "da3_source"])
def test_missing_source_asks_for_a_rebuild_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkout: str
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = _fake_sources(tmp_path, read_config(CONFIG, tmp_path / "weights"))
    absent = tmp_path / "absent"
    config = replace(
        config, **{checkout: replace(getattr(config, checkout), path=absent)}
    )
    git = _record_git(monkeypatch, config)
    with pytest.raises(FileNotFoundError, match="reactor build") as raised:
        AlayaWorldModel().load(config)
    assert str(absent) in str(raised.value)
    assert all("rev-parse" in call and str(absent) not in call for call in git)
    assert not absent.exists()
