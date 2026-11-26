# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

from __future__ import annotations

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

import hy_world_1_5_assets as assets
import prepare_weights

CONFIG = MODEL_DIR / "hy_world_1_5.yaml"
# The folders, relative to the base model, from which the pinned upstream
# pipeline (hyvideo/pipelines/worldplay_video_pipeline.py) loads each part.
UPSTREAM_READS = (
    "transformer/480p_i2v/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/llm/config.json",
    "text_encoder/byt5-small/config.json",
    "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
    "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
    "text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
    "vision_encoder/siglip/image_encoder/config.json",
    "vision_encoder/siglip/feature_extractor/preprocessor_config.json",
)


def _config(root: Path) -> assets.HYWorld15Config:
    return assets.read_config(CONFIG, root)


def _fill(config: assets.HYWorld15Config) -> None:
    """Write every file that the weights check expects."""
    for file in (*assets.huggingface_files(config), *assets.modelscope_files(config)):
        file.path.parent.mkdir(parents=True, exist_ok=True)
        file.path.write_bytes(b"x")


def _tree(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*")}


def _fake_hubs(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[dict]]:
    """Replace huggingface_hub and modelscope with fakes that write the requested files."""
    hf_calls: list[dict] = []
    ms_calls: list[dict] = []

    def hf_hub_download(**kwargs):
        hf_calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    def snapshot_download(model_id, **kwargs):
        ms_calls.append({"model_id": model_id, **kwargs})
        for name in kwargs["allow_patterns"]:
            target = Path(kwargs["local_dir"]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return kwargs["local_dir"]

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    modelscope_download = types.ModuleType("modelscope.hub.snapshot_download")
    modelscope_download.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", types.ModuleType("modelscope"))
    monkeypatch.setitem(
        sys.modules, "modelscope.hub", types.ModuleType("modelscope.hub")
    )
    monkeypatch.setitem(
        sys.modules, "modelscope.hub.snapshot_download", modelscope_download
    )
    return hf_calls, ms_calls


def _block_hubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "modelscope", None)


def _record_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every subprocess command, and still run it."""
    commands: list[list[str]] = []
    run = subprocess.run

    def recording(command, *args, **kwargs):
        commands.append(list(command))
        return run(command, *args, **kwargs)

    monkeypatch.setattr(assets.subprocess, "run", recording)
    return commands


def test_manifest_uses_the_prepared_folder_and_clones_the_pinned_source() -> None:
    config = _config(Path("/weights"))
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert "git" in manifest["build"]["system_packages"]
    steps = manifest["build"]["run"]
    assert not any(step.lstrip().startswith("--") for step in steps)
    clone = " ".join(steps)
    assert config.source.revision in clone and config.source.url in clone
    assert f"git init {config.source.path}" in clone
    assert "weights/" in (MODEL_DIR / ".dockerignore").read_text().split()


def test_every_source_is_pinned_by_full_revision() -> None:
    config = _config(Path("/weights"))
    files = (*assets.huggingface_files(config), *assets.modelscope_files(config))
    assert {file.repo_id for file in files} == {
        "tencent/HunyuanVideo-1.5",
        "tencent/HY-WorldPlay",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "google/byt5-small",
        "AI-ModelScope/Glyph-SDXL-v2",
        "google/siglip-so400m-patch14-384",
    }
    assert all(len(file.revision) == 40 for file in files)
    assert all(file.path.is_relative_to("/weights") for file in files)
    assert config.cache_path.is_relative_to(Path.home() / ".cache")


def test_prepare_builds_the_folder_that_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_calls, ms_calls = _fake_hubs(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])

    config = _config(tmp_path)
    expected = assets.huggingface_files(config)
    assert [
        (call["repo_id"], call["revision"], call["filename"], Path(call["local_dir"]))
        for call in hf_calls
    ] == [(f.repo_id, f.revision, f.filename, f.local_dir) for f in expected]
    (glyph_call,) = ms_calls
    assert glyph_call["model_id"] == config.glyph.repo_id
    assert glyph_call["revision"] == config.glyph.revision
    assert glyph_call["allow_patterns"] == list(assets.GLYPH_FILES)
    assert Path(glyph_call["local_dir"]) == config.glyph.path

    assets.check_weights(config)
    for relative in UPSTREAM_READS:
        assert (config.base_model.path / relative).is_file(), relative
    assert (
        config.action_model.path / "ar_distilled_action_model/model.safetensors"
    ).is_file()
    assert not any(path.is_symlink() for path in tmp_path.rglob("*"))


def test_prepare_is_safe_to_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hubs(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    before = _tree(tmp_path)
    prepare_weights.main(["--output", str(tmp_path)])
    assert _tree(tmp_path) == before


def test_missing_weights_name_the_fix_and_download_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_hubs(monkeypatch)
    config = _config(tmp_path)
    _fill(config)
    checkpoint = (
        config.action_model.path / "ar_distilled_action_model/model.safetensors"
    )
    checkpoint.unlink()
    before = _tree(tmp_path)

    with pytest.raises(FileNotFoundError, match="model.safetensors") as raised:
        assets.check_weights(config)
    assert str(checkpoint) in str(raised.value)
    assert assets.PREPARE_COMMAND in str(raised.value)
    assert _tree(tmp_path) == before


def test_empty_weights_folder_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_hubs(monkeypatch)
    with pytest.raises(FileNotFoundError, match="prepare_weights.py"):
        assets.check_weights(_config(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_missing_source_asks_for_a_rebuild_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_hubs(monkeypatch)
    commands = _record_commands(monkeypatch)
    config = _config(tmp_path)
    config = replace(config, source=replace(config.source, path=tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="reactor build"):
        assets.check_runtime_assets(config)
    assert commands == []
    assert not (tmp_path / "absent").exists()


def test_source_check_reads_the_revision_and_never_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "hyvideo/pipelines").mkdir(parents=True)
    (source / "hyvideo/pipelines/worldplay_video_pipeline.py").write_text("")
    (source / "assets").mkdir()
    (source / "assets/test_case.csv").write_text("image_name,caption\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(source)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "source"], check=True)

    commands = _record_commands(monkeypatch)
    config = _config(tmp_path / "weights")
    config = replace(config, source=replace(config.source, path=source))
    with pytest.raises(RuntimeError, match="reactor build"):
        assets.check_source(config)
    assert commands and all("rev-parse" in command for command in commands)


def test_load_checks_the_folder_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hy_world_1_5

    _block_hubs(monkeypatch)
    source = tmp_path / "source"
    for name in (
        ".git",
        "hyvideo/pipelines/worldplay_video_pipeline.py",
        "assets/test_case.csv",
    ):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("")
    weights = tmp_path / "weights"
    weights.mkdir()
    revision = _config(weights).source.revision
    monkeypatch.setenv(assets.SOURCE_ENV, str(source))
    monkeypatch.setattr(
        assets,
        "_run_git",
        lambda _arguments: subprocess.CompletedProcess([], 0, revision + "\n", ""),
    )
    monkeypatch.setattr(hy_world_1_5, "get_weights_path", lambda: weights)
    app = hy_world_1_5.HYWorld15()
    loaded: list[object] = []
    monkeypatch.setattr(app._engine, "load", loaded.append)

    with pytest.raises(FileNotFoundError, match="prepare_weights.py"):
        app.load(CONFIG)
    assert loaded == []
    assert list(weights.iterdir()) == []
