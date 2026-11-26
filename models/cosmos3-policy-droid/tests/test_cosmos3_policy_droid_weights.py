# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weights folder: what ``prepare_weights.py`` builds, and what ``load()`` reads from it."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

import cosmos3_policy_droid_model
import prepare_weights
from cosmos3_policy_droid_assets import (
    CHECKPOINT_IGNORE_PATTERNS,
    PREPARE_COMMAND,
    VAE,
    VAE_FILENAME,
    check_source,
    check_weights,
    read_config,
)
from cosmos3_policy_droid_model import Cosmos3PolicyModel

MODEL_DIR = Path(__file__).resolve().parent.parent
CONFIG = MODEL_DIR / "cosmos3_policy_droid.yaml"

# The Edge checkpoint repository by shape: the files serving reads, its two
# weight indexes, and the files prepare_weights.py leaves out.
SHARDS = (
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/cosmos_framework_model.safetensors",
    "vision_encoder/model.safetensors",
)
REPOSITORY = {
    "config.json": "{}",
    "model_index.json": "{}",
    "processor_config.json": "{}",
    "preprocessor_config.json": "{}",
    "video_preprocessor_config.json": "{}",
    "tokenizer.json": "{}",
    "tokenizer_config.json": "{}",
    "chat_template.jinja": "",
    "model.safetensors.index.json": json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(SHARDS)}}),
    "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
        {"weight_map": {"a": SHARDS[0].removeprefix("transformer/"), "b": SHARDS[1].removeprefix("transformer/")}}
    ),
    **{shard: "x" for shard in SHARDS},
    "README.md": "card",
    "images/cosmos3-edge-policy-droid-graph.png": "png",
    "convert_cosmos3_to_diffusers.py": "",
    ".gitattributes": "",
    "vae/config.json": "{}",
    "vae/diffusion_pytorch_model.safetensors": "x",
}
SKIPPED = ("README.md", "images/cosmos3-edge-policy-droid-graph.png", "convert_cosmos3_to_diffusers.py")


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace huggingface_hub with a fake that writes plain files the way local_dir downloads do."""
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.append({"kind": "snapshot", **kwargs})
        local_dir = Path(kwargs["local_dir"])
        for name, content in REPOSITORY.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in kwargs["ignore_patterns"]):
                continue
            (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (local_dir / name).write_text(content)
        return str(local_dir)

    def hf_hub_download(**kwargs: Any) -> str:
        calls.append({"kind": "file", **kwargs})
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    module.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_manifest_uses_the_prepared_folder_offline_and_fetches_the_pinned_source() -> None:
    config = read_config(CONFIG)
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert manifest["build"]["runtime_env"]["UV_OFFLINE"] == "1"
    steps = manifest["build"]["run"]
    assert not [step for step in steps if step.lstrip().startswith("--")]
    fetch = " ".join(steps)
    assert config.source.revision in fetch and config.source.url in fetch
    assert f"git init {config.source.path} " in fetch
    assert "checkout --detach FETCH_HEAD" in fetch
    for revision in (config.source.revision, config.checkpoint.revision, VAE.revision):
        assert len(revision) == 40
    assert "weights/" in (MODEL_DIR / ".dockerignore").read_text().splitlines()


def test_readme_names_the_pinned_revisions_and_the_deploy() -> None:
    config = read_config(CONFIG)
    readme = (MODEL_DIR / "README.md").read_text()
    for revision in (config.source.revision, config.checkpoint.revision, VAE.revision):
        assert revision in readme
    assert PREPARE_COMMAND in readme
    assert "reactor deploy production --weights ./weights" in readme


def test_prepare_builds_the_folder_that_load_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch)
    output = tmp_path / "weights"
    prepare_weights.main(["--output", str(output)])

    config = read_config(CONFIG)
    snapshot, vae = calls
    assert snapshot["repo_id"] == config.checkpoint.repo_id
    assert snapshot["revision"] == config.checkpoint.revision
    assert snapshot["ignore_patterns"] == list(CHECKPOINT_IGNORE_PATTERNS)
    assert Path(snapshot["local_dir"]) == output / config.checkpoint.repo_id
    assert (vae["repo_id"], vae["revision"], vae["filename"]) == (VAE.repo_id, VAE.revision, VAE_FILENAME)

    checkpoint, vae_file = check_weights(config, output)
    assert checkpoint == (output / config.checkpoint.repo_id).resolve()
    assert vae_file == (output / VAE.repo_id / VAE_FILENAME).resolve()
    for name in SKIPPED:
        assert not (checkpoint / name).exists()
    assert not [path for path in output.rglob("*") if path.is_symlink()]

    # A second run asks for the same pinned files and leaves a folder the check accepts.
    prepare_weights.main(["--output", str(output)])
    assert [call["revision"] for call in calls] == [config.checkpoint.revision, VAE.revision] * 2
    check_weights(config, output)


def test_a_shard_missing_from_the_index_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    config = read_config(CONFIG)
    shard = tmp_path / config.checkpoint.repo_id / SHARDS[1]
    shard.unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match=SHARDS[1]) as raised:
        check_weights(config, tmp_path)
    assert PREPARE_COMMAND in str(raised.value)


def test_missing_weights_name_the_fix_and_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(cosmos3_policy_droid_model, "check_source", lambda source, path: None)
    monkeypatch.setitem(sys.modules, "cosmos_framework", None)
    with pytest.raises(FileNotFoundError, match="nvidia/Cosmos3-Edge-Policy-DROID/config.json") as raised:
        Cosmos3PolicyModel().load(CONFIG, tmp_path)
    assert PREPARE_COMMAND in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_missing_vae_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_hub(monkeypatch)
    prepare_weights.main(["--output", str(tmp_path)])
    (tmp_path / VAE.repo_id / VAE_FILENAME).unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match=VAE_FILENAME):
        check_weights(read_config(CONFIG), tmp_path)


def test_missing_source_asks_for_a_rebuild_and_runs_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a missing checkout must not run git")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config = read_config(CONFIG)
    with pytest.raises(FileNotFoundError, match="rebuild the image with `reactor build`"):
        check_source(config.source, tmp_path / "absent")
