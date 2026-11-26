# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The weight lookup never downloads, and prepare_weights.py builds what it checks."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

lyra2_assets = importlib.import_module("lyra2_assets")
prepare_weights = importlib.import_module("prepare_weights")


def fake_hub(calls: list[tuple[str, str]]) -> SimpleNamespace:
    def fake_download(repo_id, filename, revision, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        calls.append((repo_id, revision))

    return SimpleNamespace(hf_hub_download=fake_download)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert lyra2_assets.SOURCE_REVISION in " ".join(manifest["build"]["run"])
    revisions = [lyra2_assets.CHECKPOINT_REVISION, lyra2_assets.SOURCE_REVISION]
    revisions += [revision for _, revision, _ in lyra2_assets.HUB_REPOS.values()]
    assert all(len(revision) == 40 for revision in revisions)


def test_missing_weights_name_the_fix(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        lyra2_assets.check_weights(tmp_path)


def test_lookup_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the lookup must not download")

    fake = SimpleNamespace(hf_hub_download=refuse, snapshot_download=refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    with pytest.raises(FileNotFoundError):
        lyra2_assets.check_weights(tmp_path)
    config = tmp_path / "lyra2.yaml"
    config.write_text((MODEL_DIR / "lyra2.yaml").read_text())
    with pytest.raises(FileNotFoundError):
        lyra2_assets.prepare_config(config, tmp_path, tmp_path / "cache")


def test_missing_source_says_rebuild(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rebuild the image"):
        lyra2_assets.check_source(tmp_path)


def test_prepare_builds_the_folder_that_the_check_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls))
    out = tmp_path / "weights"
    prepare_weights.main(["--output", str(out)])
    lyra2_assets.check_weights(out)
    assert {repo for repo, _ in calls} == {
        lyra2_assets.CHECKPOINT_REPO,
        *(repo for repo, _, _ in lyra2_assets.HUB_REPOS.values()),
    }
    assert all(len(revision) == 40 for _, revision in calls)
    assert (out / "source/Lyra-2/checkpoints/model/model/.metadata").is_file()
    assert not [path for path in out.rglob("*") if path.is_symlink()]


def test_hub_cache_is_keyed_by_the_id_the_source_loads(tmp_path: Path) -> None:
    # CLIPModel loads the tokenizer as `xlm-roberta-large`; offline lookup uses
    # that id, not the repository it moved to.
    revision = lyra2_assets.HUB_REPOS["xlm-roberta-large"][1]
    folder = lyra2_assets.snapshot_dir(tmp_path, "xlm-roberta-large", revision)
    assert folder.parents[1].name == "models--xlm-roberta-large"
    umt5 = lyra2_assets.HUB_REPOS["google/umt5-xxl"][2]
    assert not [name for name in umt5 if name.startswith("pytorch_model")]


def test_prepare_config_writes_nothing_into_the_weights_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub([]))
    weights = tmp_path / "weights"
    prepare_weights.main(["--output", str(weights)])
    source = tmp_path / "source"
    (source / "gui" / "api").mkdir(parents=True)
    config = tmp_path / "lyra2.yaml"
    values = yaml.safe_load((MODEL_DIR / "lyra2.yaml").read_text())
    values["source_path"] = str(source)
    config.write_text(yaml.safe_dump(values))
    before = sorted(weights.rglob("*"))
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    result = lyra2_assets.prepare_config(config, weights, tmp_path / "cache")
    assert sorted(weights.rglob("*")) == before
    assert Path(result["output_path"]).is_relative_to(tmp_path / "cache")
    assert (source / "checkpoints").resolve() == Path(result["checkpoints_path"])
