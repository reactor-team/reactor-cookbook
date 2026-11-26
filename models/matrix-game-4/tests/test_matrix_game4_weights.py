# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Startup only verifies resources, and prepare_weights.py downloads them."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

MODEL_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODEL_DIR))

assets = importlib.import_module("matrix_game4_assets")
prepare_weights = importlib.import_module("prepare_weights")


def _fake_resources(monkeypatch, handler) -> None:
    module = ModuleType("matrix_game.resources")
    module.main = handler
    package = ModuleType("matrix_game")
    package.resources = module
    monkeypatch.setitem(sys.modules, "matrix_game", package)
    monkeypatch.setitem(sys.modules, "matrix_game.resources", module)


def test_manifest_uses_prepared_folder_offline() -> None:
    manifest = yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text())
    assert manifest["runtime"]["weights_path"] == "./weights"
    assert manifest["build"]["runtime_env"]["HF_HUB_OFFLINE"] == "1"
    assert assets.SOURCE_REVISION in " ".join(manifest["build"]["run"])
    assert len(assets.RESOURCE_REVISION) == 40


def test_check_only_verifies_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def handler(argv):
        seen.append(argv[0])
        raise ValueError("Missing or corrupted resource: x")

    _fake_resources(monkeypatch, handler)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda *a, **k: pytest.fail("must not download")
        ),
    )
    with pytest.raises(
        FileNotFoundError, match="prepare_weights.py --output ./weights"
    ):
        assets.check_resources(tmp_path, tmp_path / "resources", "castle")
    assert seen == ["verify"]


def test_prepare_downloads_each_scene_then_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "matrix_game").mkdir(parents=True)
    (source / "matrix_game/resources.py").write_text(
        "CALLS = []\n\ndef main(argv):\n    CALLS.append(argv)\n"
    )
    module = prepare_weights._load_resources(source)
    monkeypatch.setattr(prepare_weights, "_load_resources", lambda _source: module)
    prepare_weights.prepare(tmp_path / "weights", source)
    calls = module.CALLS
    downloads = [c for c in calls if c[0] == "download"]
    verifies = [c for c in calls if c[0] == "verify"]
    assert len(downloads) == len(verifies) == 6
    assert calls.index(verifies[0]) > calls.index(downloads[-1])
    for call in downloads:
        assert call[call.index("--revision") + 1] == assets.RESOURCE_REVISION
        assert call[call.index("--repo-id") + 1] == assets.RESOURCE_REPO
    assert {c[c.index("--scene") + 1] for c in calls} == set(assets.SCENES)
