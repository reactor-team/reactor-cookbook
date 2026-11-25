# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for scripts/check_model_weights.py on generated model folders."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_model_weights

PINNED = "hf://acme/world@" + "a" * 40
CLEAN_MANIFEST = """\
$schema: reactor/v2
runtime:
  import: world:World
  weights_path: ./weights
build:
  runtime_env:
    HF_HUB_OFFLINE: "1"
"""


def make_model(
    root: Path, name: str = "world", manifest: str = CLEAN_MANIFEST, **files: str
) -> Path:
    """Write models/<name>/ with a manifest and the given files.

    In a file key, ``__`` is ``/`` and a trailing ``_py`` is ``.py``.
    """
    model = root / "models" / name
    model.mkdir(parents=True)
    (model / "reactor.yaml").write_text(manifest)
    for key, body in files.items():
        rel = key.replace("__", "/")
        if rel.endswith("_py"):
            rel = rel[: -len("_py")] + ".py"
        path = model / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return model


def run(
    root: Path, capsys: pytest.CaptureFixture[str], baseline: str = ""
) -> tuple[int, str]:
    baseline_path = root / "baseline.txt"
    baseline_path.write_text(baseline)
    status = check_model_weights.run(root, baseline_path)
    return status, capsys.readouterr().out


def test_clean_model_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_model(
        tmp_path,
        world_assets_py="from pathlib import Path\n\ndef resolve(root: Path) -> Path:\n    return root / 'dit.pth'\n",
    )
    status, out = run(tmp_path, capsys)
    assert status == 0, out
    assert "1 models pass, 0 in the baseline" in out


@pytest.mark.parametrize(
    "source",
    [
        "from huggingface_hub import snapshot_download\nsnapshot_download('acme/world')\n",
        "import huggingface_hub\nhuggingface_hub.hf_hub_download('acme/world', 'x')\n",
        "import importlib\nimportlib.import_module('huggingface_hub').snapshot_download('acme/world')\n",
        "hub = object()\ngetattr(hub, 'hf_hub_download')\n",
        "import torch\ntorch.hub.load('acme/world', 'model')\n",
        "from torch import hub\nhub.load('acme/world', 'model')\n",
        "from urllib.request import urlretrieve\nurlretrieve('https://example.com/w.pth', 'w.pth')\n",
    ],
)
def test_download_in_own_module_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    make_model(tmp_path, world_assets_py=source)
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert "models/world/world_assets.py:" in out
    assert "at runtime" in out


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run(['git', 'clone', 'https://github.com/acme/world.git', '/src'])\n",
        "import subprocess\n\ndef git(*args):\n    subprocess.run(['git', *args])\n\ngit('fetch', '--depth', '1')\n",
        "import os\nos.system('git clone https://github.com/acme/world.git /src')\n",
        "import shutil, subprocess\nGIT = shutil.which('git')\nsubprocess.run([GIT, 'clone', 'https://github.com/acme/world.git', '/src'])\n",
        "import subprocess\nsubprocess.run(['bash', '-c', 'cd /src && git fetch origin'])\n",
        "import subprocess\np = '/src'\nsubprocess.run(f'git -C {p} pull', shell=True)\n",
        "import git\ngit.Repo.clone_from('https://github.com/acme/world.git', '/src')\n",
    ],
)
def test_git_network_command_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    make_model(tmp_path, world_assets_py=source)
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert "models/world/world_assets.py:" in out


def test_git_helper_in_another_module_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_model(
        tmp_path,
        world_utils_py="import subprocess\n\ndef run_git(args):\n    subprocess.run(['git', *args])\n",
        world_assets_py="from world_utils import run_git\nrun_git(['clone', 'https://github.com/acme/world.git'])\n",
    )
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert "models/world/world_utils.py:4: runs `git` with clone" in out


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.check_output(['git', '-C', '/opt/world', 'rev-parse', 'HEAD'])\n",
        "import subprocess\nACTIONS = ('push', 'pull')\nsubprocess.check_output(['git', '-C', '/opt/world', 'rev-parse', 'HEAD'])\n",
        '"""The build runs git clone, so load() never does."""\n',
        "digits = [1, 2]\nACTIONS = ('fetch', 'pull')\nprint(digits, ACTIONS)\n",
    ],
)
def test_git_without_network_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    make_model(tmp_path, world_assets_py=source)
    status, out = run(tmp_path, capsys)
    assert status == 0, out


def test_prepare_scripts_and_tests_may_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    download = "from huggingface_hub import snapshot_download\nsnapshot_download('acme/world')\n"
    clone = "import subprocess\nsubprocess.run(['git', 'clone', 'https://github.com/acme/world.git'])\n"
    make_model(
        tmp_path,
        prepare_weights_py=download,
        prepare_source_py=clone,
        tests__test_world_py=download,
    )
    status, out = run(tmp_path, capsys)
    assert status == 0, out


@pytest.mark.parametrize(
    ("files", "manifest", "message"),
    [
        (
            {"world_assets_py": "def broken(:\n"},
            CLEAN_MANIFEST,
            "world_assets.py:1: does not parse",
        ),
        ({}, "- a list\n", "reactor.yaml: is not a mapping"),
        ({}, "runtime: [\n", "reactor.yaml: does not parse"),
    ],
)
def test_malformed_file_is_a_violation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    files: dict[str, str],
    manifest: str,
    message: str,
) -> None:
    make_model(tmp_path, manifest=manifest, **files)
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert message in out


def test_folder_without_manifest_is_a_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "models" / "stray").mkdir(parents=True)
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert "models/stray: has no reactor.yaml" in out


def test_missing_offline_setting_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_model(
        tmp_path, manifest="$schema: reactor/v2\nruntime:\n  weights_path: ./weights\n"
    )
    status, out = run(tmp_path, capsys)
    assert status == 1
    assert "HF_HUB_OFFLINE" in out


@pytest.mark.parametrize(
    ("dockerfile", "passes"),
    [
        ("ENV HF_HUB_OFFLINE=1 \\\n    TRANSFORMERS_OFFLINE=1\n", True),
        ("ENV PYTHONUNBUFFERED=1 \\\n    HF_HUB_OFFLINE=1\n", True),
        ("ENV HF_HUB_OFFLINE 1\n", True),
        ("ENV HF_HUB_OFFLINE=true\n", True),
        ("# ENV HF_HUB_OFFLINE=1\n", False),
        ("ENV HF_HUB_OFFLINE=0\n", False),
        ("RUN echo HF_HUB_OFFLINE=1\n", False),
    ],
)
def test_dockerfile_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], dockerfile: str, passes: bool
) -> None:
    make_model(
        tmp_path,
        manifest="$schema: reactor/v2\nruntime:\n  weights_path: ./weights\n",
        Dockerfile="FROM python:3.12\n" + dockerfile,
    )
    status, out = run(tmp_path, capsys)
    assert (status == 0) is passes, out


@pytest.mark.parametrize(
    ("value", "passes"),
    [
        ('"1"', True),
        ("1", True),
        ('"true"', True),
        ("true", True),
        ('"ON"', True),
        ('"0"', False),
        ("false", False),
    ],
)
def test_runtime_env_offline_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str, passes: bool
) -> None:
    make_model(tmp_path, manifest=CLEAN_MANIFEST.replace('"1"', value))
    status, out = run(tmp_path, capsys)
    assert (status == 0) is passes, out


@pytest.mark.parametrize(
    ("weights_path", "passes"),
    [
        ("./weights", True),
        (PINNED, True),
        ("~/.cache/reactor_registry/world", False),
        ("hf://acme/world", False),
        ("hf://acme/world@main", False),
    ],
)
def test_weights_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], weights_path: str, passes: bool
) -> None:
    make_model(tmp_path, manifest=CLEAN_MANIFEST.replace("./weights", weights_path))
    status, out = run(tmp_path, capsys)
    assert (status == 0) is passes, out


def test_baseline_skips_listed_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_model(tmp_path, manifest="$schema: reactor/v2\n")
    status, out = run(tmp_path, capsys, baseline="# old models\nworld\n")
    assert status == 0, out
    assert "0 models pass, 1 in the baseline" in out


def test_baseline_fails_when_listed_model_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_model(tmp_path)
    status, out = run(tmp_path, capsys, baseline="world\n")
    assert status == 1
    assert "world: passes; remove it from baseline.txt" in out


def test_baseline_fails_on_unknown_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_model(tmp_path)
    status, out = run(tmp_path, capsys, baseline="gone\n")
    assert status == 1
    assert "gone is not a model folder" in out
