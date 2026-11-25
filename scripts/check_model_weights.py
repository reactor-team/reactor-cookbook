# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check that every model under models/ starts from its image and weights folder only.

The rules are GUIDELINES.md's "Weights and startup". For each model folder:

- its own modules (the ``*.py`` files at the top of the folder, except
  ``prepare_weights.py``) reference no hub or URL download function and run no
  ``git`` network command;
- ``reactor.yaml``'s ``build.runtime_env`` or the Dockerfile sets
  ``HF_HUB_OFFLINE`` to 1;
- ``runtime.weights_path``, when set, is ``./weights`` or an ``hf://`` reference
  pinned to a 40-character commit.

A model named in ``scripts/model_weights_baseline.txt`` predates the rules and is
skipped. A listed model that passes fails the check, so the list only shrinks.
Exit status is 0 when every model passes, else 1.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

DOWNLOAD_NAMES = frozenset(
    {
        "snapshot_download",
        "hf_hub_download",
        "load_state_dict_from_url",
        "download_url_to_file",
        "urlretrieve",
    }
)
GIT_NETWORK_COMMANDS = frozenset({"clone", "fetch", "pull", "ls-remote", "submodule"})
PREPARE_SCRIPT = "prepare_weights.py"
PINNED_HF_REF = re.compile(r"^hf://[^/@\s]+/[^/@\s]+@[0-9a-f]{40}$")
DOCKERFILE_OFFLINE = re.compile(
    r"""\bHF_HUB_OFFLINE\s*=\s*["']?1["']?(\s|$)""", re.MULTILINE
)


@dataclass(frozen=True)
class Violation:
    """One broken rule, at a path relative to the repository root."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.message}"


def _is_torch_hub_load(node: ast.Attribute) -> bool:
    return (
        node.attr == "load"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "hub"
    )


def _git_list_line(node: ast.AST) -> int | None:
    """Return the line of a list or tuple display whose first element is the string ``git``."""
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and first.value == "git":
            return node.lineno
    return None


def check_module(path: Path, rel: str) -> list[Violation]:
    """Find download references and ``git`` network commands in one module."""
    tree = ast.parse(path.read_text(), filename=rel)
    found: list[Violation] = []
    git_lines: list[int] = []
    network_words: set[str] = set()
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            if _is_torch_hub_load(node):
                found.append(
                    Violation(rel, node.lineno, "torch.hub.load downloads at runtime")
                )
                continue
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in DOWNLOAD_NAMES:
                name = node.value
            words = node.value.split()
            if words and words[0] in GIT_NETWORK_COMMANDS:
                network_words.add(words[0])
            if (
                len(words) > 1
                and words[0] == "git"
                and words[1] in GIT_NETWORK_COMMANDS
            ):
                found.append(
                    Violation(rel, node.lineno, f"runs `git {words[1]}` at runtime")
                )
        if name in DOWNLOAD_NAMES:
            line = getattr(node, "lineno", 0)
            found.append(
                Violation(
                    rel,
                    line,
                    f"`{name}` downloads at runtime; move it to {PREPARE_SCRIPT}",
                )
            )
        line = _git_list_line(node)
        if line is not None:
            git_lines.append(line)
    if git_lines and network_words:
        commands = ", ".join(sorted(network_words))
        found.append(
            Violation(
                rel,
                min(git_lines),
                f"runs `git` with {commands} at runtime; fetch the source in the build",
            )
        )
    # An import alias and its later use name the same function on one line.
    return sorted(set(found), key=lambda v: (v.line, v.message))


def check_manifest(model: Path, rel: str) -> list[Violation]:
    """Check the hub-offline setting and ``runtime.weights_path`` of one model."""
    found: list[Violation] = []
    manifest = yaml.safe_load((model / "reactor.yaml").read_text()) or {}
    runtime_env = (manifest.get("build") or {}).get("runtime_env") or {}
    offline = str(runtime_env.get("HF_HUB_OFFLINE", "")) == "1"
    dockerfile = model / "Dockerfile"
    if not offline and dockerfile.is_file():
        offline = DOCKERFILE_OFFLINE.search(dockerfile.read_text()) is not None
    if not offline:
        found.append(
            Violation(
                f"{rel}/reactor.yaml",
                0,
                'set build.runtime_env.HF_HUB_OFFLINE: "1" (or ENV in the Dockerfile)',
            )
        )
    weights_path = (manifest.get("runtime") or {}).get("weights_path")
    if (
        weights_path is not None
        and weights_path != "./weights"
        and not PINNED_HF_REF.match(str(weights_path))
    ):
        found.append(
            Violation(
                f"{rel}/reactor.yaml",
                0,
                f"runtime.weights_path {weights_path!r} is not ./weights or hf://<org>/<repo>@<40-hex commit>",
            )
        )
    return found


def check_model(model: Path, root: Path) -> list[Violation]:
    """Return every rule one model folder breaks."""
    rel = model.relative_to(root).as_posix()
    found = check_manifest(model, rel)
    for module in sorted(model.glob("*.py")):
        if module.name != PREPARE_SCRIPT:
            found.extend(check_module(module, f"{rel}/{module.name}"))
    return found


def read_baseline(path: Path) -> set[str]:
    """Read model names, one per line; ``#`` starts a comment."""
    if not path.is_file():
        return set()
    names = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return {name for name in names if name}


def run(root: Path, baseline_path: Path) -> int:
    """Check every model under ``root/models`` and print the result; return the exit status."""
    models = sorted(
        p for p in (root / "models").iterdir() if (p / "reactor.yaml").is_file()
    )
    baseline = read_baseline(baseline_path)
    failed = False
    for name in sorted(baseline - {m.name for m in models}):
        print(f"{baseline_path.name}: {name} is not a model folder; remove it")
        failed = True
    for model in models:
        violations = check_model(model, root)
        if model.name in baseline:
            if not violations:
                print(f"{model.name}: passes; remove it from {baseline_path.name}")
                failed = True
            continue
        for violation in violations:
            print(violation)
            failed = True
    if not failed:
        print(
            f"{len(models) - len(baseline)} models pass, {len(baseline)} in the baseline"
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args(argv)
    baseline = args.baseline or args.root / "scripts" / "model_weights_baseline.txt"
    return run(args.root, baseline)


if __name__ == "__main__":
    sys.exit(main())
