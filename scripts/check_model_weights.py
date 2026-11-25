# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check that every model under models/ starts from its image and weights folder only.

The rules are GUIDELINES.md's "Weights and startup". For each model folder:

- its top-level modules (``*.py`` at the top of the folder, except
  ``prepare_*.py``) reference no hub or URL download function and run no
  ``git`` network command;
- ``reactor.yaml``'s ``build.runtime_env`` or an ``ENV`` in the Dockerfile
  turns ``HF_HUB_OFFLINE`` on;
- ``runtime.weights_path``, when set, is ``./weights`` or an ``hf://`` reference
  pinned to a 40-character commit.

The check reads source, so it cannot see every way to reach the network: a
``urlopen`` call, or code in a subfolder, passes. ``HF_HUB_OFFLINE`` is the net
for hub calls it misses.

A model named in ``scripts/model_weights_baseline.txt`` predates the rules and is
skipped. A listed model that passes fails the check, so the list only shrinks.
Exit status is 0 when every model passes, else 1.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
# The values huggingface_hub reads as true for HF_HUB_OFFLINE.
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
PINNED_HF_REF = re.compile(r"^hf://[^/@\s]+/[^/@\s]+@[0-9a-f]{40}$")
GIT_VARIABLE = re.compile(r"(?:^|_)git(?:_?(?:bin|binary|cmd|exe|executable|path))?$")
GIT_SHELL_TEXT = re.compile(
    r"\bgit\b(?:\s+\S+)*?\s+(clone|fetch|pull|ls-remote|submodule)\b"
)
DOCKERFILE_OFFLINE = re.compile(
    r"""(?:^|\s)HF_HUB_OFFLINE(?:\s*=\s*|\s+)["']?(\w+)["']?""", re.IGNORECASE
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


@dataclass
class ModuleScan:
    """What one module shows about downloads and ``git`` calls."""

    violations: list[Violation] = field(default_factory=list)
    # A ``git`` command list with variable arguments, as (module, line).
    git_runners: list[tuple[str, int]] = field(default_factory=list)
    # A network command word passed first to a call or list: ``run_git(["clone", ...])``.
    network_leads: set[str] = field(default_factory=set)


def _const(node: ast.AST) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _names_git(node: ast.AST) -> bool:
    """True for the string ``git`` or a variable whose name holds ``git`` (``GIT``, ``self.git_bin``)."""
    if _const(node) == "git":
        return True
    name = (
        node.id
        if isinstance(node, ast.Name)
        else node.attr
        if isinstance(node, ast.Attribute)
        else ""
    )
    return GIT_VARIABLE.search(name.lower()) is not None


def _is_hub_load(node: ast.Attribute) -> bool:
    """True for ``torch.hub.load`` and for ``hub.load`` after ``from torch import hub``."""
    if node.attr != "load":
        return False
    value = node.value
    return (isinstance(value, ast.Attribute) and value.attr == "hub") or (
        isinstance(value, ast.Name) and value.id == "hub"
    )


def _text(node: ast.AST) -> str | None:
    """The literal text of a string or f-string; each placeholder becomes ``{}``."""
    text = _const(node)
    if text is not None:
        return text
    if isinstance(node, ast.JoinedStr):
        return "".join(_const(part) or "{}" for part in node.values)
    return None


def scan_module(source: str, rel: str) -> ModuleScan:
    """Find download references and ``git`` network commands in one module."""
    scan = ModuleScan()
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as err:
        scan.violations.append(
            Violation(rel, err.lineno or 0, f"does not parse: {err.msg}")
        )
        return scan
    # A bare string statement (a docstring) is prose, not a command.
    prose = {id(stmt.value) for stmt in ast.walk(tree) if isinstance(stmt, ast.Expr)}
    found = scan.violations
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.alias):
            name = node.name.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Attribute):
            if _is_hub_load(node):
                found.append(
                    Violation(
                        rel,
                        line,
                        "`hub.load` downloads at runtime; move it to prepare_weights.py",
                    )
                )
            elif node.attr == "clone_from":
                found.append(
                    Violation(
                        rel,
                        line,
                        "`clone_from` clones at runtime; fetch the source in the build",
                    )
                )
            name = node.attr
        elif id(node) not in prose and (text := _text(node)) is not None:
            if text in DOWNLOAD_NAMES:
                name = text
            match = GIT_SHELL_TEXT.search(text)
            if match:
                found.append(
                    Violation(
                        rel,
                        line,
                        f"runs `git {match.group(1)}` at runtime; fetch the source in the build",
                    )
                )
        if name in DOWNLOAD_NAMES:
            found.append(
                Violation(
                    rel,
                    line,
                    f"`{name}` downloads at runtime; move it to prepare_weights.py",
                )
            )

        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            first = _const(node.elts[0])
            if first in GIT_NETWORK_COMMANDS:
                scan.network_leads.add(first)
            if _names_git(node.elts[0]):
                words = sorted(
                    {
                        w
                        for e in node.elts[1:]
                        if (w := _const(e)) in GIT_NETWORK_COMMANDS
                    }
                )
                if words:
                    found.append(
                        Violation(
                            rel,
                            line,
                            f"runs `git {words[0]}` at runtime; fetch the source in the build",
                        )
                    )
                elif any(_const(e) is None for e in node.elts[1:]):
                    scan.git_runners.append((rel, line))
        if (
            isinstance(node, ast.Call)
            and node.args
            and (first := _const(node.args[0])) in GIT_NETWORK_COMMANDS
        ):
            scan.network_leads.add(first)
    # An import alias and its later use name the same function on one line.
    scan.violations = sorted(set(found), key=lambda v: (v.line, v.message))
    return scan


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dockerfile_offline(text: str) -> bool:
    """True when an ``ENV`` instruction turns ``HF_HUB_OFFLINE`` on; comments do not count."""
    instructions = re.sub(r"\\\r?\n", " ", text).splitlines()
    for instruction in instructions:
        words = instruction.strip().split(None, 1)
        if len(words) == 2 and words[0].upper() == "ENV":
            match = DOCKERFILE_OFFLINE.search(" " + words[1])
            if match and match.group(1).lower() in TRUE_VALUES:
                return True
    return False


def check_manifest(model: Path, rel: str) -> list[Violation]:
    """Check the hub-offline setting and ``runtime.weights_path`` of one model."""
    where = f"{rel}/reactor.yaml"
    try:
        manifest = yaml.safe_load((model / "reactor.yaml").read_text())
    except yaml.YAMLError as err:
        return [Violation(where, 0, f"does not parse: {err}")]
    if manifest is not None and not isinstance(manifest, dict):
        return [Violation(where, 0, "is not a mapping")]
    manifest = _mapping(manifest)
    found: list[Violation] = []
    offline_value = _mapping(_mapping(manifest.get("build")).get("runtime_env")).get(
        "HF_HUB_OFFLINE"
    )
    offline = str(offline_value).strip().lower() in TRUE_VALUES
    dockerfile = model / "Dockerfile"
    if not offline and dockerfile.is_file():
        offline = _dockerfile_offline(dockerfile.read_text())
    if not offline:
        found.append(
            Violation(
                where,
                0,
                'set build.runtime_env.HF_HUB_OFFLINE: "1" (or ENV in the Dockerfile)',
            )
        )
    weights_path = _mapping(manifest.get("runtime")).get("weights_path")
    if (
        weights_path is not None
        and weights_path != "./weights"
        and not PINNED_HF_REF.match(str(weights_path))
    ):
        found.append(
            Violation(
                where,
                0,
                f"runtime.weights_path {weights_path!r} is not ./weights or hf://<org>/<repo>@<40-hex commit>",
            )
        )
    return found


def check_model(model: Path, root: Path) -> list[Violation]:
    """Return every rule one model folder breaks."""
    rel = model.relative_to(root).as_posix()
    if not (model / "reactor.yaml").is_file():
        return [Violation(rel, 0, "has no reactor.yaml")]
    found = check_manifest(model, rel)
    runners: list[tuple[str, int]] = []
    leads: set[str] = set()
    for module in sorted(model.glob("*.py")):
        if module.name.startswith("prepare_"):
            continue
        module_rel = f"{rel}/{module.name}"
        try:
            source = module.read_text()
        except UnicodeDecodeError:
            found.append(Violation(module_rel, 0, "is not UTF-8"))
            continue
        scan = scan_module(source, module_rel)
        found.extend(scan.violations)
        runners.extend(scan.git_runners)
        leads |= scan.network_leads
    # A helper runs ["git", *args] and a caller elsewhere passes "clone" first.
    if runners and leads:
        module_rel, line = runners[0]
        found.append(
            Violation(
                module_rel,
                line,
                f"runs `git` with {', '.join(sorted(leads))} at runtime; fetch the source in the build",
            )
        )
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
        p
        for p in (root / "models").iterdir()
        if p.is_dir() and not p.name.startswith(".")
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
