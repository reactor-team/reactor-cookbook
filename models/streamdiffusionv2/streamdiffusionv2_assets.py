"""Prepare the pinned StreamDiffusionV2 source, the Wan2.1 base snapshot, and the checkpoint."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SNAPSHOT_MARKER = ".reactor-snapshot.json"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_WAN_PATTERNS = ("config.json", "*.pth", "*.safetensors", "google/umt5-xxl/*", "LICENSE.txt")
_WAN_REQUIRED = (
    "config.json",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model.safetensors",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
)


@dataclass(frozen=True)
class Snapshot:
    """One pinned public Hugging Face snapshot and where it lands under the weights root."""

    repo_id: str
    revision: str
    target: Path


@dataclass(frozen=True)
class StreamDiffusionV2Config:
    """Validated locations of everything the model half loads."""

    config_path: Path
    weights_root: Path
    source_path: Path
    source_url: str
    source_revision: str
    wan: Snapshot
    checkpoint: Snapshot
    checkpoint_subdir: str
    checkpoint_bytes: int


def read_config(config_path: Path | None) -> StreamDiffusionV2Config:
    """Read the model YAML and resolve its paths under Runtime's weights root."""
    if config_path is None:
        raise ValueError("StreamDiffusionV2 requires runtime.config in reactor.yaml")
    document = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), str(config_path))
    source = _mapping(document.get("source"), "source")
    wan = _mapping(document.get("wan"), "wan")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    root = get_weights_path().resolve()
    subdir = str(checkpoint.get("path", "ckpts/wan_causal_dmd_v2v"))
    return StreamDiffusionV2Config(
        config_path=config_path,
        weights_root=root,
        source_path=(root / str(source.get("path", "StreamDiffusionV2"))).resolve(),
        source_url=_https(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        wan=Snapshot(str(wan.get("repo_id", "")), _revision(wan.get("revision"), "wan.revision"),
                     (root / str(wan.get("path", "wan_models/Wan2.1-T2V-1.3B"))).resolve()),
        # The Hub repo keeps the student under wan_causal_dmd_v2v/; it downloads into ckpts/.
        checkpoint=Snapshot(str(checkpoint.get("repo_id", "")),
                            _revision(checkpoint.get("revision"), "checkpoint.revision"),
                            (root / subdir).parent.resolve()),
        checkpoint_subdir=subdir,
        checkpoint_bytes=int(checkpoint.get("bytes", 0)),
    )


def prepare_assets(config: StreamDiffusionV2Config) -> None:
    """Fetch the pinned source and both public snapshots once; check the checkpoint is the tested file."""
    _ensure_source(config)
    _ensure_snapshot(config.wan, _WAN_PATTERNS, _WAN_REQUIRED)
    model_pt = f"{Path(config.checkpoint_subdir).name}/model.pt"
    _ensure_snapshot(config.checkpoint, (f"{Path(config.checkpoint_subdir).name}/*",), (model_pt,))
    path = config.weights_root / config.checkpoint_subdir / "model.pt"
    size = path.stat().st_size
    if size != config.checkpoint_bytes:
        raise RuntimeError(
            f"{path} is {size} bytes; the tested StreamDiffusionV2 1.3B checkpoint is "
            f"{config.checkpoint_bytes} bytes. Delete it to download {config.checkpoint.repo_id} again."
        )


def _ensure_source(config: StreamDiffusionV2Config) -> None:
    """Clone the pinned upstream revision once."""
    source = config.source_path
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading StreamDiffusionV2 source", url=config.source_url, revision=config.source_revision)
        with tempfile.TemporaryDirectory(prefix=".reactor-streamdiffusionv2-", dir=source.parent) as temporary:
            checkout = Path(temporary) / "checkout"
            _git(["clone", "--filter=blob:none", "--no-checkout", config.source_url, str(checkout)])
            _git(["-C", str(checkout), "checkout", "--detach", config.source_revision])
            with suppress(FileExistsError):
                checkout.rename(source)
    actual = _git(["-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(f"StreamDiffusionV2 source at {source} is {actual}; expected {config.source_revision}")


def _ensure_snapshot(snapshot: Snapshot, patterns: tuple[str, ...], required: tuple[str, ...]) -> None:
    """Download one pinned snapshot once, then check the files the model loads are there."""
    target = snapshot.target
    marker = target / f".{snapshot.repo_id.replace('/', '--')}{SNAPSHOT_MARKER}"
    expected = {"repo_id": snapshot.repo_id, "revision": snapshot.revision}
    if _json_matches(marker, expected) and all((target / f).is_file() for f in required):
        return
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading snapshot", repo_id=snapshot.repo_id, revision=snapshot.revision)
    command = [sys.executable, str(Path(__file__).with_name("download_snapshot.py")),
               "--repo-id", snapshot.repo_id, "--revision", snapshot.revision, "--local-dir", str(target)]
    for pattern in patterns:
        command.extend(["--allow-pattern", pattern])
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "0"
    subprocess.run(command, check=True, env=environment)
    missing = [f for f in required if not (target / f).is_file()]
    if missing:
        raise RuntimeError(f"{snapshot.repo_id}@{snapshot.revision} is missing {missing}")
    pending = marker.with_suffix(".tmp")
    pending.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    os.replace(pending, marker)


def _git(arguments: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git on an explicit checkout the runtime mounted."""
    command = ["git", *arguments]
    if "-C" in arguments:
        checkout = Path(arguments[arguments.index("-C") + 1]).resolve()
        command = ["git", "-c", f"safe.directory={checkout}", *arguments]
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _json_matches(path: Path, expected: dict[str, str]) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _revision(value: object, name: str) -> str:
    revision = str(value or "")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full 40-character revision")
    return revision


def _https(value: object, name: str) -> str:
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url
