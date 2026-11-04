"""Prepare the pinned StreamForce source, the Wan2.2 base snapshot, and the checkpoint."""

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
_WAN_PATTERNS = ("*.json", "*.pth", "*.safetensors", "google/umt5-xxl/*")
_WAN_REQUIRED = (
    "config.json",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model.safetensors.index.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
)


@dataclass(frozen=True)
class StreamForceConfig:
    """Validated locations of everything the model half loads."""

    config_path: Path
    source_path: Path
    source_url: str
    source_revision: str
    wan_root: Path
    wan_repo_id: str
    wan_revision: str
    checkpoint_path: Path
    checkpoint_bytes: int
    checkpoint_url: str


def read_config(config_path: Path | None) -> StreamForceConfig:
    """Read the model YAML and resolve its paths under Runtime's weights root."""
    if config_path is None:
        raise ValueError("StreamForce requires runtime.config in reactor.yaml")
    document = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), str(config_path))
    source = _mapping(document.get("source"), "source")
    wan = _mapping(document.get("wan"), "wan")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    root = get_weights_path()
    return StreamForceConfig(
        config_path=config_path,
        source_path=(root / str(source.get("path", "StreamForce"))).resolve(),
        source_url=_https(source.get("url"), "source.url"),
        source_revision=_revision(source.get("revision"), "source.revision"),
        wan_root=root.resolve(),
        wan_repo_id=str(wan.get("repo_id", "")),
        wan_revision=_revision(wan.get("revision"), "wan.revision"),
        checkpoint_path=(root / str(checkpoint.get("path", "model.pt"))).resolve(),
        checkpoint_bytes=int(checkpoint.get("bytes", 0)),
        checkpoint_url=_https(checkpoint.get("download_url"), "checkpoint.download_url"),
    )


def prepare_assets(config: StreamForceConfig) -> None:
    """Fetch what is public, and refuse to start without the checkpoint."""
    _ensure_checkpoint(config)
    _ensure_source(config)
    _ensure_wan(config)


def _ensure_checkpoint(config: StreamForceConfig) -> None:
    """The checkpoint is a manual download; check it is there and is the tested file."""
    path = config.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(
            f"StreamForce checkpoint not found at {path}. Download it from "
            f"{config.checkpoint_url} and save it there (see the README)."
        )
    size = path.stat().st_size
    if size != config.checkpoint_bytes:
        raise RuntimeError(
            f"{path} is {size} bytes; the tested StreamForce v1 checkpoint is "
            f"{config.checkpoint_bytes} bytes. Download it again from {config.checkpoint_url}."
        )


def _ensure_source(config: StreamForceConfig) -> None:
    """Clone the pinned upstream revision once and apply streamforce.patch exactly once."""
    source = config.source_path
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading StreamForce source", url=config.source_url, revision=config.source_revision)
        with tempfile.TemporaryDirectory(prefix=".reactor-streamforce-", dir=source.parent) as temporary:
            checkout = Path(temporary) / "checkout"
            _git(["clone", "--filter=blob:none", "--no-checkout", config.source_url, str(checkout)])
            _git(["-C", str(checkout), "checkout", "--detach", config.source_revision])
            with suppress(FileExistsError):
                checkout.rename(source)
    actual = _git(["-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    if actual != config.source_revision:
        raise RuntimeError(f"StreamForce source at {source} is {actual}; expected {config.source_revision}")
    patch = Path(__file__).with_name("streamforce.patch")
    if _git(["-C", str(source), "apply", "--reverse", "--check", str(patch)], check=False).returncode == 0:
        return
    forward = _git(["-C", str(source), "apply", "--check", str(patch)], check=False)
    if forward.returncode != 0:
        raise RuntimeError(f"StreamForce source does not take streamforce.patch: {forward.stderr.strip()}")
    logger.info("applying streamforce.patch", source=str(source))
    _git(["-C", str(source), "apply", str(patch)])


def _ensure_wan(config: StreamForceConfig) -> None:
    """Download the pinned Wan2.2-TI2V-5B snapshot (text encoder, VAE, base DiT) once."""
    target = config.wan_root / "Wan2.2-TI2V-5B"
    marker = target / SNAPSHOT_MARKER
    expected = {"repo_id": config.wan_repo_id, "revision": config.wan_revision}
    if _json_matches(marker, expected) and all((target / f).is_file() for f in _WAN_REQUIRED):
        return
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading Wan2.2 base snapshot", repo_id=config.wan_repo_id, revision=config.wan_revision)
    command = [sys.executable, str(Path(__file__).with_name("download_snapshot.py")),
               "--repo-id", config.wan_repo_id, "--revision", config.wan_revision, "--local-dir", str(target)]
    for pattern in _WAN_PATTERNS:
        command.extend(["--allow-pattern", pattern])
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "0"
    subprocess.run(command, check=True, env=environment)
    missing = [f for f in _WAN_REQUIRED if not (target / f).is_file()]
    if missing:
        raise RuntimeError(f"{config.wan_repo_id}@{config.wan_revision} is missing {missing}")
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
