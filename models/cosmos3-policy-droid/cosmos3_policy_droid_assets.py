"""Configuration, upstream source, and checkpoint resolution for Cosmos3-Policy-DROID.

The model half reads ``cosmos3_policy_droid.yaml`` through :func:`read_config`,
clones the pinned public `cosmos-framework` source into the weights root with
:func:`ensure_source_checkout` and puts it on ``sys.path`` with
:func:`activate_source`, and routes every Hugging Face lookup that source
makes through :func:`route_checkpoint_downloads`, so the policy checkpoint and
the video tokenizer it depends on land beside it.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SOURCE_ENV = "COSMOS_FRAMEWORK_PATH"
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Repository:
    """Describe one public Git repository pinned to an immutable commit.

    Attributes:
        path: Where the checkout lives; relative paths resolve under the
            weights root the model half receives.
        url: The repository to clone.
        revision: The full commit hash the checkout must be at.
    """

    path: Path
    url: str
    revision: str


@dataclass(frozen=True)
class PolicyConfig:
    """Describe which checkpoint to serve and how to sample it.

    Attributes:
        source: The pinned upstream `cosmos-framework` source.
        checkpoint: Hugging Face repository id of a DROID policy checkpoint,
            or a local directory holding one.
        hf_revision: Revision of the repository to download.
        format_prompt_as_json: Serve the task as the structured JSON prompt the
            checkpoint was trained on. ``None`` keeps the checkpoint's default.
        guidance_interval: Inclusive denoising-timestep range that applies
            classifier-free guidance, or ``None`` to apply it at every step.
        guidance: Classifier-free guidance scale.
        num_steps: Denoising steps per prediction.
        resolution: Action transform resolution tier.
        action_chunk_size: Number of action steps in a chunk.
        device_id: CUDA device the policy loads on.
        warmup: Run two throwaway predictions at load so compilation happens
            before the first session.
    """

    source: Repository
    checkpoint: str
    hf_revision: str
    format_prompt_as_json: bool | None
    guidance_interval: tuple[int, int] | None
    guidance: float
    num_steps: int
    resolution: str
    action_chunk_size: int
    device_id: int
    warmup: bool


DEFAULT_SOURCE = Repository(
    path=Path("source/cosmos-framework"),
    url="https://github.com/NVIDIA/cosmos-framework.git",
    revision="cf5d68c00d97ccd2480a2320ed652b92dec63102",
)


def read_config(config_path: Path | None) -> PolicyConfig:
    """Parse the model's YAML config, applying the Edge checkpoint's defaults."""
    raw: dict = {}
    if config_path is not None:
        raw = yaml.safe_load(config_path.read_text()) or {}
    interval = raw.get("guidance_interval")
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("guidance_interval must be a pair [low, high]")
        interval = (int(interval[0]), int(interval[1]))
    fmt = raw.get("format_prompt_as_json")
    return PolicyConfig(
        source=_source(raw.get("source")),
        checkpoint=str(raw.get("checkpoint", "nvidia/Cosmos3-Edge-Policy-DROID")),
        hf_revision=str(raw.get("hf_revision", "main")),
        format_prompt_as_json=None if fmt is None else bool(fmt),
        guidance_interval=interval,
        guidance=float(raw.get("guidance", 3.0)),
        num_steps=int(raw.get("num_steps", 4)),
        resolution=str(raw.get("resolution", "480")),
        action_chunk_size=int(raw.get("action_chunk_size", 32)),
        device_id=int(raw.get("device_id", 0)),
        warmup=bool(raw.get("warmup", True)),
    )


def _source(raw: object) -> Repository:
    if raw is None:
        return DEFAULT_SOURCE
    if not isinstance(raw, dict):
        raise ValueError("source must be a mapping with path, url, and revision")
    revision = str(raw.get("revision", DEFAULT_SOURCE.revision))
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"source.revision must be a full 40-hex commit hash, got {revision!r}")
    return Repository(
        path=Path(str(raw.get("path", DEFAULT_SOURCE.path))),
        url=str(raw.get("url", DEFAULT_SOURCE.url)),
        revision=revision,
    )


def resolve_source_path(source: Repository, weights_root: Path) -> Path:
    """Return the checkout directory: ``COSMOS_FRAMEWORK_PATH`` if set, else under the weights root."""
    override = os.environ.get(SOURCE_ENV)
    if override:
        return Path(override)
    return source.path if source.path.is_absolute() else weights_root / source.path


def ensure_source_checkout(source: Repository, path: Path) -> None:
    """Clone the pinned source if absent and require it to be at the pinned revision, unmodified.

    The clone is a blobless partial clone checked out detached at the commit,
    made in a temporary sibling directory and renamed into place, so a failed
    download never leaves a half-populated checkout behind.
    """
    if not path.exists():
        logger.info(
            "downloading cosmos-framework source: url=%s revision=%s -> %s", source.url, source.revision, path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cosmos-framework-source-", dir=path.parent) as temporary:
            checkout = Path(temporary) / "checkout"
            _run(["git", "clone", "--filter=blob:none", "--no-checkout", source.url, str(checkout)])
            _run(["git", "-C", str(checkout), "checkout", "--detach", source.revision])
            checkout.rename(path)
    if not (path / ".git").exists():
        raise RuntimeError(f"cosmos-framework source must be a Git checkout: {path}")
    git = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
    actual = _run([*git, "rev-parse", "HEAD"]).stdout.strip()
    if actual != source.revision:
        raise RuntimeError(f"cosmos-framework source revision is {actual}; expected {source.revision}")
    if _run([*git, "status", "--porcelain"]).stdout.strip():
        raise RuntimeError(f"cosmos-framework source checkout has local changes: {path}")
    if not (path / "cosmos_framework" / "__init__.py").is_file():
        raise RuntimeError(f"cosmos-framework checkout has no cosmos_framework package: {path}")


def activate_source(path: Path) -> None:
    """Put the checkout first on ``sys.path`` so ``import cosmos_framework`` resolves to it."""
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def route_checkpoint_downloads(cache_dir: Path) -> None:
    """Resolve the framework's Hugging Face lookups through ``huggingface_hub``.

    Upstream's ``checkpoint_db`` downloads by shelling out to a separate ``uv``
    project. Both of its Hugging Face resolvers are replaced with
    ``huggingface_hub`` calls that cache under ``cache_dir``, so a checkpoint
    already present is reused and one that is absent is fetched. Idempotent:
    a second call leaves the first routing in place.
    """
    import huggingface_hub as hub
    from cosmos_framework.utils import checkpoint_db as db

    if getattr(db.CheckpointDirHf._download, "_routed", False):
        return
    cache = str(cache_dir)

    def dir_download(self) -> str:
        include = list(self.include) or None
        exclude = list(self.exclude) or None
        if self.subdirectory:
            include = [os.path.join(self.subdirectory, p) for p in (include or ["*"])]
        path = hub.snapshot_download(
            self.repository,
            revision=self.revision,
            repo_type=self.repository_type.value,
            allow_patterns=include,
            ignore_patterns=exclude,
            cache_dir=cache,
        )
        return os.path.join(path, self.subdirectory) if self.subdirectory else path

    def file_download(self) -> str:
        return hub.hf_hub_download(
            self.repository,
            self.filename,
            revision=self.revision,
            repo_type=self.repository_type.value,
            cache_dir=cache,
        )

    dir_download._routed = True  # type: ignore[attr-defined]
    file_download._routed = True  # type: ignore[attr-defined]
    db.CheckpointDirHf._download = dir_download
    db.CheckpointFileHf._download = file_download


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bootstrap command and preserve its full failure output."""
    return subprocess.run(command, check=True, text=True, capture_output=True)
