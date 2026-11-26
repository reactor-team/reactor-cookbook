"""Configuration, the pinned upstream source, and the weights folder for Cosmos3-Policy-DROID.

The model half reads ``cosmos3_policy_droid.yaml`` through :func:`read_config`,
checks the `cosmos-framework` checkout the image build made with
:func:`check_source`, puts it on ``sys.path`` with :func:`activate_source`, and
resolves the policy checkpoint and the video tokenizer under the weights folder
with :func:`check_weights`. Nothing here downloads: ``prepare_weights.py``
builds the weights folder.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCE_ENV = "COSMOS_FRAMEWORK_PATH"
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
FIX = f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Repository:
    """Describe the public Git repository the image build fetches at an immutable commit.

    Attributes:
        path: Where the image holds the checkout.
        url: The repository the build fetches.
        revision: The full commit hash the checkout must be at.
    """

    path: Path
    url: str
    revision: str


@dataclass(frozen=True)
class HubFiles:
    """Describe one Hugging Face repository pinned to an immutable commit.

    Attributes:
        repo_id: The repository id, ``<org>/<name>``. The weights folder holds
            its files under ``<repo_id>/``.
        revision: The full commit hash of the files.
    """

    repo_id: str
    revision: str


@dataclass(frozen=True)
class PolicyConfig:
    """Describe which checkpoint to serve and how to sample it.

    Attributes:
        source: The pinned upstream `cosmos-framework` source.
        checkpoint: The DROID policy checkpoint in the weights folder.
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
    checkpoint: HubFiles
    format_prompt_as_json: bool | None
    guidance_interval: tuple[int, int] | None
    guidance: float
    num_steps: int
    resolution: str
    action_chunk_size: int
    device_id: int
    warmup: bool


DEFAULT_SOURCE = Repository(
    path=Path("/opt/cosmos-framework"),
    url="https://github.com/NVIDIA/cosmos-framework.git",
    revision="cf5d68c00d97ccd2480a2320ed652b92dec63102",
)
DEFAULT_CHECKPOINT = HubFiles(
    repo_id="nvidia/Cosmos3-Edge-Policy-DROID",
    revision="a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14",
)

# The Wan 2.2 video tokenizer both DROID checkpoints name by its registry path
# (`pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth`); the pinned source's
# checkpoint registry maps that path to this file at this revision.
VAE = HubFiles(repo_id="Wan-AI/Wan2.2-TI2V-5B", revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e")
VAE_FILENAME = "Wan2.2_VAE.pth"

# The checkpoint repository files serving does not read: the model card, its
# media, and the conversion script. `vae/`, a diffusers copy of the video
# tokenizer, stays in the folder although the native loader reads
# `Wan2.2_VAE.pth`; it is 1.4 GB.
CHECKPOINT_IGNORE_PATTERNS = ("*.md", "*.png", "*.mp4", "*.py", ".gitattributes")
# Files every DROID checkpoint carries: the model config, the processor the
# serving path builds from the checkpoint, and the two weight indexes. The
# check also requires every shard those indexes name.
CHECKPOINT_FILES = (
    "config.json",
    "model_index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)
CHECKPOINT_INDEXES = (
    "model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
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
        checkpoint=_checkpoint(raw.get("checkpoint")),
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
    return Repository(
        path=Path(str(raw.get("path", DEFAULT_SOURCE.path))),
        url=str(raw.get("url", DEFAULT_SOURCE.url)),
        revision=_revision(raw.get("revision", DEFAULT_SOURCE.revision), "source.revision"),
    )


def _checkpoint(raw: object) -> HubFiles:
    if raw is None:
        return DEFAULT_CHECKPOINT
    if not isinstance(raw, dict) or "repo_id" not in raw or "revision" not in raw:
        raise ValueError("checkpoint must be a mapping with repo_id and revision")
    repo_id = str(raw["repo_id"])
    if repo_id.count("/") != 1:
        raise ValueError(f"checkpoint.repo_id must be <org>/<name>, got {repo_id!r}")
    return HubFiles(repo_id=repo_id, revision=_revision(raw["revision"], "checkpoint.revision"))


def _revision(raw: object, key: str) -> str:
    revision = str(raw)
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{key} must be a full 40-hex commit hash, got {revision!r}")
    return revision


def resolve_source_path(source: Repository) -> Path:
    """Return the checkout directory: ``COSMOS_FRAMEWORK_PATH`` if set, else ``source.path``."""
    override = os.environ.get(SOURCE_ENV)
    return Path(override) if override else source.path


def check_source(source: Repository, path: Path) -> None:
    """Require the checkout the image build made, at the pinned revision.

    Raises:
        FileNotFoundError: The checkout is absent; rebuilding the image makes it.
        RuntimeError: The checkout is at another revision.
    """
    if not (path / ".git").exists() or not (path / "cosmos_framework" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"cosmos-framework source not found at {path}; rebuild the image with `reactor build`"
        )
    actual = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual != source.revision:
        raise RuntimeError(
            f"cosmos-framework source revision is {actual}; expected {source.revision}. "
            "Rebuild the image with `reactor build`."
        )


def activate_source(path: Path) -> None:
    """Put the checkout first on ``sys.path`` so ``import cosmos_framework`` resolves to it."""
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def checkpoint_dir(weights_root: Path, checkpoint: HubFiles) -> Path:
    """Return the folder that holds ``checkpoint``'s files in the weights folder."""
    return Path(weights_root) / checkpoint.repo_id


def vae_path(weights_root: Path) -> Path:
    """Return the video tokenizer file in the weights folder."""
    return Path(weights_root) / VAE.repo_id / VAE_FILENAME


def check_weights(config: PolicyConfig, weights_root: Path) -> tuple[Path, Path]:
    """Return the absolute checkpoint folder and video tokenizer file; never download.

    Raises:
        FileNotFoundError: A file is absent. The message names the first
            missing path and the command that builds the folder.
    """
    root = Path(weights_root).expanduser().resolve()
    checkpoint = checkpoint_dir(root, config.checkpoint)
    expected = [checkpoint / name for name in (*CHECKPOINT_FILES, *CHECKPOINT_INDEXES)]
    for index in CHECKPOINT_INDEXES:
        path = checkpoint / index
        if path.is_file():
            shards = set(json.loads(path.read_text())["weight_map"].values())
            expected += sorted(path.parent / shard for shard in shards)
    vae = vae_path(root)
    expected.append(vae)
    missing = [path for path in dict.fromkeys(expected) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cosmos3-Policy-DROID weights are incomplete in {root}: {missing[0]} is missing "
            f"({len(missing)} files). {FIX}"
        )
    return checkpoint, vae
