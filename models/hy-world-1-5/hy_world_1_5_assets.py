"""Read the HY-World 1.5 config and check its source, weights, and example images.

Nothing here downloads. The image holds the upstream source at the pinned
revision, and ``prepare_weights.py`` builds the weights folder.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

SOURCE_ENV = "HY_WORLDPLAY_PATH"
PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
FIX = f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
# Writable state of the model process. The weights folder is read-only.
CACHE_ROOT = Path("~/.cache/hy-world-1-5")

# The files of each pinned repository that the upstream pipeline reads, at
# their repository paths. The repositories also hold other checkpoints and
# framework copies, which the weights folder skips.
BASE_MODEL_FILES = (
    "scheduler/scheduler_config.json",
    "transformer/480p_i2v/config.json",
    "transformer/480p_i2v/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
ACTION_MODEL_FILES = ("ar_distilled_action_model/model.safetensors",)
QWEN_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    *(f"model-0000{index}-of-00005.safetensors" for index in range(1, 6)),
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
BYT5_FILES = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
GLYPH_FILES = (
    "assets/color_idx.json",
    "assets/multilingual_10-lang_idx.json",
    "checkpoints/byt5_model.pt",
)
VISION_ENCODER_FILES = ("config.json", "model.safetensors")
VISION_PROCESSOR_FILES = ("preprocessor_config.json",)

# Where the upstream pipeline reads each encoder, relative to the base model.
_QWEN_DIR = "text_encoder/llm"
_BYT5_DIR = "text_encoder/byt5-small"
_GLYPH_DIR = "text_encoder/Glyph-SDXL-v2"
_VISION_DIR = "vision_encoder/siglip"

_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Repository:
    """Describe one public Git repository pinned to an immutable commit."""

    path: Path
    url: str
    revision: str


@dataclass(frozen=True)
class HubAsset:
    """Describe one public model repository pinned to an immutable revision."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class HubFile:
    """One file of a pinned repository and the folder it is written into."""

    repo_id: str
    revision: str
    filename: str
    local_dir: Path

    @property
    def path(self) -> Path:
        """Return the file's path in the weights folder."""
        return self.local_dir / self.filename


@dataclass(frozen=True)
class ExampleImage:
    """Pair one built-in reference image with its official scene prompt."""

    path: Path
    prompt: str


@dataclass(frozen=True)
class HYWorld15Config:
    """Hold validated source, model, rollout, and interaction settings."""

    source: Repository
    base_model: HubAsset
    action_model: HubAsset
    qwen: HubAsset
    byt5: HubAsset
    glyph: HubAsset
    vision_encoder: HubAsset
    weights_root: Path
    cache_path: Path
    seed: int
    inference_steps: int
    memory_frames: int
    temporal_context_size: int
    stabilization_level: int
    points_in_sphere: int
    max_chunks: int


def read_config(config_path: Path | None, weights_root: Path) -> HYWorld15Config:
    """Read and validate the HY-World 1.5 adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds.
    """
    if config_path is None:
        raise ValueError("HY-World 1.5 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    source_raw = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    memory = _mapping(document.get("memory"), "memory")
    stream = _mapping(document.get("stream"), "stream")

    source_path = Path(
        os.environ.get(SOURCE_ENV) or str(source_raw.get("path", "/opt/HY-WorldPlay"))
    ).expanduser()
    base_raw = _mapping(assets.get("base_model"), "assets.base_model")
    action_raw = _mapping(assets.get("action_model"), "assets.action_model")
    base_path = weights_root / _relative_path(
        base_raw.get("path"), "assets.base_model.path"
    )
    action_path = weights_root / _relative_path(
        action_raw.get("path"), "assets.action_model.path"
    )

    inference_steps = int(inference.get("steps", 4))
    if inference_steps != 4:
        raise ValueError("inference.steps must remain 4 for the distilled checkpoint")
    memory_frames = int(memory.get("frames", 20))
    temporal_context_size = int(memory.get("temporal_context_size", 12))
    stabilization_level = int(memory.get("stabilization_level", 15))
    points_in_sphere = int(memory.get("points_in_sphere", 50_000))
    if (
        memory_frames,
        temporal_context_size,
        stabilization_level,
        points_in_sphere,
    ) != (
        20,
        12,
        15,
        50_000,
    ):
        raise ValueError(
            "memory settings must retain the official distilled autoregressive defaults"
        )
    max_chunks = int(stream.get("max_chunks", 512))
    if max_chunks < 8:
        raise ValueError("stream.max_chunks must be at least 8")

    return HYWorld15Config(
        source=Repository(
            path=source_path,
            url=_url(source_raw.get("url"), "source.url"),
            revision=_revision(source_raw.get("revision"), "source.revision"),
        ),
        base_model=_asset(base_path, base_raw, "assets.base_model"),
        action_model=_asset(action_path, action_raw, "assets.action_model"),
        qwen=_asset(base_path / _QWEN_DIR, assets.get("qwen"), "assets.qwen"),
        byt5=_asset(base_path / _BYT5_DIR, assets.get("byt5"), "assets.byt5"),
        glyph=_asset(base_path / _GLYPH_DIR, assets.get("glyph"), "assets.glyph"),
        vision_encoder=_asset(
            base_path / _VISION_DIR,
            assets.get("vision_encoder"),
            "assets.vision_encoder",
        ),
        weights_root=weights_root,
        cache_path=CACHE_ROOT.expanduser(),
        seed=int(inference.get("seed", 1)),
        inference_steps=inference_steps,
        memory_frames=memory_frames,
        temporal_context_size=temporal_context_size,
        stabilization_level=stabilization_level,
        points_in_sphere=points_in_sphere,
        max_chunks=max_chunks,
    )


def huggingface_files(config: HYWorld15Config) -> tuple[HubFile, ...]:
    """Return every Hugging Face file of the weights folder, at its pinned revision."""
    vision = config.vision_encoder
    return (
        *_files(config.base_model, config.base_model.path, BASE_MODEL_FILES),
        *_files(config.action_model, config.action_model.path, ACTION_MODEL_FILES),
        *_files(config.qwen, config.qwen.path, QWEN_FILES),
        *_files(config.byt5, config.byt5.path, BYT5_FILES),
        *_files(vision, vision.path / "image_encoder", VISION_ENCODER_FILES),
        *_files(vision, vision.path / "feature_extractor", VISION_PROCESSOR_FILES),
    )


def modelscope_files(config: HYWorld15Config) -> tuple[HubFile, ...]:
    """Return every ModelScope file of the weights folder, at its pinned revision."""
    return _files(config.glyph, config.glyph.path, GLYPH_FILES)


def check_runtime_assets(config: HYWorld15Config) -> None:
    """Check the source in the image, the weights folder, and the example table."""
    check_source(config)
    check_weights(config)
    load_examples(config)


def check_source(config: HYWorld15Config) -> None:
    """Require the upstream checkout in the image at the pinned revision.

    The check reads the checkout's revision and runs no other ``git`` command.
    """
    source = config.source
    required = (
        ".git",
        "hyvideo/pipelines/worldplay_video_pipeline.py",
        "assets/test_case.csv",
    )
    missing = [name for name in required if not (source.path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"HY-World source at {source.path} lacks {', '.join(missing)}; "
            "rebuild the image with `reactor build`"
        )
    actual = _run_git(["-C", str(source.path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != source.revision:
        raise RuntimeError(
            f"HY-World source revision is {actual}; expected {source.revision}. "
            "Rebuild the image with `reactor build`."
        )


def check_weights(config: HYWorld15Config) -> None:
    """Raise ``FileNotFoundError`` naming the fix when a weights file is absent."""
    expected = [file.path for file in huggingface_files(config)]
    expected += [file.path for file in modelscope_files(config)]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"HY-World 1.5 weights are incomplete in {config.weights_root}: "
            f"{missing[0]} is missing ({len(missing)} files). {FIX}"
        )


def load_examples(config: HYWorld15Config) -> tuple[ExampleImage, ...]:
    """Return every usable image and caption from the official example table."""
    table = config.source.path / "assets/test_case.csv"
    examples: list[ExampleImage] = []
    with table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_name = (row.get("image_name") or "").strip()
            prompt = (row.get("caption") or "").strip()
            image_path = (config.source.path / image_name).resolve()
            if image_path.is_file() and prompt:
                examples.append(ExampleImage(image_path, prompt))
    if not examples:
        raise RuntimeError(f"No usable built-in examples found in {table}")
    return tuple(examples)


def _files(
    asset: HubAsset, local_dir: Path, filenames: tuple[str, ...]
) -> tuple[HubFile, ...]:
    """Pair each file name of one pinned repository with its destination folder."""
    return tuple(
        HubFile(asset.repo_id, asset.revision, filename, local_dir)
        for filename in filenames
    )


def _asset(path: Path, raw: object, name: str) -> HubAsset:
    """Build a pinned model asset from one YAML mapping."""
    value = _mapping(raw, name)
    return HubAsset(
        path=path,
        repo_id=_repo_id(value.get("repo_id"), f"{name}.repo_id"),
        revision=_revision(value.get("revision"), f"{name}.revision"),
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _relative_path(value: object, name: str) -> Path:
    """Return a path that stays inside the weights folder."""
    path = Path(str(value or ""))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path inside the weights folder")
    return path


def _revision(value: object, name: str) -> str:
    """Return one full immutable Git or model revision."""
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(f"{name} must be a 40-character hexadecimal revision")
    return value


def _repo_id(value: object, name: str) -> str:
    """Return one owner/name public model identifier."""
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError(f"{name} must be an owner/name repository identifier")
    return value


def _url(value: object, name: str) -> str:
    """Return one public HTTPS Git repository URL."""
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return value


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git on the image's checkout without an interactive credential prompt."""
    # The image's checkout can belong to another user than the runtime's.
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
