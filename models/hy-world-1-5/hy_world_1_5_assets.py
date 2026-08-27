"""Prepare pinned public HY-World 1.5 sources, weights, and example images."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

SOURCE_ENV = "HY_WORLDPLAY_PATH"
BASE_MODEL_ENV = "HY_WORLDPLAY_BASE_MODEL_PATH"
ACTION_MODEL_ENV = "HY_WORLDPLAY_ACTION_MODEL_PATH"
VISION_MODEL_ENV = "HY_WORLDPLAY_VISION_ENCODER_PATH"

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
    flux_vision: HubAsset
    public_vision_fallback: HubAsset
    cache_path: Path
    seed: int
    inference_steps: int
    memory_frames: int
    temporal_context_size: int
    stabilization_level: int
    points_in_sphere: int
    max_chunks: int


def read_config(config_path: Path | None) -> HYWorld15Config:
    """Read and validate the HY-World 1.5 adapter YAML."""
    if config_path is None:
        raise ValueError("HY-World 1.5 requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    weights = get_weights_path()
    source_raw = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    memory = _mapping(document.get("memory"), "memory")
    stream = _mapping(document.get("stream"), "stream")

    source_path = Path(
        os.environ.get(SOURCE_ENV, weights / str(source_raw["path"]))
    ).expanduser()
    base_raw = _mapping(assets.get("base_model"), "assets.base_model")
    action_raw = _mapping(assets.get("action_model"), "assets.action_model")
    base_path = Path(
        os.environ.get(BASE_MODEL_ENV, weights / str(base_raw["path"]))
    ).expanduser()
    action_path = Path(
        os.environ.get(ACTION_MODEL_ENV, weights / str(action_raw["path"]))
    ).expanduser()
    vision_raw = _mapping(assets["flux_vision"], "assets.flux_vision")
    vision_path = Path(
        os.environ.get(VISION_MODEL_ENV, weights / str(vision_raw["path"]))
    ).expanduser()

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
        qwen=_asset(
            weights / str(_mapping(assets["qwen"], "assets.qwen")["path"]),
            assets["qwen"],
            "assets.qwen",
        ),
        byt5=_asset(
            weights / str(_mapping(assets["byt5"], "assets.byt5")["path"]),
            assets["byt5"],
            "assets.byt5",
        ),
        glyph=_asset(
            weights / str(_mapping(assets["glyph"], "assets.glyph")["path"]),
            assets["glyph"],
            "assets.glyph",
        ),
        flux_vision=_asset(vision_path, vision_raw, "assets.flux_vision"),
        public_vision_fallback=_asset(
            weights
            / str(
                _mapping(
                    assets["public_vision_fallback"], "assets.public_vision_fallback"
                )["path"]
            ),
            assets["public_vision_fallback"],
            "assets.public_vision_fallback",
        ),
        cache_path=weights / str(assets.get("cache", "cache")),
        seed=int(inference.get("seed", 1)),
        inference_steps=inference_steps,
        memory_frames=memory_frames,
        temporal_context_size=temporal_context_size,
        stabilization_level=stabilization_level,
        points_in_sphere=points_in_sphere,
        max_chunks=max_chunks,
    )


def prepare_runtime_assets(config: HYWorld15Config) -> None:
    """Download every missing public source and model asset into the weights root."""
    config.cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(config.cache_path / "huggingface"))
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE", str(config.cache_path / "huggingface/hub")
    )
    os.environ.setdefault("MODELSCOPE_CACHE", str(config.cache_path / "modelscope"))
    _ensure_git_checkout(config.source)
    _ensure_hf_snapshot(
        config.base_model,
        allow_patterns=("transformer/480p_i2v/**", "vae/**", "scheduler/**"),
    )
    _ensure_hf_file(
        config.action_model,
        filename="ar_distilled_action_model/model.safetensors",
    )
    _ensure_hf_snapshot(config.qwen)
    _ensure_hf_snapshot(config.byt5)
    _ensure_modelscope_snapshot(config.glyph)
    _ensure_vision_encoder(config)
    validate_runtime_paths(config)


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


def validate_runtime_paths(config: HYWorld15Config) -> None:
    """Require the complete source, checkpoint, encoder, and example layout."""
    required = {
        "HY-World source": config.source.path
        / "hyvideo/pipelines/worldplay_video_pipeline.py",
        "distilled action checkpoint": config.action_model.path
        / "ar_distilled_action_model/model.safetensors",
        "480p transformer": config.base_model.path / "transformer/480p_i2v/config.json",
        "VAE": config.base_model.path / "vae/config.json",
        "scheduler": config.base_model.path / "scheduler/scheduler_config.json",
        "Qwen text encoder": config.qwen.path / "config.json",
        "ByT5 encoder": config.byt5.path / "config.json",
        "Glyph encoder": config.glyph.path / "checkpoints/byt5_model.pt",
        "SigLIP image encoder": config.flux_vision.path / "image_encoder/config.json",
        "SigLIP image processor": config.flux_vision.path
        / "feature_extractor/preprocessor_config.json",
        "example table": config.source.path / "assets/test_case.csv",
    }
    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing HY-World runtime assets:\n" + "\n".join(missing)
        )
    load_examples(config)


def assemble_base_model(config: HYWorld15Config) -> None:
    """Link separately pinned encoders into the directory layout the pipeline expects."""
    links = {
        config.base_model.path / "text_encoder/llm": config.qwen.path,
        config.base_model.path / "text_encoder/byt5-small": config.byt5.path,
        config.base_model.path / "text_encoder/Glyph-SDXL-v2": config.glyph.path,
        config.base_model.path / "vision_encoder/siglip": config.flux_vision.path,
    }
    for destination, target in links.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_target = Path(os.path.relpath(target, destination.parent))
        if destination.is_symlink():
            if (
                Path(os.readlink(destination)) == relative_target
                and destination.resolve() == target.resolve()
            ):
                continue
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(
                f"Model layout path already exists and is not the expected link: {destination}"
            )
        destination.symlink_to(relative_target, target_is_directory=True)


def _ensure_git_checkout(repository: Repository) -> None:
    """Clone a missing source tree and require its configured immutable revision."""
    if not repository.path.exists():
        logger.info(
            "downloading source checkout",
            url=repository.url,
            destination=str(repository.path),
        )
        repository.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hy-world-source-", dir=repository.path.parent
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            _run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    repository.url,
                    str(checkout),
                ]
            )
            _run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "checkout",
                    "--detach",
                    repository.revision,
                ]
            )
            checkout.rename(repository.path)
    if not (repository.path / ".git").exists():
        raise RuntimeError(f"HY-World source must be a Git checkout: {repository.path}")
    git = [
        "git",
        "-c",
        f"safe.directory={repository.path}",
        "-C",
        str(repository.path),
    ]
    actual = _run([*git, "rev-parse", "HEAD"]).stdout.strip()
    if actual != repository.revision:
        raise RuntimeError(
            f"HY-World source revision is {actual}; expected {repository.revision}"
        )
    dirty = _run([*git, "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise RuntimeError(
            f"HY-World source checkout has local changes: {repository.path}"
        )


def _ensure_hf_snapshot(
    asset: HubAsset, allow_patterns: tuple[str, ...] | None = None
) -> None:
    """Download a pinned Hugging Face snapshot into its configured path."""
    marker = asset.path / ".reactor-revision"
    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == asset.revision
    ):
        return
    from huggingface_hub import snapshot_download

    asset.path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        local_dir=asset.path,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    marker.write_text(asset.revision + "\n", encoding="utf-8")


def _ensure_hf_file(asset: HubAsset, *, filename: str) -> None:
    """Download one pinned Hugging Face file without duplicating it."""
    destination = asset.path / filename
    if destination.is_file():
        return
    from huggingface_hub import hf_hub_download

    asset.path.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        filename=filename,
        local_dir=asset.path,
    )


def _ensure_modelscope_snapshot(asset: HubAsset) -> None:
    """Download the pinned public Glyph snapshot with ModelScope."""
    marker = asset.path / ".reactor-revision"
    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == asset.revision
    ):
        return
    from modelscope.hub.snapshot_download import snapshot_download

    asset.path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        asset.repo_id,
        revision=asset.revision,
        local_dir=str(asset.path),
        allow_patterns=[
            "assets/color_idx.json",
            "assets/multilingual_10-lang_idx.json",
            "checkpoints/byt5_model.pt",
        ],
    )
    marker.write_text(asset.revision + "\n", encoding="utf-8")


def _ensure_vision_encoder(config: HYWorld15Config) -> None:
    """Prepare the official gated SigLIP encoder or its public architecture match."""
    if os.environ.get(VISION_MODEL_ENV):
        # read_config resolved the override into config.flux_vision.path already;
        # an explicit path must be a complete layout, so downloads never run.
        if not (config.flux_vision.path / "image_encoder/config.json").is_file():
            raise FileNotFoundError(
                f"{VISION_MODEL_ENV} is not a HY-World SigLIP layout: "
                f"{config.flux_vision.path}"
            )
        return
    if (config.flux_vision.path / "image_encoder/config.json").is_file():
        return
    try:
        _ensure_hf_snapshot(
            config.flux_vision,
            allow_patterns=("image_encoder/**", "feature_extractor/**"),
        )
        return
    except Exception as error:  # noqa: BLE001 - every gated-asset failure selects the fallback
        logger.warning(
            "official gated vision encoder unavailable; preparing the public "
            "SigLIP SO400M architecture match",
            error=str(error),
        )

    _ensure_hf_snapshot(config.public_vision_fallback)
    from transformers import SiglipImageProcessor, SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(config.public_vision_fallback.path)
    processor = SiglipImageProcessor.from_pretrained(config.public_vision_fallback.path)
    image_encoder = config.flux_vision.path / "image_encoder"
    feature_extractor = config.flux_vision.path / "feature_extractor"
    image_encoder.mkdir(parents=True, exist_ok=True)
    feature_extractor.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(image_encoder)
    processor.save_pretrained(feature_extractor)
    del model


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bootstrap command and preserve its full failure output."""
    return subprocess.run(command, check=True, text=True, capture_output=True)
