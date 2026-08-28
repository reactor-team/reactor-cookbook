"""Resolve Echo-WM source, model assets, configuration, and built-in scenes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from reactor_runtime import get_weights_path


@dataclass(frozen=True)
class ModelAsset:
    """Describe one Hugging Face asset pinned to an immutable revision."""

    path: Path
    repo_id: str
    revision: str
    filename: str | None = None


@dataclass(frozen=True)
class ExampleScene:
    """Describe one upstream image, prompt, field of view, and seed."""

    image: Path
    prompt: str
    fov_degrees: float
    seed: int


@dataclass(frozen=True)
class EchoWMConfig:
    """Hold validated source, asset, inference, and interaction settings."""

    source_path: Path
    source_url: str
    source_revision: str
    checkpoint: ModelAsset
    gemma: ModelAsset
    cache_dir: Path
    runtime_dir: Path
    width: int
    height: int
    fps: float
    seed: int
    default_upload_prompt: str
    timesteps: tuple[int, ...]
    video_local_attn_size: int
    video_sink_size: int
    video_chunk_size: int
    max_chunks: int
    video_decode_context_latents: int
    video_decode_tiling: bool
    attention_backend: str
    attention_benchmark: bool
    warmup_chunks: int
    profile_cuda: bool
    translation_speed: float
    rotation_speed_degrees: float
    pitch_speed_degrees: float
    pitch_limit_degrees: float
    fov_degrees: float
    example_names: tuple[str, ...]

    @property
    def wm_root(self) -> Path:
        """Return the pinned upstream Echo-WM package root."""
        return self.source_path / "echo_wm"

    @property
    def frames_per_chunk(self) -> int:
        """Return decoded RGB frames produced by one causal latent block."""
        return self.video_chunk_size * 8


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be zero or more")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _asset(value: Any, name: str) -> ModelAsset:
    raw = _mapping(value, name)
    return ModelAsset(
        path=_local_path(raw["path"]),
        repo_id=str(raw["repo_id"]),
        revision=str(raw["revision"]),
        filename=str(raw["filename"]) if raw.get("filename") else None,
    )


def read_config(path: Path | None) -> EchoWMConfig:
    """Read and validate the Echo-WM adapter configuration."""
    if path is None:
        raise ValueError("Echo-WM requires echo_wm.yaml")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = _mapping(document.get("source"), "source")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    motion = _mapping(document.get("motion"), "motion")
    examples = _mapping(document.get("examples"), "examples")
    width = _positive_int(inference.get("width"), "inference.width")
    height = _positive_int(inference.get("height"), "inference.height")
    if width % 32 or height % 32:
        raise ValueError("Echo-WM width and height must be divisible by 32")
    video_chunk_size = _positive_int(
        inference.get("video_chunk_size"), "inference.video_chunk_size"
    )
    if video_chunk_size != 3:
        raise ValueError("Echo-WM Flash requires inference.video_chunk_size=3")
    timesteps = tuple(int(value) for value in inference.get("timesteps", ()))
    if timesteps != (1000, 750, 500, 250):
        raise ValueError("Echo-WM Flash requires timesteps [1000, 750, 500, 250]")
    attention_backend = str(inference.get("attention_backend", "pytorch"))
    if attention_backend not in {"pytorch", "flash_attention_4"}:
        raise ValueError(
            "inference.attention_backend must be pytorch or flash_attention_4"
        )
    names = tuple(str(value) for value in examples.get("cases", ()))
    if not names:
        raise ValueError("examples.cases must contain at least one upstream case")
    default_upload_prompt = str(inference.get("default_upload_prompt", "")).strip()
    if not default_upload_prompt:
        raise ValueError("inference.default_upload_prompt must be non-empty")
    return EchoWMConfig(
        source_path=_local_path(source["path"]),
        source_url=str(source["url"]),
        source_revision=str(source["revision"]),
        checkpoint=_asset(assets.get("checkpoint"), "assets.checkpoint"),
        gemma=_asset(assets.get("gemma"), "assets.gemma"),
        cache_dir=_local_path(assets["cache_dir"]),
        runtime_dir=_local_path(assets["runtime_dir"]),
        width=width,
        height=height,
        fps=float(inference.get("fps", 24.0)),
        seed=int(inference.get("seed", 42)),
        default_upload_prompt=default_upload_prompt,
        timesteps=timesteps,
        video_local_attn_size=_positive_int(
            inference.get("video_local_attn_size"),
            "inference.video_local_attn_size",
        ),
        video_sink_size=_positive_int(
            inference.get("video_sink_size"), "inference.video_sink_size"
        ),
        video_chunk_size=video_chunk_size,
        max_chunks=_positive_int(inference.get("max_chunks"), "inference.max_chunks"),
        video_decode_context_latents=_positive_int(
            inference.get("video_decode_context_latents"),
            "inference.video_decode_context_latents",
        ),
        video_decode_tiling=_boolean(
            inference.get("video_decode_tiling", True),
            "inference.video_decode_tiling",
        ),
        attention_backend=attention_backend,
        attention_benchmark=_boolean(
            inference.get("attention_benchmark", False),
            "inference.attention_benchmark",
        ),
        warmup_chunks=_nonnegative_int(
            inference.get("warmup_chunks", 0), "inference.warmup_chunks"
        ),
        profile_cuda=_boolean(
            inference.get("profile_cuda", True), "inference.profile_cuda"
        ),
        translation_speed=float(motion.get("translation_speed", 0.05)),
        rotation_speed_degrees=float(motion.get("rotation_speed_degrees", 0.4)),
        pitch_speed_degrees=float(motion.get("pitch_speed_degrees", 0.2)),
        pitch_limit_degrees=float(motion.get("pitch_limit_degrees", 40.0)),
        fov_degrees=float(motion.get("fov_degrees", 70.0)),
        example_names=names,
    )


def configure_cache_environment(config: EchoWMConfig) -> None:
    """Keep model and compiled-kernel caches under the mounted weights root."""
    runtime_cache = config.runtime_dir / "cache"
    environment = {
        "HF_HOME": config.cache_dir,
        "XDG_CACHE_HOME": runtime_cache,
        "TORCHINDUCTOR_CACHE_DIR": runtime_cache / "torchinductor",
        "CUDA_CACHE_PATH": runtime_cache / "cuda",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": runtime_cache / "flash-attention-4",
    }
    for name, path in environment.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")


def prepare_assets(config: EchoWMConfig) -> None:
    """Clone or download missing public assets into configured NVMe paths."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    if not (config.source_path / ".git").is_dir():
        config.source_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", config.source_url, str(config.source_path)],
            check=True,
        )
    git = ["git", "-c", f"safe.directory={config.source_path}"]
    subprocess.run(
        [
            *git,
            "-C",
            str(config.source_path),
            "checkout",
            "--detach",
            config.source_revision,
        ],
        check=True,
    )
    actual_revision = subprocess.run(
        [*git, "-C", str(config.source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != config.source_revision:
        raise RuntimeError(
            f"Echo-WM source revision is {actual_revision}; expected {config.source_revision}"
        )
    if not config.checkpoint.path.is_file():
        from huggingface_hub import hf_hub_download

        if config.checkpoint.filename is None:
            raise ValueError("Echo-WM checkpoint filename is required")
        config.checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=config.checkpoint.repo_id,
            revision=config.checkpoint.revision,
            filename=config.checkpoint.filename,
            local_dir=config.checkpoint.path.parent,
            cache_dir=config.cache_dir,
        )
    if not (config.gemma.path / "model.safetensors.index.json").is_file():
        from huggingface_hub import snapshot_download

        config.gemma.path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=config.gemma.repo_id,
                revision=config.gemma.revision,
                local_dir=config.gemma.path,
                cache_dir=config.cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                "Gemma 3 is gated; accept its public license and set HF_TOKEN before startup"
            ) from error


def load_examples(config: EchoWMConfig) -> tuple[ExampleScene, ...]:
    """Load the configured upstream causal example metadata."""
    root = config.wm_root / "examples" / "wm_causal_cases"
    scenes: list[ExampleScene] = []
    for name in config.example_names:
        case_dir = root / name
        image = case_dir / "input.jpg"
        metadata_path = case_dir / "case.json"
        if not image.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Echo-WM example {name} is incomplete under {case_dir}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scenes.append(
            ExampleScene(
                image=image,
                prompt=str(metadata["prompt"]),
                fov_degrees=float(metadata.get("fov_deg", config.fov_degrees)),
                seed=int(metadata.get("seed", config.seed)),
            )
        )
    return tuple(scenes)


def activate_source(config: EchoWMConfig) -> None:
    """Put the pinned upstream packages first on the import path."""
    paths = (
        config.wm_root,
        config.wm_root / "ltx-core" / "src",
        config.wm_root / "ltx-causal" / "src",
        config.wm_root / "ltx-pipelines" / "src",
    )
    for path in reversed(paths):
        resolved = str(path.resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)


def _local_path(value: Any) -> Path:
    """Resolve one configured path under Reactor's mounted weights root."""
    path = Path(str(value)).expanduser()
    candidate = path if path.is_absolute() else get_weights_path() / path
    return Path(os.path.abspath(candidate))
