"""Check the pinned source and weights folder for SANA-WM. Nothing here downloads."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
STREAMING_DIR = "SANA-WM_streaming"
TEXT_ENCODER_DIR = "gemma-2-2b-it"
PI3X_DIR = "Pi3X"
STREAMING_REQUIRED_FILES = (
    "sana_dit/model.pt",
    "ltx2_causal_vae/config.json",
    "ltx2_causal_vae/diffusion_pytorch_model.safetensors",
    "refiner_diffusers/connectors/config.json",
    "refiner_diffusers/connectors/diffusion_pytorch_model.safetensors",
    "refiner_diffusers/transformer/config.json",
    "refiner_diffusers/transformer/diffusion_pytorch_model.safetensors",
    "gemma3_12b/added_tokens.json",
    "gemma3_12b/chat_template.json",
    "gemma3_12b/config.json",
    "gemma3_12b/generation_config.json",
    "gemma3_12b/model.safetensors.index.json",
    "gemma3_12b/preprocessor_config.json",
    "gemma3_12b/processor_config.json",
    "gemma3_12b/special_tokens_map.json",
    "gemma3_12b/tokenizer.model",
    "gemma3_12b/tokenizer_config.json",
    *(f"gemma3_12b/model-0000{i}-of-00005.safetensors" for i in range(1, 6)),
)
TEXT_ENCODER_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
PI3X_REQUIRED_FILES = ("config.json", "model.safetensors")


def configure_runtime_caches() -> Path:
    """Point the library caches at the home folder. The weights folder stays read-only."""
    root = Path.home() / ".cache" / "sana-wm"
    cache_paths = {
        "HF_HOME": root / "huggingface",
        "TORCH_HOME": root / "torch",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    return root


@dataclass(frozen=True)
class HubAsset:
    """One immutable Hugging Face asset."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class BuiltInScene:
    """One image, prompt, and calibration example."""

    name: str
    image: Path
    prompt: Path
    intrinsics: Path


@dataclass(frozen=True)
class SanaWMConfig:
    """Resolved source, assets, and native streaming settings."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    streaming: HubAsset
    stage1_text_encoder: HubAsset
    pi3x_model: HubAsset
    pi3x_source_path: Path
    pi3x_source_url: str
    pi3x_source_revision: str
    scenes: tuple[BuiltInScene, ...]
    seed: int
    refiner_seed: int
    upstream_fast_defaults: bool
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_dynamic: bool
    max_chunks: int
    num_cached_blocks: int
    refiner_kv_max_frames: int
    translation_speed: float
    rotation_speed_degrees: float
    pitch_limit_degrees: float
    weights_root: Path

    @property
    def streaming_path(self) -> Path:
        """Return the SANA-WM streaming snapshot inside the weights folder."""
        return self.weights_root / STREAMING_DIR

    @property
    def text_encoder_path(self) -> Path:
        """Return the Stage-1 Gemma text-encoder snapshot."""
        return self.weights_root / TEXT_ENCODER_DIR

    @property
    def pi3x_path(self) -> Path:
        """Return the Pi3X calibration weights."""
        return self.weights_root / PI3X_DIR


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a mapping or raise a configuration error naming its field."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_str(mapping: Mapping[str, Any], key: str, name: str) -> str:
    """Return one required non-empty string from a configuration mapping."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _hub_asset(mapping: Mapping[str, Any], name: str) -> HubAsset:
    """Parse one immutable Hugging Face asset declaration."""
    return HubAsset(
        repo_id=_required_str(mapping, "repo_id", name),
        revision=_required_str(mapping, "revision", name),
    )


def read_config(config_path: Path | None, weights_root: Path) -> SanaWMConfig:
    """Read and validate the adapter YAML handed to `load()`."""
    if config_path is None:
        raise ValueError("SANA-WM requires a configuration path")
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    source = _mapping(root.get("source"), "source")
    assets = _mapping(root.get("assets"), "assets")
    inference = _mapping(root.get("inference"), "inference")
    motion = _mapping(root.get("motion"), "motion")

    source_path = Path(_required_str(source, "path", "source"))

    upstream_config = source_path / _required_str(source, "config", "source")
    scene_values = root.get("scenes")
    if not isinstance(scene_values, list) or not scene_values:
        raise ValueError("scenes must be a non-empty list")
    scenes: list[BuiltInScene] = []
    for index, value in enumerate(scene_values):
        scene = _mapping(value, f"scenes[{index}]")
        scenes.append(
            BuiltInScene(
                name=_required_str(scene, "name", f"scenes[{index}]"),
                image=source_path / _required_str(scene, "image", f"scenes[{index}]"),
                prompt=source_path / _required_str(scene, "prompt", f"scenes[{index}]"),
                intrinsics=source_path
                / _required_str(scene, "intrinsics", f"scenes[{index}]"),
            )
        )

    max_chunks = int(inference.get("max_chunks", 512))
    num_cached_blocks = int(inference.get("num_cached_blocks", 2))
    refiner_kv_max_frames = int(inference.get("refiner_kv_max_frames", 11))
    if max_chunks < 1:
        raise ValueError("inference.max_chunks must be positive")
    if num_cached_blocks < 1:
        raise ValueError("inference.num_cached_blocks must be positive")
    if refiner_kv_max_frames < 4:
        raise ValueError(
            "inference.refiner_kv_max_frames must preserve sink plus one block"
        )

    pi3x = _mapping(assets.get("pi3x"), "assets.pi3x")
    return SanaWMConfig(
        weights_root=weights_root,
        source_path=source_path,
        source_url=_required_str(source, "url", "source"),
        source_revision=_required_str(source, "revision", "source"),
        upstream_config=upstream_config,
        streaming=_hub_asset(
            _mapping(assets.get("streaming"), "assets.streaming"),
            "assets.streaming",
        ),
        stage1_text_encoder=_hub_asset(
            _mapping(assets.get("stage1_text_encoder"), "assets.stage1_text_encoder"),
            "assets.stage1_text_encoder",
        ),
        pi3x_model=_hub_asset(
            _mapping(pi3x.get("model"), "assets.pi3x.model"),
            "assets.pi3x.model",
        ),
        pi3x_source_path=Path(_required_str(pi3x, "source_path", "assets.pi3x")),
        pi3x_source_url=_required_str(pi3x, "source_url", "assets.pi3x"),
        pi3x_source_revision=_required_str(pi3x, "source_revision", "assets.pi3x"),
        scenes=tuple(scenes),
        seed=int(inference.get("seed", 42)),
        refiner_seed=int(inference.get("refiner_seed", 42)),
        upstream_fast_defaults=bool(inference.get("upstream_fast_defaults", True)),
        torch_compile=bool(inference.get("torch_compile", True)),
        torch_compile_mode=str(
            inference.get("torch_compile_mode", "max-autotune-no-cudagraphs")
        ),
        torch_compile_dynamic=bool(inference.get("torch_compile_dynamic", True)),
        max_chunks=max_chunks,
        num_cached_blocks=num_cached_blocks,
        refiner_kv_max_frames=refiner_kv_max_frames,
        translation_speed=float(motion.get("translation_speed", 0.025)),
        rotation_speed_degrees=float(motion.get("rotation_speed_degrees", 0.6)),
        pitch_limit_degrees=float(motion.get("pitch_limit_degrees", 60.0)),
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run Git without invoking a shell and return trimmed stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _check_checkout(name: str, path: Path, revision: str) -> None:
    """Require a Git checkout at the pinned revision in the image."""
    if not (path / ".git").exists():
        raise FileNotFoundError(
            f"{name} source is missing at {path}; rebuild the image with `reactor build`"
        )
    actual = _git("rev-parse", "HEAD", cwd=path)
    if actual != revision:
        raise ValueError(
            f"{name} source revision mismatch at {path}: expected {revision}, got {actual}"
        )


def check_source(config: SanaWMConfig) -> None:
    """Verify the SANA source in the image. This function never clones."""
    _check_checkout("SANA", config.source_path, config.source_revision)
    required = [config.upstream_config]
    for scene in config.scenes:
        required.extend([scene.image, scene.prompt, scene.intrinsics])
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SANA source is missing required files: {missing}")
    _check_checkout("Pi3", config.pi3x_source_path, config.pi3x_source_revision)


def check_weights(config: SanaWMConfig) -> None:
    """Raise when the weights folder lacks a file that the model reads."""
    required = (
        [config.streaming_path / name for name in STREAMING_REQUIRED_FILES]
        + [config.text_encoder_path / name for name in TEXT_ENCODER_REQUIRED_FILES]
        + [config.pi3x_path / name for name in PI3X_REQUIRED_FILES]
    )
    missing = [
        str(path) for path in required if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"SANA-WM weights lack: {', '.join(missing)}. "
            f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."
        )


def check_runtime(config: SanaWMConfig) -> None:
    """Check the source and the weights. This function never downloads."""
    check_source(config)
    check_weights(config)
