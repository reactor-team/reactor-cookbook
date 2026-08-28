"""Resolve pinned public source and model assets for SANA-WM."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _configure_runtime_caches() -> None:
    """Keep downloaded assets inside the Runtime's persistent weights mount."""
    value = os.environ.get("REACTOR_WEIGHTS_PATH", "").strip()
    if not value:
        return
    root = Path(value).expanduser().resolve()
    cache_paths = {
        "HF_HOME": root / "huggingface",
        "TORCH_HOME": root / "torch",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))


_configure_runtime_caches()

from huggingface_hub import snapshot_download
from reactor_runtime.paths import get_weights_path

from sana_wm_types import BuiltInScene, HubAsset, SanaWMConfig

_SOURCE_ENV = "SANA_WM_SOURCE_PATH"


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


def read_config(config_path: Path | None) -> SanaWMConfig:
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

    source_override = os.environ.get(_SOURCE_ENV, "").strip()
    if source_override:
        source_path = Path(source_override).expanduser().resolve()
    else:
        source_path = (
            get_weights_path() / "sana-wm" / _required_str(source, "path", "source")
        ).resolve()

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
        pi3x_source_url=_required_str(pi3x, "source_url", "assets.pi3x"),
        pi3x_source_revision=_required_str(pi3x, "source_revision", "assets.pi3x"),
        scenes=tuple(scenes),
        seed=int(inference.get("seed", 42)),
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


def prepare_source(config: SanaWMConfig) -> None:
    """Clone the pinned SANA source when absent and verify an existing checkout."""
    source = config.source_path
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", config.source_url, str(source))
        _git("checkout", "--detach", config.source_revision, cwd=source)
    if not (source / ".git").exists():
        raise ValueError(f"SANA source path is not a Git checkout: {source}")
    revision = _git("rev-parse", "HEAD", cwd=source)
    if revision != config.source_revision:
        raise ValueError(
            f"SANA source revision mismatch at {source}: expected "
            f"{config.source_revision}, got {revision}"
        )
    required = [config.upstream_config]
    for scene in config.scenes:
        required.extend([scene.image, scene.prompt, scene.intrinsics])
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SANA source is missing required files: {missing}")


def resolve_model_assets(config: SanaWMConfig) -> tuple[Path, Path]:
    """Download pinned streaming and Stage-1 text-encoder snapshots."""
    streaming = Path(
        snapshot_download(
            repo_id=config.streaming.repo_id,
            revision=config.streaming.revision,
        )
    )
    text_encoder = Path(
        snapshot_download(
            repo_id=config.stage1_text_encoder.repo_id,
            revision=config.stage1_text_encoder.revision,
        )
    )
    return streaming, text_encoder


def resolve_pi3x_assets(config: SanaWMConfig) -> tuple[Path, Path]:
    """Prepare pinned Pi3X source and weights for calibration on demand."""
    source = get_weights_path() / "sana-wm" / "Pi3"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", config.pi3x_source_url, str(source))
        _git("checkout", "--detach", config.pi3x_source_revision, cwd=source)
    revision = _git("rev-parse", "HEAD", cwd=source)
    if revision != config.pi3x_source_revision:
        raise ValueError(
            f"Pi3X source revision mismatch at {source}: expected "
            f"{config.pi3x_source_revision}, got {revision}"
        )
    model = Path(
        snapshot_download(
            repo_id=config.pi3x_model.repo_id,
            revision=config.pi3x_model.revision,
        )
    )
    return source, model
