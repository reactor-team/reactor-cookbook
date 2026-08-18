"""Load OpenDreamer configuration, source modules, and conditioning assets."""

from __future__ import annotations

import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from opendreamer_types import (
    DEMO_CHOICES,
    DemoConfig,
    OpenDreamerConfig,
    RolloutConditioning,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_UPSTREAM_ENV = "OPENDREAMER_PATH"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def read_config(config_path: Path | None) -> OpenDreamerConfig:
    """Read and validate the OpenDreamer model YAML."""
    if config_path is None:
        raise ValueError("OpenDreamer requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    source = _mapping(document.get("source"), "source")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    conditioning = _mapping(document.get("conditioning", {}), "conditioning")
    source_revision = str(source.get("revision", ""))
    checkpoint_revision = str(checkpoint.get("revision", ""))
    if not _REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source.revision must be a full 40-character Git revision")
    if not _REVISION_PATTERN.fullmatch(checkpoint_revision):
        raise ValueError("checkpoint.revision must be a full 40-character revision")

    platform = str(document.get("platform", "cuda"))
    if platform not in {"cuda", "auto"}:
        raise ValueError("platform must be cuda or auto")
    num_steps = int(document.get("num_steps", 4))
    if num_steps <= 0 or num_steps & (num_steps - 1):
        raise ValueError("num_steps must be a positive power of two")
    tau_ctx_target = float(document.get("tau_ctx_target", 0.9))
    if not 0.0 < tau_ctx_target < 1.0:
        raise ValueError("tau_ctx_target must be between 0 and 1")
    conditioning_frames = int(conditioning.get("frames", 16))
    if conditioning_frames < 16:
        raise ValueError("conditioning.frames must be at least 16")
    warmup_steps = int(document.get("warmup_steps", 1))
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    memory_fraction = float(document.get("memory_fraction", 0.9))
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")

    demos = _read_demos(conditioning.get("demos"))
    return OpenDreamerConfig(
        source_revision=source_revision,
        checkpoint_repo_id=str(checkpoint["repo_id"]),
        checkpoint_revision=checkpoint_revision,
        platform=platform,
        seed=int(document.get("seed", 0)),
        num_steps=num_steps,
        tau_ctx_target=tau_ctx_target,
        conditioning_frames=conditioning_frames,
        demos=demos,
        warmup_steps=warmup_steps,
        memory_fraction=memory_fraction,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _read_demos(value: Any) -> tuple[DemoConfig, ...]:
    """Validate the three demos exposed by the runtime command schema."""
    if not isinstance(value, list) or len(value) != len(DEMO_CHOICES):
        raise ValueError("conditioning.demos must define demo_1, demo_2, and demo_3")
    demos: list[DemoConfig] = []
    for index, item in enumerate(value):
        demo = _mapping(item, f"conditioning.demos[{index}]")
        expected_name = DEMO_CHOICES[index]
        name = str(demo.get("name", ""))
        if name != expected_name:
            raise ValueError(
                f"conditioning.demos[{index}].name must be {expected_name}"
            )
        start_frame = int(demo.get("start_frame", 0))
        if start_frame < 0:
            raise ValueError(
                f"conditioning.demos[{index}].start_frame must be non-negative"
            )
        video = _relative_upstream_asset(
            demo.get("video"), f"conditioning.demos[{index}].video"
        )
        actions = _relative_upstream_asset(
            demo.get("actions"),
            f"conditioning.demos[{index}].actions",
        )
        demos.append(
            DemoConfig(
                name=name,
                video=video,
                actions=actions,
                start_frame=start_frame,
            )
        )
    return tuple(demos)


def _relative_upstream_asset(value: Any, name: str) -> Path:
    """Return a safe path relative to the configured upstream checkout."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay within OPENDREAMER_PATH")
    return path


def upstream_root() -> Path:
    """Return the external OpenDreamer checkout configured for this process.

    Raises:
        RuntimeError: If ``OPENDREAMER_PATH`` is unset or not an OpenDreamer checkout.
    """
    configured = os.environ.get(_UPSTREAM_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {_UPSTREAM_ENV} to the OpenDreamer repository checkout before starting Reactor"
        )

    root = Path(configured).expanduser().resolve()
    required = (
        root / "dreamer/actions.py",
        root / "dreamer/checkpointing.py",
        root / "dreamer/generation.py",
        root / "dreamer/models.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{_UPSTREAM_ENV}={root} is not an OpenDreamer checkout; missing: {joined}"
        )
    return root


def upstream_asset(upstream_root: Path, relative_path: Path) -> Path:
    """Resolve one validated demo asset inside the upstream checkout."""
    return (upstream_root / relative_path).resolve()


def ensure_demo_assets(upstream_root: Path, demos: tuple[DemoConfig, ...]) -> None:
    """Download the public default demo when its configured files are missing."""
    missing = {
        path
        for demo in demos
        for path in (
            upstream_asset(upstream_root, demo.video),
            upstream_asset(upstream_root, demo.actions),
        )
        if not path.is_file()
    }
    if not missing:
        return

    module_name = (
        f"{__package__}.opendreamer_assets" if __package__ else "opendreamer_assets"
    )
    assets = importlib.import_module(module_name)
    output_dir = upstream_root / "samples" / "vpt"
    default_paths = set(assets.demo_paths(output_dir))
    if not missing.issubset(default_paths):
        return
    logger.info(
        "downloading missing OpenDreamer demo assets", directory=str(output_dir)
    )
    assets.ensure_demo_assets(output_dir)


def verify_source_revision(source_path: Path, expected: str) -> None:
    """Require the local upstream clone to match the pinned public revision."""
    if not (source_path / "dreamer").is_dir():
        raise FileNotFoundError(f"OpenDreamer source not found at {source_path}")
    if not (source_path / ".git").exists():
        raise RuntimeError(
            f"OpenDreamer source at {source_path} must be a Git checkout"
        )
    result = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"OpenDreamer source is {actual}; expected {expected}")


def prepare_process_environment(config: OpenDreamerConfig) -> None:
    """Set JAX process options before importing JAX."""
    if config.platform == "cuda":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.memory_fraction))


def load_dependencies(source_path: Path) -> dict[str, Any]:
    """Import optional OpenDreamer dependencies after config validation."""
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    jax = importlib.import_module("jax")
    actions = importlib.import_module("dreamer.actions")
    checkpointing = importlib.import_module("dreamer.checkpointing")
    generation = importlib.import_module("dreamer.generation")
    models = importlib.import_module("dreamer.models")
    parallel = importlib.import_module("dreamer.parallel")
    utils = importlib.import_module("dreamer.utils")
    return {
        "jax": jax,
        "jnp": importlib.import_module("jax.numpy"),
        "nnx": importlib.import_module("flax.nnx"),
        "snapshot_download": importlib.import_module(
            "huggingface_hub"
        ).snapshot_download,
        "action_type": actions.Actions,
        "binary_actions": actions.NUM_BINARY_ACTIONS,
        "camera_classes": actions.NUM_CAMERA_CLASSES,
        "key_to_index": actions.key_to_index,
        "mouse_to_categorical": actions.mouse_movement_to_categorical,
        "parse_action_dicts": actions.parse_action_dicts,
        "shift_actions": actions.shift_actions,
        "bundle_type": checkpointing.DynamicsCheckpointBundle,
        "schedule_type": generation.DenoiseSchedule,
        "next_frame": generation.next_frame,
        "tokenizer_caches_type": models.TokenizerCaches,
        "build_parallel": parallel.build_parallel,
        "normalize_latents": utils.normalize_latents,
    }


def mesh_context(jax: Any, mesh: Any) -> AbstractContextManager[Any]:
    """Return the mesh context supported by the installed JAX version."""
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def read_conditioning_sequence(
    video_path: Path,
    actions_path: Path,
    target_shape: tuple[int, int, int],
    *,
    start_frame: int,
    required_frames: int,
    dependencies: Mapping[str, Any],
) -> RolloutConditioning:
    """Read one configured video and action window from disk."""
    if not video_path.is_file():
        raise FileNotFoundError(f"conditioning video not found at {video_path}")
    if not actions_path.is_file():
        raise FileNotFoundError(f"conditioning actions not found at {actions_path}")
    frames = decode_video_frames(
        video_path,
        target_shape,
        start_frame=start_frame,
        required_frames=required_frames,
    )
    action_dicts = load_action_dicts(actions_path.read_text())
    actions = prepare_conditioning_actions(
        action_dicts,
        start_frame=start_frame,
        required_frames=required_frames,
        dependencies=dependencies,
    )
    return RolloutConditioning(frames=frames, actions=actions)


def decode_conditioning_image(
    data: bytes,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Decode and center-crop one upload into an OpenDreamer RGB frame."""
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    height, width, channels = target_shape
    if channels != 3:
        raise ValueError(f"OpenDreamer requires three RGB channels, got {channels}.")
    content_height = height - 8 if height > 8 else height
    try:
        with image_module.open(io.BytesIO(data)) as uploaded:
            rgb = image_ops.exif_transpose(uploaded).convert("RGB")
            fitted = image_ops.fit(
                rgb,
                (width, content_height),
                method=image_module.Resampling.LANCZOS,
            )
            frame = np.asarray(fitted, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError("Could not decode the uploaded conditioning image.") from error
    return prepare_video_frame(frame, target_shape, index=0)


def decode_video_frames(
    source: Path | io.BytesIO,
    target_shape: tuple[int, int, int],
    *,
    start_frame: int,
    required_frames: int,
) -> np.ndarray:
    """Decode consecutive exact-size RGB frames from an MP4 source."""
    av = importlib.import_module("av")
    frames: list[np.ndarray] = []
    try:
        with av.open(source, mode="r") as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index < start_frame:
                    continue
                if len(frames) == required_frames:
                    break
                rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                frames.append(prepare_video_frame(rgb, target_shape, index=index))
    except (av.FFmpegError, IndexError) as error:
        raise ValueError("Could not decode the conditioning MP4.") from error
    if len(frames) < required_frames:
        raise ValueError(
            f"Conditioning video has {len(frames)} frames from offset {start_frame}; "
            f"expected at least {required_frames}."
        )
    return np.ascontiguousarray(np.stack(frames))


def prepare_video_frame(
    frame: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    index: int,
) -> np.ndarray:
    """Validate one model frame and pad the native 640x360 VPT format."""
    height, width, channels = target_shape
    if frame.shape == target_shape:
        return np.ascontiguousarray(frame)
    if frame.shape == (height - 8, width, channels):
        return np.pad(frame, ((4, 4), (0, 0), (0, 0)), mode="constant")
    raise ValueError(
        f"Conditioning frame {index} has shape {frame.shape}; expected "
        f"{target_shape} or {(height - 8, width, channels)}."
    )


def load_action_dicts(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array or newline-delimited VPT action objects."""
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("The conditioning action file is empty.")
    if stripped.startswith("["):
        document = json.loads(text)
        if not isinstance(document, list) or not all(
            isinstance(item, dict) for item in document
        ):
            raise ValueError("The conditioning JSON must be an array of objects.")
        return document

    actions: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid conditioning JSONL at line {line_number}."
            ) from error
        if not isinstance(item, dict):
            raise ValueError(
                f"Conditioning JSONL line {line_number} must be an object."
            )
        actions.append(item)
    if not actions:
        raise ValueError("The conditioning action file contains no actions.")
    return actions


def prepare_conditioning_actions(
    action_dicts: list[dict[str, Any]],
    *,
    start_frame: int,
    required_frames: int,
    dependencies: Mapping[str, Any],
) -> Any:
    """Parse, batch, shift, and slice actions to match conditioning frames."""
    required_actions = start_frame + required_frames
    if len(action_dicts) < required_actions:
        raise ValueError(
            f"Conditioning actions contain {len(action_dicts)} entries; "
            f"expected at least {required_actions}."
        )
    jnp = dependencies["jnp"]
    action_type = dependencies["action_type"]
    parsed = dependencies["parse_action_dicts"](action_dicts[:required_actions])

    def add_batch(value: Any) -> Any:
        return None if value is None else jnp.asarray(value)[None]

    batched = action_type(
        binary=add_batch(parsed.binary),
        categorical=add_batch(parsed.categorical),
        continuous=add_batch(parsed.continuous),
    )
    shifted = dependencies["shift_actions"](
        batched,
        int(dependencies["camera_classes"]),
    )
    return shifted[:, start_frame:required_actions]
