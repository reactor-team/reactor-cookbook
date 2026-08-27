"""Provide configuration, import, image, and tensor helpers for DIAMOND."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from diamond_types import AdapterConfig

UPSTREAM_ENV = "DIAMOND_PATH"
_INFERENCE_IMPORT_STUBS = ("ale_py", "wandb")


@contextmanager
def _inference_import_scope() -> Iterator[None]:
    """Provide placeholders for upstream dependencies unused during inference."""
    inserted: list[str] = []
    for module_name in _INFERENCE_IMPORT_STUBS:
        if module_name not in sys.modules:
            sys.modules[module_name] = ModuleType(module_name)
            inserted.append(module_name)
    try:
        yield
    finally:
        for module_name in inserted:
            sys.modules.pop(module_name, None)


def decode_spawn_image(
    data: bytes,
    *,
    full_resolution: tuple[int, int],
    low_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode and center-crop one upload into DIAMOND's CHW frame sizes.

    Args:
        data: Encoded image bytes.
        full_resolution: Target ``(height, width)`` for the upsampler.
        low_resolution: Target ``(height, width)`` for the world model.

    Returns:
        Full-resolution and low-resolution contiguous uint8 RGB arrays.

    Raises:
        ValueError: If Pillow cannot decode the upload.
    """
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    full_height, full_width = full_resolution
    low_height, low_width = low_resolution
    try:
        with image_module.open(BytesIO(data)) as uploaded:
            rgb = image_ops.exif_transpose(uploaded).convert("RGB")
            fitted = image_ops.fit(
                rgb,
                (full_width, full_height),
                method=image_module.Resampling.LANCZOS,
            )
            low = fitted.resize(
                (low_width, low_height),
                resample=image_module.Resampling.LANCZOS,
            )
            full_array = np.asarray(fitted, dtype=np.uint8).transpose(2, 0, 1)
            low_array = np.asarray(low, dtype=np.uint8).transpose(2, 0, 1)
    except (OSError, ValueError) as error:
        raise ValueError("could not decode uploaded spawn image") from error
    return np.ascontiguousarray(full_array), np.ascontiguousarray(low_array)


def read_config(config_path: Path | None) -> AdapterConfig:
    """Read and validate the adapter's model configuration.

    Args:
        config_path: Model configuration path supplied by Reactor Runtime.

    Returns:
        The validated adapter configuration.

    Raises:
        TypeError: If the YAML document is not a mapping.
        ValueError: If the path or a supported option is invalid.
    """
    if config_path is None:
        raise ValueError("DIAMOND requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise TypeError(f"{config_path}: expected a YAML mapping")

    profile = str(document.get("profile", "fast"))
    if profile not in {"fast", "higher_quality"}:
        raise ValueError("profile must be 'fast' or 'higher_quality'")
    device = str(document.get("device", "auto"))
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    return AdapterConfig(
        repo_id=str(document.get("repo_id", "eloialonso/diamond")),
        revision=str(document["revision"]),
        device=device,
        profile=profile,
        seed=int(document.get("seed", 0)),
    )


def upstream_root() -> Path:
    """Return the external DIAMOND checkout configured for this process.

    Returns:
        The validated DIAMOND repository root.

    Raises:
        RuntimeError: If ``DIAMOND_PATH`` is unset or does not identify a CSGO checkout.
    """
    configured = os.environ.get(UPSTREAM_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {UPSTREAM_ENV} to the DIAMOND repository checkout before starting Reactor"
        )

    root = Path(configured).expanduser().resolve()
    required = (
        root / "config/trainer.yaml",
        root / "src/agent.py",
        root / "src/csgo/action_processing.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{UPSTREAM_ENV}={root} is not a DIAMOND CSGO checkout; missing: {joined}"
        )
    return root


def load_upstream_modules(upstream_root: Path) -> dict[str, Any]:
    """Import DIAMOND from an external, unmodified ``src`` tree.

    Args:
        upstream_root: Validated DIAMOND repository root.

    Returns:
        The upstream modules used by the adapter.
    """
    source = str(upstream_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    with _inference_import_scope():
        return {
            "agent": importlib.import_module("agent"),
            "world": importlib.import_module("envs"),
            "action": importlib.import_module("csgo.action_processing"),
            "pygame": importlib.import_module("pygame"),
        }


def load_adapter_dependencies() -> dict[str, Any]:
    """Import DIAMOND's optional dependencies when model loading begins.

    Returns:
        The third-party objects needed to load DIAMOND.
    """
    hydra = importlib.import_module("hydra")
    return {
        "torch": importlib.import_module("torch"),
        "snapshot_download": importlib.import_module(
            "huggingface_hub"
        ).snapshot_download,
        "compose": hydra.compose,
        "initialize_config_dir": hydra.initialize_config_dir,
        "instantiate": importlib.import_module("hydra.utils").instantiate,
        "omega_conf": importlib.import_module("omegaconf").OmegaConf,
    }


def select_device(requested: str, torch_module: Any) -> Any:
    """Return the requested accelerator, preferring MPS on Apple Silicon.

    Args:
        requested: Requested device or ``auto``.
        torch_module: Imported PyTorch module.

    Returns:
        A PyTorch device.

    Raises:
        RuntimeError: If an explicitly requested accelerator is unavailable.
    """
    if requested == "auto":
        if torch_module.cuda.is_available():
            requested = "cuda"
        elif torch_module.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch_module.device(requested)


def resolve_upstream_eval(expression: str) -> float:
    """Resolve the one trusted expression used by DIAMOND's sampler config.

    Args:
        expression: Expression supplied by the pinned upstream config.

    Returns:
        Positive infinity for DIAMOND's supported expression.

    Raises:
        ValueError: If the expression is outside the pinned config contract.
    """
    if expression != 'float("inf")':
        raise ValueError(f"unsupported DIAMOND config expression: {expression!r}")
    return math.inf


def to_video_frame(observation: Any) -> np.ndarray:
    """Convert one DIAMOND NCHW observation into contiguous uint8 RGB.

    Args:
        observation: Batched DIAMOND observation tensor in the ``[-1, 1]`` range.

    Returns:
        A contiguous HWC uint8 RGB frame.
    """
    frame = (
        observation[0]
        .detach()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(frame)
