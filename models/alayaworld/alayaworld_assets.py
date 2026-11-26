"""Read the AlayaWorld config and check its sources, weights, and scene inputs.

Nothing here downloads. The image holds the AlayaWorld and Depth-Anything-3
sources at their pinned revisions, and ``prepare_weights.py`` builds the
weights folder.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

PREPARE_COMMAND = "HF_TOKEN=<your token> python prepare_weights.py --output ./weights"
# The files of each pinned repository that the model reads, keyed by the
# `assets` entry of alayaworld.yaml. Each lands at its repository path in the
# entry's folder.
ASSET_FILES: dict[str, tuple[str, ...]] = {
    "model": ("merged_infer.safetensors",),
    "gemma": (
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        *(f"model-0000{index}-of-00005.safetensors" for index in range(1, 6)),
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ),
    "da3_model": ("config.json", "model.safetensors"),
}


@dataclass(frozen=True)
class Asset:
    """Describe one weights-folder directory and the pinned repository it holds."""

    path: Path
    repo_id: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class SourceCheckout:
    """Describe one upstream checkout that the image build makes."""

    name: str
    path: Path
    url: str
    revision: str


@dataclass(frozen=True)
class AlayaWorldConfig:
    """Hold validated source, asset, inference, and interaction settings."""

    weights_root: Path
    source: SourceCheckout
    da3_source: SourceCheckout
    upstream_config: Path
    upload_template: Path
    random_inputs: tuple[Path, ...]
    model: Asset
    gemma: Asset
    da3_model: Asset
    seed: int
    compile_mode: str
    warmup_chunks: int
    attention_backend: str
    flex_attention: bool
    ttc: bool
    bank_taehv: bool
    taehv_path: Path | None
    decode_overlap_latents: int
    max_chunks_per_rollout: int
    strafe_units_per_second: float
    vertical_units_per_second: float
    forward_units_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float
    roll_degrees_per_second: float

    @property
    def source_path(self) -> Path:
        """Return the AlayaWorld checkout in the image."""
        return self.source.path

    @property
    def checkpoint_path(self) -> Path:
        """Return the merged AlayaWorld checkpoint in the weights folder."""
        return self.model.path / ASSET_FILES["model"][0]


_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_COMPILE_MODES = ["none", "default", "reduce-overhead", "max-autotune"]
_ATTENTION_BACKENDS = ["flash_attention_4", "pytorch", "upstream"]
_SCENE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_SCENE_MEMBER_SUFFIXES = (
    "_video.mp4",
    "_camera.pt",
    "_prompt.txt",
    *tuple(f"_image{extension}" for extension in _SCENE_IMAGE_EXTENSIONS),
)


def read_config(config_path: Path | None, weights_root: Path) -> AlayaWorldConfig:
    """Read and validate the AlayaWorld adapter YAML.

    ``weights_root`` is the folder that ``prepare_weights.py`` builds; asset
    paths resolve under it. Source paths are absolute paths in the image.
    """
    if config_path is None:
        raise ValueError("AlayaWorld requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    inputs = _mapping(document.get("inputs"), "inputs")
    motion = _mapping(document.get("motion"), "motion")
    decode = _mapping(document.get("decode"), "decode")
    stream = _mapping(document.get("stream", {}), "stream")

    compile_mode = str(inference.get("compile", "reduce-overhead"))
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(
            f"inference.compile must be one of {', '.join(_COMPILE_MODES)}"
        )
    attention_backend = str(inference.get("attention_backend", "pytorch"))
    if attention_backend not in _ATTENTION_BACKENDS:
        raise ValueError(
            f"inference.attention_backend must be one of {', '.join(_ATTENTION_BACKENDS)}"
        )
    warmup_chunks = int(inference.get("warmup_chunks", 0))
    if warmup_chunks < 0:
        raise ValueError("inference.warmup_chunks must be zero or more")

    overlap = int(decode.get("overlap_latents", 6))
    if overlap <= 0:
        raise ValueError("decode.overlap_latents must be positive")
    max_chunks_per_rollout = int(stream.get("max_chunks_per_rollout", 512))
    if max_chunks_per_rollout < 1:
        raise ValueError("stream.max_chunks_per_rollout must be positive")

    motion_rates = {
        "strafe_units_per_second": float(motion.get("strafe_units_per_second", 0.126)),
        "vertical_units_per_second": float(
            motion.get("vertical_units_per_second", 0.261)
        ),
        "forward_units_per_second": float(
            motion.get("forward_units_per_second", 1.905)
        ),
        "pitch_degrees_per_second": float(
            motion.get("pitch_degrees_per_second", 4.039)
        ),
        "yaw_degrees_per_second": float(motion.get("yaw_degrees_per_second", 9.375)),
        "roll_degrees_per_second": float(motion.get("roll_degrees_per_second", 4.094)),
    }
    for name, value in motion_rates.items():
        if value <= 0:
            raise ValueError(f"motion.{name} must be positive")

    source = _source(document.get("source"), "source", "AlayaWorld source")
    da3_source = _source(
        document.get("da3_source"), "da3_source", "Depth-Anything-3 source"
    )
    taehv_raw = assets.get("taehv")
    random_inputs_raw = inputs.get("random_images")
    if not isinstance(random_inputs_raw, list) or not random_inputs_raw:
        raise ValueError("inputs.random_images must be a non-empty YAML list")
    return AlayaWorldConfig(
        weights_root=weights_root,
        source=source,
        da3_source=da3_source,
        upstream_config=_path(source.path, inference["config"]),
        upload_template=_path(source.path, inputs["upload_template"]),
        random_inputs=tuple(_path(source.path, value) for value in random_inputs_raw),
        model=_asset(weights_root, assets, "model"),
        gemma=_asset(weights_root, assets, "gemma"),
        da3_model=_asset(weights_root, assets, "da3_model"),
        seed=int(inference.get("seed", 1234)),
        compile_mode=compile_mode,
        warmup_chunks=warmup_chunks,
        attention_backend=attention_backend,
        flex_attention=bool(inference.get("flex_attention", True)),
        ttc=bool(inference.get("ttc", False)),
        bank_taehv=bool(inference.get("bank_taehv", False)),
        taehv_path=_path(weights_root, taehv_raw) if taehv_raw else None,
        decode_overlap_latents=overlap,
        max_chunks_per_rollout=max_chunks_per_rollout,
        strafe_units_per_second=motion_rates["strafe_units_per_second"],
        vertical_units_per_second=motion_rates["vertical_units_per_second"],
        forward_units_per_second=motion_rates["forward_units_per_second"],
        pitch_degrees_per_second=motion_rates["pitch_degrees_per_second"],
        yaw_degrees_per_second=motion_rates["yaw_degrees_per_second"],
        roll_degrees_per_second=motion_rates["roll_degrees_per_second"],
    )


def check_runtime_assets(config: AlayaWorldConfig) -> None:
    """Require both source checkouts, every weights file, and every scene input."""
    check_source(config)
    if not config.upstream_config.is_file():
        raise FileNotFoundError(
            f"AlayaWorld inference config not found: {config.upstream_config}"
        )
    _validate_scene_triplet(config.upload_template, "upload template")
    for index, path in enumerate(config.random_inputs):
        _validate_scene_triplet(path, f"random image {index}")
    check_weights(config)
    if config.bank_taehv and (
        config.taehv_path is None or not config.taehv_path.is_file()
    ):
        raise FileNotFoundError(
            "inference.bank_taehv requires assets.taehv, a file in the weights folder"
        )


def check_source(config: AlayaWorldConfig) -> None:
    """Require each upstream checkout in the image at its pinned revision."""
    for checkout in (config.source, config.da3_source):
        if not checkout.path.is_dir():
            raise FileNotFoundError(
                f"{checkout.name} not found at {checkout.path}; "
                "rebuild the image with `reactor build`"
            )
        actual = _run_git(
            [
                "-c",
                f"safe.directory={checkout.path}",
                "-C",
                str(checkout.path),
                "rev-parse",
                "HEAD",
            ],
            checkout.name,
        ).stdout.strip()
        if actual != checkout.revision:
            raise RuntimeError(
                f"{checkout.name} revision is {actual}; expected {checkout.revision}. "
                "Rebuild the image with `reactor build`."
            )


def check_weights(config: AlayaWorldConfig) -> None:
    """Raise ``FileNotFoundError`` naming the fix when a weights file is absent."""
    missing = [
        asset.path / name
        for asset in (config.model, config.gemma, config.da3_model)
        for name in asset.files
        if not (asset.path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"AlayaWorld weights are incomplete in {config.weights_root}: "
            f"{missing[0]} is missing ({len(missing)} files). Accept the Gemma "
            f"terms on Hugging Face, run `{PREPARE_COMMAND}`, and deploy with "
            "`--weights ./weights`."
        )


def scene_prompt_path(path: Path) -> Path:
    """Return the prompt member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_prompt.txt")


def scene_image_path(path: Path) -> Path:
    """Return the first supported still-image member for a configured scene."""
    prefix = _scene_prefix(path)
    for extension in _SCENE_IMAGE_EXTENSIONS:
        candidate = Path(f"{prefix}_image{extension}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"scene image not found for prefix: {prefix}")


def load_scene_metadata(path: Path, torch_module: Any) -> dict[str, Any]:
    """Load a fresh metadata mapping from the upload camera template."""
    value = torch_module.load(
        _scene_camera_path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(value, dict) or "cam_c2w" not in value:
        raise ValueError("upload camera template must contain cam_c2w metadata")
    return cast(dict[str, Any], value)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _source(value: object, name: str, label: str) -> SourceCheckout:
    """Read one absolute image path and pinned public repository."""
    document = _mapping(value, name)
    path = Path(str(document.get("path") or ""))
    if not path.is_absolute():
        raise ValueError(f"{name}.path must be the absolute checkout path in the image")
    return SourceCheckout(
        name=label,
        path=path,
        url=_repository_url(document.get("url"), f"{name}.url"),
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _asset(weights_root: Path, assets: dict[str, Any], key: str) -> Asset:
    """Read one weights-folder directory and its pinned repository identity."""
    name = f"assets.{key}"
    document = _mapping(assets.get(key), name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a Hugging Face repository")
    raw_path = str(document.get("path") or "")
    path = Path(raw_path)
    if not raw_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name}.path must be a folder inside the weights folder")
    return Asset(
        path=weights_root / path,
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
        files=ASSET_FILES[key],
    )


def _path(base_path: Path, value: object) -> Path:
    """Resolve a configured path relative to its owning directory."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _revision(value: object, name: str) -> str:
    """Return an immutable 40-character revision."""
    revision = str(value or "")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be a 40-character commit revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return the HTTPS URL for a public source repository."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _run_git(arguments: list[str], name: str) -> subprocess.CompletedProcess[str]:
    """Run a local Git query without an interactive credential prompt."""
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to check the {name} revision") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to read the {name} revision: {detail}") from error


def _scene_prefix(path: Path) -> Path:
    """Return a scene prefix from a prefix or one of its triplet members."""
    value = str(path)
    for suffix in _SCENE_MEMBER_SUFFIXES:
        if value.endswith(suffix):
            return Path(value[: -len(suffix)])
    return path


def _scene_camera_path(path: Path) -> Path:
    """Return the camera metadata member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_camera.pt")


def _validate_scene_triplet(path: Path, name: str) -> None:
    """Require the image, camera, and prompt files used by one scene."""
    members = {
        "image": scene_image_path(path),
        "camera": _scene_camera_path(path),
        "prompt": scene_prompt_path(path),
    }
    for member_name, member_path in members.items():
        if not member_path.is_file():
            raise FileNotFoundError(f"{name} {member_name} not found: {member_path}")
