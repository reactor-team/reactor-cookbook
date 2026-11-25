# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Lyra 2.0 paths, weight checks and the pinned hub revisions.

The check only resolves and verifies files. It never downloads. Build the
weights folder with ``prepare_weights.py``.
"""

import os
from pathlib import Path

import yaml

PREPARE_COMMAND = "python prepare_weights.py --output ./weights"
FIX = f"Run `{PREPARE_COMMAND}` and deploy with `--weights ./weights`."

CHECKPOINT_REPO = "nvidia/Lyra-2.0"
CHECKPOINT_REVISION = "c178c3fcf12b63cf98f6749999e6ecb63901669f"
# The folder in the weights root that the pinned source reads as `checkpoints/`.
CHECKPOINT_DIR = "source/Lyra-2/checkpoints"
CHECKPOINT_FILES = (
    "image_encoder/model.pth",
    "lora/detail_enhancer.safetensors",
    "lora/dmd_distillation.safetensors",
    "lora/realism_boost.safetensors",
    "model/model/.metadata",
    *(f"model/model/__{index}_0.distcp" for index in range(64)),
    "recon/model.pt",
    "text_encoder/encoder.pth",
    "text_encoder/negative_prompt.pt",
    "vae/images_mean_std.pt",
    "vae/vae.pth",
    "vae/video_mean_std.pt",
)

# Hub repos that the pinned source reads through the Hugging Face cache.
HUB_CACHE_DIR = "hf-cache/hub"
HUB_REPOS = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": (
        "b2359bdf726fb44ef62acca04d629dcf158053e7",
        ("config.json", "model.safetensors"),
    ),
    "google/umt5-xxl": (
        "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
        (
            "config.json",
            "generation_config.json",
            *(f"pytorch_model-0000{i}-of-00006.bin" for i in range(1, 7)),
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    "Ruicheng/moge-vitl": (
        "ad326bfb61facd6c52b5a825bc1e34d7c97d9672",
        ("model.pt",),
    ),
}

SOURCE_URL = "https://github.com/nv-tlabs/lyra.git"
SOURCE_REVISION = "9fffc9adc37004091ecf26ef03abfb3abdf4d59a"


def snapshot_dir(weights_root: Path, repo_id: str, revision: str) -> Path:
    """Return the Hugging Face cache snapshot folder of a pinned repo revision."""
    name = "models--" + repo_id.replace("/", "--")
    return weights_root / HUB_CACHE_DIR / name / "snapshots" / revision


def check_weights(weights_root: Path) -> None:
    """Raise ``FileNotFoundError`` naming the fix when a prepared file is absent."""
    root = Path(weights_root)
    expected = [root / CHECKPOINT_DIR / name for name in CHECKPOINT_FILES]
    for repo_id, (revision, names) in HUB_REPOS.items():
        folder = snapshot_dir(root, repo_id, revision)
        expected += [folder / name for name in names]
        expected.append(root / HUB_CACHE_DIR / folder.parents[1].name / "refs" / "main")
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Lyra-2 weights are incomplete in {root}: {missing[0]} is missing "
            f"({len(missing)} files). {FIX}"
        )


def check_source(source: Path) -> None:
    """Raise ``FileNotFoundError`` when the image lacks the pinned Lyra source."""
    if not (source / "gui" / "api").is_dir():
        raise FileNotFoundError(
            f"The Lyra source is missing at {source}; rebuild the image with "
            "`reactor build`."
        )


def prepare_config(
    config_path: Path, weights_root: Path, cache_root: Path | None = None
) -> dict:
    """Resolve the config paths and cache variables. Nothing is downloaded or written
    into the weights folder."""
    weights_root = Path(weights_root)
    cache_root = cache_root or Path("~/.cache/lyra-2").expanduser()
    check_weights(weights_root)
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source_path"]).expanduser()
    config["source_path"] = str((config_path.parent / source).resolve())
    check_source(Path(config["source_path"]))
    config["checkpoints_path"] = str(
        (weights_root / Path(config["checkpoints_path"]).expanduser()).resolve()
    )
    for key in ("cache_path", "output_path"):
        config[key] = str((cache_root / Path(config[key]).expanduser()).resolve())
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_path"])
    # The hub cache is read-only in the weights folder. Other caches are writable.
    for key, value in {
        "HF_HOME": weights_root / "hf-cache",
        "HUGGINGFACE_HUB_CACHE": weights_root / HUB_CACHE_DIR,
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache / "xdg",
    }.items():
        os.environ[key] = str(value)
    checkpoints = Path(config["checkpoints_path"])
    link = Path(config["source_path"]) / "checkpoints"
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoints:
            raise ValueError(
                f"Source checkpoint location {link} does not match {checkpoints}"
            )
    else:
        # The link sits in the image source folder, not in the weights folder.
        link.symlink_to(checkpoints, target_is_directory=True)
    return config
