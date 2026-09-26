"""Pinned source, checkpoints, and uploaded visual conditioning."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from open_oasis_types import OpenOasisConfig
from PIL import Image, ImageOps
from reactor_runtime import UploadedFile, get_weights_path


def read_config(path: Path | None) -> OpenOasisConfig:
    if path is None:
        raise ValueError("Open-Oasis requires open_oasis.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OpenOasisConfig(
        source_path=os.environ.get(
            "OPEN_OASIS_PATH",
            os.path.expandvars(str(raw["source"]["path"])),
        ),
        source_revision=str(raw["source"]["revision"]),
        checkpoint_repo_id=str(raw["checkpoint"]["repo_id"]),
        checkpoint_revision=str(raw["checkpoint"]["revision"]),
        model_filename=str(raw["checkpoint"]["model_filename"]),
        vae_filename=str(raw["checkpoint"]["vae_filename"]),
        seed=int(raw["inference"]["seed"]),
        ddim_steps=int(raw["inference"]["ddim_steps"]),
        context_frames=int(raw["inference"]["context_frames"]),
        fps=float(raw["inference"]["fps"]),
    )


def prepare_source(config: OpenOasisConfig) -> Path:
    root = Path(config.source_path)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/etched-ai/open-oasis.git", str(root)],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config.source_revision:
        raise RuntimeError(
            f"Open-Oasis source revision {revision} does not match pinned {config.source_revision}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def download_checkpoints(config: OpenOasisConfig) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    cache = get_weights_path() / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": config.checkpoint_repo_id,
        "revision": config.checkpoint_revision,
        "cache_dir": cache,
        "token": os.environ.get("HF_KEY"),
    }
    return (
        Path(hf_hub_download(filename=config.model_filename, **kwargs)),
        Path(hf_hub_download(filename=config.vae_filename, **kwargs)),
    )


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image = (
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize((640, 360), Image.Resampling.BILINEAR)
        )
        return np.asarray(image, dtype=np.uint8)[None]


def decode_video(upload: UploadedFile, offset: int, count: int) -> np.ndarray:
    from torchvision.io import read_video

    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(upload.data)
        handle.flush()
        frames = read_video(handle.name, pts_unit="sec", output_format="TCHW")[0]
    frames = frames[offset : offset + count]
    if len(frames) != count:
        raise ValueError(f"video contains fewer than {offset + count} frames")
    from torch.nn import functional

    frames = functional.interpolate(
        frames.float(), size=(360, 640), mode="bilinear", align_corners=False
    )
    return frames.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
