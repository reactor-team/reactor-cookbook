"""The pinned checkpoints, the files the weights folder holds, and where the runtime's caches go."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: Hugging Face repositories and revisions, and the folder under the weights root each one goes to.
CHECKPOINTS = (
    ("kyutai/AURA-1.3B", "53558287fd7c62d33fc4217f5f4e7b8d2d729526", "aura"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a", "wan2_1"),
    ("kyutai/moshiko-pytorch-bf16", "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289", "mimi"),
)

#: The files the model reads, under the weights root, with the upstream file patterns that fetch them.
DIT = "aura/stage2_dmd/step-6500_ema_merged.safetensors"
VAE = "wan2_1/Wan2.1_VAE.pth"
TEXT_ENCODER = "wan2_1/models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER = "wan2_1/google/umt5-xxl"
MIMI = "mimi/tokenizer-e351c8d8-checkpoint125.safetensors"
PATTERNS = {
    "aura": ["stage2_dmd/step-6500_ema_merged.safetensors"],
    "wan2_1": [
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/*",
    ],
    "mimi": ["tokenizer-e351c8d8-checkpoint125.safetensors"],
}


def prepare_weights(weights_root: Path) -> Path:
    """Check the weights, and point the compile caches at a writable folder.

    The caches go to ``AURA_WORK_DIR``, or a per-user folder under the system's temporary directory, so the weights
    can be mounted read-only. A cache variable already set in the environment wins.

    Returns:
        The weights root, holding ``aura/``, ``wan2_1/`` and ``mimi/``.

    Raises:
        RuntimeError: A weight file is missing.
    """
    missing = [
        name
        for name in (DIT, VAE, TEXT_ENCODER, MIMI)
        if not (weights_root / name).is_file()
    ]
    if not (weights_root / TOKENIZER).is_dir():
        missing.append(TOKENIZER)
    if missing:
        raise RuntimeError(
            f"incomplete weights under {weights_root}; run prepare_weights.py: {missing}"
        )
    override = os.environ.get("AURA_WORK_DIR")
    work = (
        Path(override).expanduser()
        if override
        else Path(tempfile.gettempdir()) / f"aura-{os.getuid()}"
    )
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, relative in {
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
    }.items():
        Path(os.environ.setdefault(name, str(work / relative))).mkdir(
            parents=True, exist_ok=True
        )
    return weights_root
