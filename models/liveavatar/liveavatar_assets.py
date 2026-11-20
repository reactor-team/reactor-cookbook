"""The pinned upstream source and checkpoints, and where the runtime's caches go."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

SOURCE_REVISION = "c3c47d031d8bf3247428333c1fe610579c71a551"
BASE_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
LORA_REVISION = "92cdccd12a91e8a63767a7c821b7c75e51d5a172"

#: Files the weights folder must hold: the base checkpoint in ``wan2_2/``, LiveAvatar's in ``liveavatar_lora/``.
REQUIRED = (
    "wan2_2/config.json",
    "wan2_2/Wan2.1_VAE.pth",
    "liveavatar_lora/liveavatar.safetensors",
)


def prepare_weights(weights_root: Path, source: Path) -> Path:
    """Check the upstream checkout and the weights, and point the compile caches at a writable folder.

    The caches go to ``LIVEAVATAR_WORK_DIR``, or a per-user folder under the system's temporary directory, so the
    weights can be mounted read-only. A cache variable already set in the environment wins.

    Returns:
        The weights root, holding ``wan2_2/`` and ``liveavatar_lora/``.

    Raises:
        RuntimeError: The checkout is not the pinned revision, or a weight file is missing.
    """
    if not source.is_dir():
        raise RuntimeError(f"the upstream LiveAvatar source is not at {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"expected upstream LiveAvatar {SOURCE_REVISION}, found {revision}"
        )
    missing = [name for name in REQUIRED if not (weights_root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete weights under {weights_root}; run prepare_weights.py: {missing}"
        )
    override = os.environ.get("LIVEAVATAR_WORK_DIR")
    work = (
        Path(override).expanduser()
        if override
        else Path(tempfile.gettempdir()) / f"liveavatar-{os.getuid()}"
    )
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, relative in {
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUTE_DSL_CACHE_DIR": "cute",
    }.items():
        Path(os.environ.setdefault(name, str(work / relative))).mkdir(
            parents=True, exist_ok=True
        )
    return weights_root
