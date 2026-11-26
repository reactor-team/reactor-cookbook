# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Build the HY-World 1.5 weights folder from the pinned Hugging Face and ModelScope revisions.

Run it once, on any machine with about 95 GB free, before ``reactor run`` or
``reactor deploy``. Every source is public and ungated, so no token is needed::

    pip install huggingface-hub modelscope pyyaml
    python prepare_weights.py --output ./weights

The folder holds ``HunyuanVideo-1.5`` (the 480p transformer, VAE, and
scheduler, with the Qwen, ByT5, Glyph, and SigLIP encoders in the subfolders
the upstream pipeline reads) and ``HY-WorldPlay`` (the distilled action
checkpoint). The files are plain files, not links. ``load()`` reads the folder
read-only. A second run downloads only the files that are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hy_world_1_5_assets import (
    check_weights,
    huggingface_files,
    modelscope_files,
    read_config,
)

_MODEL_DIR = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> None:
    """Download the pinned files into ``output``, then check them as ``load()`` does."""
    from huggingface_hub import hf_hub_download
    from modelscope.hub.snapshot_download import snapshot_download

    config = read_config(config_path, output)
    for file in huggingface_files(config):
        hf_hub_download(
            repo_id=file.repo_id,
            revision=file.revision,
            filename=file.filename,
            local_dir=file.local_dir,
        )
    glyph = config.glyph
    snapshot_download(
        glyph.repo_id,
        revision=glyph.revision,
        local_dir=str(glyph.path),
        allow_patterns=[file.filename for file in modelscope_files(config)],
    )
    check_weights(config)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and build the weights folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_MODEL_DIR / "weights")
    parser.add_argument("--config", type=Path, default=_MODEL_DIR / "hy_world_1_5.yaml")
    args = parser.parse_args(argv)
    prepare(args.config, args.output)
    print(f"weights ready in {args.output}; deploy with --weights {args.output}")


if __name__ == "__main__":
    main()
