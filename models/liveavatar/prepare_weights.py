"""Materialize pinned checkpoints without symlinks; reuse local NVMe blobs."""

import argparse
import errno
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from liveavatar_assets import BASE_REVISION, LORA_REVISION, configure_cache_environment


def link_or_copy(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.exists():
        if os.path.samefile(source, destination):
            return str(destination)
        raise FileExistsError(
            f"Refusing to overwrite an existing weight: {destination}"
        )
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
    return str(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    configure_cache_environment()
    for repo, revision, folder in [
        ("Wan-AI/Wan2.2-S2V-14B", BASE_REVISION, "wan2_2"),
        ("Quark-Vision/Live-Avatar", LORA_REVISION, "liveavatar_lora"),
    ]:
        source = snapshot_download(
            repo, revision=revision, local_files_only=not args.allow_download
        )
        shutil.copytree(
            source, args.output / folder, copy_function=link_or_copy, dirs_exist_ok=True
        )
    print(f"Weights ready at {args.output.resolve()}")


if __name__ == "__main__":
    main()
