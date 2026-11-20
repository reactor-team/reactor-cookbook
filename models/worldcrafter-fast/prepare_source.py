"""Materialize a pinned upstream archive without changing an editable checkout."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from worldcrafter_assets import SOURCE_REVISION


def main():
    root = Path(os.environ.get("WORLDCRAFTER_ROOT", "/local-ssd/worldcrafter-reactor"))
    source = Path(os.environ.get("WORLDCRAFTER_SOURCE", str(root / "source")))
    marker = source / ".reactor-source.json"
    if marker.exists():
        if json.loads(marker.read_text())["revision"] != SOURCE_REVISION:
            raise RuntimeError("Source revision mismatch")
        if not all((source/name).is_file() for name in ("demo/engine.py", "uvenv/uv.lock", "pyproject.toml")):
            raise RuntimeError("Prepared source is incomplete")
        return
    if source.exists() and any(source.iterdir()):
        raise RuntimeError("Refusing to replace a nonempty source directory")
    url = f"https://codeload.github.com/TencentARC/WorldCrafter/tar.gz/{SOURCE_REVISION}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("Source archive exceeds the download limit")
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".worldcrafter-source-", dir=source.parent) as directory:
        stage = Path(directory)/"source"
        stage.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                full_path = Path(member.name)
                if (full_path.is_absolute() or ".." in full_path.parts or member.issym()
                        or member.islnk() or not (member.isdir() or member.isfile())):
                    raise ValueError("Unsafe source archive entry")
                parts = full_path.parts[1:]
                if not parts:
                    continue
                target = stage.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        if not all((stage/name).is_file() for name in ("demo/engine.py", "uvenv/uv.lock", "pyproject.toml")):
            raise ValueError("Source archive is incomplete")
        (stage/marker.name).write_text(json.dumps(dict(revision=SOURCE_REVISION)) + "\n")
        if source.exists():
            source.rmdir()  # Only the pre-existing empty destination can be removed.
        stage.rename(source)


if __name__ == "__main__":
    main()
