"""Cold-start integrity, interrupted preparation and mounted-weight contracts."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

import prepare_source
import worldcrafter_assets as assets


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    weights = tmp_path / "mounted"
    weights.mkdir()
    payloads = {"inference_config.json": b"{}", "transformer/model.bin": b"weights"}
    manifest = dict(revision=assets.MODEL_REVISION, files={})
    for name, payload in payloads.items():
        path = weights/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest["files"][name] = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    inventory = tmp_path/"inventory.json"
    inventory.write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, "WEIGHT_MANIFEST", inventory)
    return weights, tmp_path/"ssd"/"verified.json"


def test_config_alone_does_not_mark_weights_ready(checkpoint):
    weights, cache = checkpoint
    (weights/"transformer/model.bin").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        assets.verify_weights(weights, cache)
    assert not cache.exists()


@pytest.mark.parametrize("payload", [b"short", b"corrupt"])
def test_partial_or_corrupt_weight_has_no_receipt(checkpoint, payload):
    weights, cache = checkpoint
    (weights/"transformer/model.bin").write_bytes(payload)
    with pytest.raises(ValueError):
        assets.verify_weights(weights, cache)
    assert not cache.exists()


def test_matching_receipt_cannot_skip_content_verification(checkpoint):
    weights, cache = checkpoint
    assets.verify_weights(weights, cache)
    original = json.loads(cache.read_text())
    assets.verify_weights(weights, cache)
    assert json.loads(cache.read_text()) == original
    model = weights/"transformer/model.bin"
    stamp = model.stat()
    model.write_bytes(b"corrupt")
    os.utime(model, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    # Reproduce indistinguishable metadata even on filesystems whose ctime
    # changed during this write. Matching metadata must not bypass hashing.
    stamp = model.stat()
    original["files"]["transformer/model.bin"] = [stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns]
    cache.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="checksum"):
        assets.verify_weights(weights, cache)
    assert not cache.exists()
    assert sorted(p.name for p in weights.iterdir()) == ["inference_config.json", "transformer"]


def test_corrupt_receipt_is_rebuilt(checkpoint):
    weights, cache = checkpoint
    cache.parent.mkdir()
    cache.write_text("interrupted write")
    assets.verify_weights(weights, cache)
    assert json.loads(cache.read_text())["revision"] == assets.MODEL_REVISION


def test_runtime_uses_native_weights_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDCRAFTER_ROOT", str(tmp_path/"runtime"))
    monkeypatch.delenv("WORLDCRAFTER_WEIGHTS", raising=False)
    mounted = tmp_path/"reactor-mounted-checkpoint"
    assert assets.read_settings(None, mounted).weights == mounted


def archive_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in files:
            member = tarfile.TarInfo(name)
            member.size = 4
            archive.addfile(member, io.BytesIO(b"test"))
    return buffer.getvalue()


def test_interrupted_source_does_not_poison_retry(tmp_path, monkeypatch):
    source = tmp_path/"source"
    monkeypatch.setenv("WORLDCRAFTER_SOURCE", str(source))
    broken = archive_bytes(["root/demo/engine.py"])
    monkeypatch.setattr(prepare_source.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(broken))
    with pytest.raises(ValueError, match="incomplete"):
        prepare_source.main()
    assert not source.exists()
    assert not list(tmp_path.glob(".worldcrafter-source-*"))
    complete = archive_bytes(["root/demo/engine.py", "root/uvenv/uv.lock", "root/pyproject.toml"])
    monkeypatch.setattr(prepare_source.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(complete))
    prepare_source.main()
    prepare_source.main()
    assert json.loads((source/".reactor-source.json").read_text())["revision"] == assets.SOURCE_REVISION


@pytest.mark.parametrize("name", ["root/../../outside", "/absolute/file"])
def test_source_archive_cannot_escape_destination(tmp_path, monkeypatch, name):
    source = tmp_path/"source"
    monkeypatch.setenv("WORLDCRAFTER_SOURCE", str(source))
    payload = archive_bytes([name])
    monkeypatch.setattr(prepare_source.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_source.main()
    assert not source.exists()


def test_existing_source_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDCRAFTER_SOURCE", str(tmp_path))
    user_file = tmp_path/"user.py"
    user_file.write_text("keep")
    with pytest.raises(RuntimeError, match="nonempty"):
        prepare_source.main()
    assert user_file.read_text() == "keep"


@pytest.fixture
def geocalib(tmp_path, monkeypatch):
    payload = b"geocalib checkpoint"
    monkeypatch.setattr(assets, "GEOCALIB_SIZE", len(payload))
    monkeypatch.setattr(assets, "GEOCALIB_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path/"torch"/assets.GEOCALIB_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def test_staged_geocalib_checkpoint_passes(geocalib):
    assets.verify_geocalib(geocalib)


def test_missing_geocalib_checkpoint_fails_before_any_download(geocalib):
    geocalib.unlink()
    with pytest.raises(FileNotFoundError, match="GeoCalib"):
        assets.verify_geocalib(geocalib)


def test_corrupt_geocalib_checkpoint_fails(geocalib):
    geocalib.write_bytes(b"geocalib checkpoinX")
    with pytest.raises(ValueError, match="checksum"):
        assets.verify_geocalib(geocalib)
