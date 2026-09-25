import subprocess
from pathlib import Path

import pytest
import yaml

import liveavatar_assets as assets
from prepare_weights import link_or_copy


def test_manifest_has_native_yaml_build_and_three_gpu_profile():
    manifest = yaml.safe_load((Path(__file__).parents[1] / "reactor.yaml").read_text())
    assert manifest["model"]["resources"]["gpu"]["count"] == 3
    assert manifest["build"]["runtime_env"]["LIVEAVATAR_TURBO"] == "1"
    assert manifest["build"]["runtime_env"]["LIVEAVATAR_STEPS"] == "4"
    assert manifest["build"]["runtime_version"] == "3.2.5"
    assert "--force-reinstall flash-attn-4==4.0.0b30" in manifest["build"]["run"][0]
    assert "from flash_attn.cute import" in manifest["build"]["run"][1]
    assert not (Path(__file__).parents[1] / "Dockerfile").exists()


def test_local_weights_are_resolved_without_hf_download(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(assets, "SOURCE", source)
    monkeypatch.setattr(assets, "LOCAL_WEIGHTS_ONLY", True)
    monkeypatch.setattr(assets, "mounted_weights_path", lambda: tmp_path)
    monkeypatch.setattr(assets, "configure_cache_environment", lambda: None)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: assets.SOURCE_REVISION + "\n",
    )

    def reject_source_mutation(*args, **kwargs):
        pytest.fail("Preparing existing assets must not mutate upstream source")

    monkeypatch.setattr(subprocess, "run", reject_source_mutation)
    for name in [
        "wan2_2/config.json",
        "wan2_2/Wan2.1_VAE.pth",
        "liveavatar_lora/liveavatar.safetensors",
    ]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture")
    assert assets.prepare_assets() == (
        tmp_path / "wan2_2",
        tmp_path / "liveavatar_lora",
    )
    (tmp_path / "wan2_2/config.json").unlink()
    with pytest.raises(RuntimeError, match="Incomplete mounted weights"):
        assets.prepare_assets()


def test_weight_materialization_dereferences_and_never_overwrites(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    reference = tmp_path / "snapshot"
    reference.symlink_to(blob)
    destination = tmp_path / "materialized"
    link_or_copy(reference, destination)
    assert not destination.is_symlink()
    assert destination.samefile(blob)
    link_or_copy(reference, destination)
    other = tmp_path / "other"
    other.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        link_or_copy(other, destination)
