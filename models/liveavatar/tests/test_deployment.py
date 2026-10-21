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
    assert manifest["build"]["runtime_version"] == "3.6.0"
    assert "--force-reinstall flash-attn-4==4.0.0b30" in manifest["build"]["run"][0]
    assert "from flash_attn.cute import" in manifest["build"]["run"][1]
    assert not (Path(__file__).parents[1] / "Dockerfile").exists()


def test_local_weights_are_resolved_without_hf_download(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
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
    assert assets.prepare_assets(tmp_path, source, local_only=True) == (
        tmp_path / "wan2_2",
        tmp_path / "liveavatar_lora",
    )
    (tmp_path / "wan2_2/config.json").unlink()
    with pytest.raises(RuntimeError, match="Incomplete mounted weights"):
        assets.prepare_assets(tmp_path, source, local_only=True)


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


def test_cache_setup_respects_operator_paths(tmp_path, monkeypatch):
    import os

    keys = [
        "HF_HOME",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TMPDIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    custom = tmp_path / "operator-cache"
    monkeypatch.setenv("HF_HOME", str(custom))
    work = assets.configure_cache_environment(tmp_path / "weights")
    assert os.environ["HF_HOME"] == str(custom)
    assert custom.is_dir()
    assert work == tmp_path / "weights/.runtime"
    assert Path(os.environ["TMPDIR"]).is_relative_to(work)


def test_model_load_uses_explicit_settings_without_asset_preparation(
    tmp_path, monkeypatch
):
    import pickle
    import liveavatar_parallel
    from liveavatar_model import LiveAvatarModel, LiveAvatarSettings
    from unittest.mock import Mock

    settings = LiveAvatarSettings(
        source=tmp_path,
        base=tmp_path / "base",
        lora=tmp_path / "lora",
        work=tmp_path / "work",
    )
    assert pickle.loads(pickle.dumps(settings)) == settings
    backend = Mock()
    monkeypatch.setattr(liveavatar_parallel, "ParallelBackend", backend)
    monkeypatch.setattr(
        assets,
        "prepare_assets",
        Mock(side_effect=AssertionError("model must not prepare assets")),
    )
    model = LiveAvatarModel()
    model.load(settings)
    backend.assert_called_once_with(settings=settings, rank=0, world_size=1)


def test_model_files_do_not_read_deployment_environment():
    import ast

    root = Path(__file__).parents[1]
    for name in [
        "liveavatar_model.py",
        "liveavatar_parallel.py",
        "liveavatar_parallel_worker.py",
        "liveavatar_turbo.py",
    ]:
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                assert ast.unparse(node.value) != "os.environ", (name, node.lineno)
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in {"os.getenv", "os.environ.get"}, (
                    name,
                    node.lineno,
                )
