"""Session cleanup and portable storage without model weights."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

import lyra2
from lyra2 import Lyra2
from lyra2_schema import Lyra2State


def test_session_end_clears_conditioning_and_progress():
    model = Lyra2()
    model.state = Lyra2State()
    model.backend = Mock()
    model.image = Path("selected.jpg")
    model.image_name = "selected.jpg"
    model.active_prompt = "old scene"
    model.chunk = 5
    model.generating = True
    model.ended()
    model.backend.clear.assert_called_once_with()
    assert model.image is None and model.image_name is None
    assert model.active_prompt is None and model.chunk == 0
    assert model.generating is False
    model.started()
    snapshot = model._state()
    assert snapshot.completed_chunks == 0
    assert snapshot.active_prompt is None and snapshot.image_name is None
    assert snapshot.prompt == ""


@pytest.mark.parametrize("absolute", [False, True])
def test_storage_paths_resolve_without_host_specific_directories(
    tmp_path, monkeypatch, absolute
):
    weights = tmp_path / "mounted-weights"
    cache = tmp_path / "explicit-cache" if absolute else Path("hf-cache")
    output = tmp_path / "explicit-output" if absolute else Path("outputs")
    config = {
        "source_path": str(tmp_path / "source"),
        "cache_path": str(cache),
        "output_path": str(output),
        "translation_per_frame": 0.0021875,
        "rotation_degrees_per_frame": 0.125,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(lyra2, "get_weights_path", lambda: weights)
    backend = Mock()
    monkeypatch.setattr(lyra2, "Lyra2Backend", backend)
    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "PYTORCH_ALLOC_CONF",
    ):
        monkeypatch.setenv(name, "existing-user-setting")
    model = Lyra2()
    model.load(path)
    assert Path(model.config["cache_path"]) == (cache if absolute else weights / cache)
    assert Path(model.config["output_path"]) == (
        output if absolute else weights / output
    )
    assert Path(model.config["cache_path"]).is_dir()
    assert Path(model.config["output_path"]).is_dir()
    backend.assert_called_once_with(model.config)
