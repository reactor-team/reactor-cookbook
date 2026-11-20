"""Check the worker's release-recipe settings and output quantization without weights."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import worker


class _Decoded:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value


def test_publish_chunk_matches_official_quantization():
    # A bright chunk whose minimum stays above -0.05 is still a [-1, 1] decode.
    rng = np.random.default_rng(0)
    decoded = rng.uniform(-0.04, 1.2, size=(1, 3, 2, 4, 5)).astype(np.float32)
    session = worker._InteractiveSession(1, None)
    session.publish_chunk(0, _Decoded(decoded))
    frames = session._results.get().frames
    # diffusers postprocess_video, then infer_single.py's uint8 conversion.
    expected = np.clip(np.transpose(decoded[0], (1, 2, 3, 0)) * 0.5 + 0.5, 0.0, 1.0)
    expected = (expected * 255.0).round().astype(np.uint8)
    np.testing.assert_array_equal(frames, expected)


def test_load_argv_uses_released_depth_resolution():
    runtime = SimpleNamespace(
        _runtime_root=Path("root"),
        _settings={"base_model": "base", "transformer": "dit", "vigeo_path": "vigeo"},
        _stability_prompt="Scene",
        _default_image=Path("image.jpg"),
    )
    argv = worker.EvokeRuntime._build_load_argv(runtime)
    assert argv[argv.index("--geo_da3_process_res") + 1] == "644"


def test_release_recipe_constants():
    assert "walking backwards" in worker._NEGATIVE_PROMPT
    assert (
        worker._POSE_SMOOTH_WINDOW,
        worker._MAX_DEGREES_PER_CHUNK,
        worker._MAX_TRANSLATION_PER_CHUNK,
        worker._POSE_TAIL_UNUSED,
    ) == (5, 10.0, 3.0, 3)
