"""Optional pinned-upstream CPU camera and RGB oracles; no GPU or weights required."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lingbot_world_v2_camera import load_camera_path
from lingbot_world_v2_backend import LingBotBackend


def test_camera_rounding_matches_native_torch_before_ray_embedding():
    pytest.importorskip("torch")
    source = os.environ.get("LINGBOT_NATIVE_SOURCE")
    if not source:
        pytest.skip("Set LINGBOT_NATIVE_SOURCE to the pinned upstream checkout")
    spec = importlib.util.spec_from_file_location(
        "native_lingbot_camera", Path(source) / "wan/utils/cam_utils.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    rng = np.random.default_rng(42)
    for _ in range(10):
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 161, axis=0)
        poses[:, :3, :3] = Rotation.from_rotvec(
            np.cumsum(rng.normal(0, 0.02, (161, 3)), axis=0)
        ).as_matrix()
        poses[:, :3, 3] = np.cumsum(rng.normal(0, 0.1, (161, 3)), axis=0)
        path = load_camera_path(
            json.dumps(
                dict(poses=poses.tolist(), intrinsics=[503, 503, 416, 240])
            ).encode(),
            max_chunks=320,
            chunk_latents=4,
        )
        count = len(path.chunks) * 4
        expected = native.compute_relative_poses(
            native.interpolate_camera_poses(
                np.arange(161),
                poses[:, :3, :3],
                poses[:, :3, 3],
                np.linspace(0, 160, count),
            ),
            framewise=True,
        ).numpy()
        np.testing.assert_array_equal(path.chunks.reshape(-1, 4, 4), expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_decoded_rgb_matches_native_save_video_before_encoding(monkeypatch, dtype):
    torch = pytest.importorskip("torch")
    source = os.environ.get("LINGBOT_NATIVE_SOURCE")
    if not source:
        pytest.skip("Set LINGBOT_NATIVE_SOURCE to the pinned upstream checkout")
    # Invoke the real pinned save_video; intercept only its video encoder.
    spec = importlib.util.spec_from_file_location(
        "native_lingbot_video", Path(source) / "wan/utils/utils.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    captured = []
    monkeypatch.setattr(
        native.imageio,
        "get_writer",
        lambda *args, **kwargs: SimpleNamespace(
            append_data=lambda frame: captured.append(frame.copy()), close=lambda: None
        ),
    )
    # Include clipping, exact endpoints, and fractional pixels on both sides of
    # uint8 boundaries. Zero must become 127, not the 128 produced by rounding.
    values = torch.linspace(-1.5, 1.5, 3 * 3 * 4 * 8).reshape(3, 3, 4, 8)
    values[0, 0, 0, :5] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    decoded_dtype = getattr(torch, dtype)
    model = SimpleNamespace(
        z_dim=3,
        conv2=lambda value: value,
        decoder=lambda value, **kwargs: value.to(decoded_dtype),
        _feat_map=[],
    )
    backend = LingBotBackend.__new__(LingBotBackend)
    backend._torch = torch
    backend._pipe = SimpleNamespace(
        vae=SimpleNamespace(
            model=model, scale=(torch.zeros(3), torch.ones(3)), dtype=torch.float32
        )
    )
    actual = backend._decode_chunk(values)
    native.save_video(values.to(decoded_dtype).float().clamp(-1, 1).unsqueeze(0))
    expected = np.stack(captured)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0, 0, :5, 0], [0, 0, 127, 255, 255])
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous
