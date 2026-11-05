"""Optional pinned-upstream CPU camera oracle; no GPU or weights required."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lingbot_world_v1_camera import load_camera_path


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
            chunk_latents=3,
        )
        count = len(path.chunks) * 3
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
