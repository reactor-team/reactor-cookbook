"""Streaming output reclamation preserves the native long-term spatial bank."""
from types import SimpleNamespace

from alayaworld_utils import compact_rollout_cache


def test_spatial_memory_is_retained_beyond_320_frames():
    bank = SimpleNamespace(
        pixels=[object() for _ in range(500)],
        frame_indices=list(range(500)),
        depths=[object() for _ in range(500)],
        world_points={i: object() for i in range(500)},
    )
    members = dict(vars(bank))
    predictions = [object(), object(), object()]
    cache = SimpleNamespace(spatial_bank=bank, preds=predictions.copy())
    compact_rollout_cache(cache)
    assert cache.preds == predictions[-1:]
    assert cache.spatial_bank is bank
    for name, value in members.items():
        assert vars(bank)[name] is value
        assert len(value) == 500
