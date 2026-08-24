# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Checkpoint-derived schema for the session-start handshake.

The checkpoint's modality config (RLWRLD's source of truth) defines what the
model expects; Reactor carries those values to the client, it doesn't define
them. ``build_schema`` is the pure seam: modality config in, ``model_schema``
message payload out. Kept free of torch/runtime imports so it is testable
anywhere and safe for ``reactor schema`` on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_schema(
    *,
    views: Sequence[str],
    declared_views: Sequence[str],
    video_delta_indices: Sequence[int],
    state_dims: Mapping[str, int],
    action_dims: Mapping[str, int],
    action_horizon: int,
    exec_horizon: int,
    control_hz: float,
    resolution: Sequence[int],
    rtc_mode: str,
    embodiment: str,
    state_fallback: str,
    state_source: str,
    state_tag_keys: Sequence[str],
) -> dict[str, Any]:
    """Assemble the ``model_schema`` payload from checkpoint-derived values.

    All inputs are the values ``load()`` resolved from the checkpoint (with
    config fallbacks) — exactly what this process serves with, so the
    handshake can never disagree with the pipeline's behaviour. Normalizes
    everything to JSON-native types. Raises ``ValueError`` if the
    checkpoint's camera views don't match the input tracks this port
    declares — a checkpoint this port cannot serve must fail at load, not
    feed garbage at runtime.
    """
    views = list(views)
    missing = [v for v in views if v not in set(declared_views)]
    extra = [v for v in declared_views if v not in set(views)]
    if missing or extra:
        raise ValueError(
            f"checkpoint camera views {views} do not match the declared input "
            f"tracks {list(declared_views)} (missing tracks: {missing}, "
            f"unused tracks: {extra})"
        )

    return {
        "views": [str(v) for v in views],
        "resolution": [int(resolution[0]), int(resolution[1])],
        "video_delta_indices": [int(d) for d in video_delta_indices],
        "control_hz": float(control_hz),
        "state_dims": {str(k): int(v) for k, v in state_dims.items()},
        "action_dims": {str(k): int(v) for k, v in action_dims.items()},
        "action_horizon": int(action_horizon),
        "exec_horizon": int(exec_horizon),
        "rtc_mode": str(rtc_mode),
        "dtype": "float32",
        "embodiment": str(embodiment),
        "state_fallback": str(state_fallback),
        "state_source": str(state_source),
        "state_tag_keys": [str(k) for k in state_tag_keys],
    }
