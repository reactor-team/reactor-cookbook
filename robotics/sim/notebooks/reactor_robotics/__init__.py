# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Minimal clients for Reactor-served robotics policies.

Six models over one shared transport. Each script stands alone; this table
is the map.

| Client | Model | Style | Chunk |
|---|---|---|---|
| :class:`~reactor_robotics.xwam.XwamClient` | ``xwam`` | lock-step, ``chunk_id`` echoed as ``step`` | ``(32, 14)`` delta joints |
| :class:`~reactor_robotics.lingbot_va.LingbotVaClient` | ``lingbot-va`` | lock-step, driven by the executed-action echo | ``(16, 7)`` eef deltas |
| :class:`~reactor_robotics.cosmos_droid.CosmosDroidClient` | ``cosmos-nano-policy-droid`` | stateless, one chunk per executed-step report | ``(32, 8)`` absolute joints |
| :class:`~reactor_robotics.groot_n17.GrootN17Client` | ``groot-n17`` | free-running, paired by engine ordering | ``(40, 17)`` across 3 named fields |
| :class:`~reactor_robotics.dreamzero.DreamZeroClient` | ``dreamzero`` | free-running, ignores chunks from old frames | ``(24, 8)`` absolute joints |
| :class:`~reactor_robotics.xr1_robocasa365.Xr1Robocasa365Client` | ``xr1-robocasa365`` | lock-step, echo-gated from the first chunk | ``(16, 60)`` packed, first 12 live |

``xwam`` is the reference implementation of the generic robot-policy contract
(``robot-policy-client-contract.md``); the other five each depart from it, and
each client's module docstring states how.

All six sit on :class:`~reactor_robotics.session.ReactorSession`, which
manages the connection lifecycle: handlers registered before ``connect()``,
tracks published only after ``READY``, and one paced stamped-frame publisher.

An evaluation harness and a controller for a physical robot use the *same
client*: they differ only in where the frames and the state come from, never
in the wire protocol.
"""

from .cosmos_droid import (
    ACTION_SHAPE as COSMOS_DROID_ACTION_SHAPE,
    TRACKS as COSMOS_DROID_TRACKS,
    CosmosDroidClient,
    CosmosDroidPrediction,
)
from .dreamzero import (
    ACTION_SHAPE as DREAMZERO_ACTION_SHAPE,
    FRANKA_JOINT_LIMITS,
    TRACKS as DREAMZERO_TRACKS,
    DreamZeroClient,
    DreamZeroPrediction,
)
from .groot_n17 import (
    ACTION_DIMS as GROOT_N17_ACTION_DIMS,
    ACTION_HORIZON as GROOT_N17_ACTION_HORIZON,
    VIEWS as GROOT_N17_VIEWS,
    GrootN17Client,
    GrootN17Prediction,
)
from .lingbot_va import (
    ACTION_SHAPE as LINGBOT_VA_ACTION_SHAPE,
    VIEWS as LINGBOT_VA_VIEWS,
    LingbotVaClient,
    LingbotVaPrediction,
)
from .session import (
    DEFAULT_API_URL,
    ReactorSession,
    api_url,
    describe_api_key,
    require_api_key,
)
from .track import RepeatingFrameTrack
from .xwam import (
    ACTION_SHAPE as XWAM_ACTION_SHAPE,
    VIEWS as XWAM_VIEWS,
    XwamClient,
    XwamPrediction,
)
from .xr1_robocasa365 import (
    ACTION_SHAPE as XR1_ROBOCASA365_ACTION_SHAPE,
    TRACKS as XR1_ROBOCASA365_TRACKS,
    Xr1Robocasa365Client,
    Xr1Robocasa365Prediction,
)

__all__ = [
    "COSMOS_DROID_ACTION_SHAPE",
    "COSMOS_DROID_TRACKS",
    "CosmosDroidClient",
    "CosmosDroidPrediction",
    "DEFAULT_API_URL",
    "DREAMZERO_ACTION_SHAPE",
    "DREAMZERO_TRACKS",
    "DreamZeroClient",
    "DreamZeroPrediction",
    "FRANKA_JOINT_LIMITS",
    "GROOT_N17_ACTION_DIMS",
    "GROOT_N17_ACTION_HORIZON",
    "GROOT_N17_VIEWS",
    "GrootN17Client",
    "GrootN17Prediction",
    "LINGBOT_VA_ACTION_SHAPE",
    "LINGBOT_VA_VIEWS",
    "LingbotVaClient",
    "LingbotVaPrediction",
    "ReactorSession",
    "RepeatingFrameTrack",
    "XR1_ROBOCASA365_ACTION_SHAPE",
    "XR1_ROBOCASA365_TRACKS",
    "XWAM_ACTION_SHAPE",
    "XWAM_VIEWS",
    "Xr1Robocasa365Client",
    "Xr1Robocasa365Prediction",
    "XwamClient",
    "XwamPrediction",
    "api_url",
    "describe_api_key",
    "require_api_key",
]
