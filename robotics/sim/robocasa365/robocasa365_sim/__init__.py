# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""RoboCasa365 driven by a Reactor-served `xr1-robocasa365`.

The vendor's rollout loop, task registry, seeding and success criteria run
UNMODIFIED; only the client is swapped. :class:`ReactorEvalClient` matches
the vendor `EvalClient` interface and carries the observation over three
named WebRTC video tracks instead of the vendor's raw TCP socket.

Run an evaluation with ``python -m robocasa365_sim.entry``; check the
client-side wiring offline with ``python check_wiring.py``.
"""

from .client import ReactorEvalClient

__all__ = ["ReactorEvalClient"]
