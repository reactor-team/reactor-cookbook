# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Client-owned scheduling state for RLDX real-time chunking (RTC)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RTCTiming:
    """Resolved action/prefix/execution horizons for one loaded policy."""

    action_horizon: int
    delay: int
    exec_horizon: int
    enabled: bool


def resolve_rtc_timing(
    *,
    action_horizon: int,
    mode: str,
    delay: int,
    exec_horizon: int,
) -> RTCTiming:
    """Resolve and validate ``H``/``d``/``s`` without loading torch."""

    horizon = int(action_horizon)
    if horizon < 1:
        raise ValueError(f"action_horizon must be positive, got {horizon}")

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"none", "guided", "trained"}:
        raise ValueError(f"unsupported RTC mode {mode!r}")
    if normalized_mode == "none":
        return RTCTiming(
            action_horizon=horizon,
            delay=0,
            exec_horizon=horizon,
            enabled=False,
        )

    d = int(delay)
    s = int(exec_horizon) if int(exec_horizon) > 0 else horizon - d
    if not (0 < d <= s <= horizon - d):
        raise ValueError(
            f"RTC must satisfy 0 < delay ({d}) <= exec_horizon ({s}) "
            f"<= action_horizon - delay ({horizon - d})"
        )
    return RTCTiming(
        action_horizon=horizon,
        delay=d,
        exec_horizon=s,
        enabled=True,
    )


@dataclass(frozen=True)
class RTCRequest:
    """One client-triggered RTC inference request."""

    request_id: int
    base_plan_id: int
    install_step: int
    rtc_prefix_len: int
    action_prefix: tuple[tuple[float, ...], ...]

    def prefix_array(self) -> np.ndarray:
        """Return the physical-unit action prefix expected by ``RLDXPolicy``."""

        return np.asarray(self.action_prefix, dtype=np.float32)


def build_rtc_request(
    *,
    request_id: int,
    base_plan_id: int,
    install_step: int,
    rtc_prefix_len: int,
    action_prefix: Sequence[Sequence[float]],
    expected_base_plan_id: int | None,
    configured_delay: int,
    action_dim: int,
) -> RTCRequest:
    """Validate the plan chain and the physical-unit prefix from the client."""

    for name, value in (
        ("request_id", request_id),
        ("base_plan_id", base_plan_id),
        ("install_step", install_step),
        ("rtc_prefix_len", rtc_prefix_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")

    if request_id < 0:
        raise ValueError("request_id must be non-negative")
    if install_step < 0:
        raise ValueError("install_step must be non-negative")

    cold_start = expected_base_plan_id is None
    expected_base = -1 if cold_start else expected_base_plan_id
    if base_plan_id != expected_base:
        raise ValueError(
            f"base_plan_id={base_plan_id} does not match active plan "
            f"{expected_base}; reset before resynchronizing"
        )

    expected_prefix_len = 0 if cold_start else int(configured_delay)
    if rtc_prefix_len != expected_prefix_len:
        raise ValueError(
            f"rtc_prefix_len={rtc_prefix_len} must be {expected_prefix_len} "
            f"for base_plan_id={base_plan_id}"
        )

    rows = tuple(tuple(row) for row in action_prefix)
    if len(rows) != rtc_prefix_len:
        raise ValueError(
            f"action_prefix has {len(rows)} rows but rtc_prefix_len={rtc_prefix_len}"
        )

    parsed_rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != action_dim:
            raise ValueError(
                f"action_prefix[{row_index}] has dim {len(row)}; expected {action_dim}"
            )
        parsed: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"action_prefix[{row_index}][{column_index}] must be numeric"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"action_prefix[{row_index}][{column_index}] must be finite"
                )
            parsed.append(value)
        parsed_rows.append(tuple(parsed))

    return RTCRequest(
        request_id=request_id,
        base_plan_id=base_plan_id,
        install_step=install_step,
        rtc_prefix_len=rtc_prefix_len,
        action_prefix=tuple(parsed_rows),
    )


class RTCRequestMailbox:
    """Single-flight, monotonically ordered RTC request mailbox."""

    def __init__(self) -> None:
        self._pending: RTCRequest | None = None
        self._last_request_id: int | None = None

    @property
    def pending(self) -> RTCRequest | None:
        return self._pending

    def offer(self, request: RTCRequest) -> None:
        if self._pending is not None:
            raise ValueError(
                f"request {self._pending.request_id} is still pending; "
                "only one RTC inference may be in flight"
            )
        if (
            self._last_request_id is not None
            and request.request_id <= self._last_request_id
        ):
            raise ValueError(
                f"request_id={request.request_id} is stale; last accepted request was "
                f"{self._last_request_id}"
            )
        self._pending = request
        self._last_request_id = request.request_id

    def take(self) -> RTCRequest | None:
        request = self._pending
        self._pending = None
        return request

    def clear(self) -> None:
        self._pending = None
        self._last_request_id = None
