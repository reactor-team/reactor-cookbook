# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for the client-owned RTC plan protocol."""

from __future__ import annotations

import numpy as np
import pytest

from rldx1_rtc import (
    RTCRequestMailbox,
    build_rtc_request,
    resolve_rtc_timing,
)


def request(
    *,
    request_id: int = 0,
    base_plan_id: int = -1,
    install_step: int = 0,
    prefix_len: int = 0,
    prefix=(),
    expected_base_plan_id: int | None = None,
):
    return build_rtc_request(
        request_id=request_id,
        base_plan_id=base_plan_id,
        install_step=install_step,
        rtc_prefix_len=prefix_len,
        action_prefix=prefix,
        expected_base_plan_id=expected_base_plan_id,
        configured_delay=2,
        action_dim=3,
    )


def test_disabled_rtc_uses_the_full_action_horizon():
    timing = resolve_rtc_timing(
        action_horizon=16,
        mode="none",
        delay=0,
        exec_horizon=0,
    )
    assert not timing.enabled
    assert (timing.delay, timing.exec_horizon) == (0, 16)


def test_enabled_rtc_keeps_h_d_and_s_distinct():
    timing = resolve_rtc_timing(
        action_horizon=16,
        mode="guided",
        delay=2,
        exec_horizon=8,
    )
    assert timing.enabled
    assert (timing.action_horizon, timing.delay, timing.exec_horizon) == (16, 2, 8)


def test_zero_exec_horizon_defaults_to_h_minus_d():
    timing = resolve_rtc_timing(
        action_horizon=16,
        mode="guided",
        delay=2,
        exec_horizon=0,
    )
    assert timing.exec_horizon == 14


@pytest.mark.parametrize(
    ("delay", "exec_horizon"),
    [(0, 8), (3, 2), (2, 15)],
)
def test_invalid_rtc_timing_is_rejected(delay, exec_horizon):
    with pytest.raises(ValueError, match="RTC must satisfy"):
        resolve_rtc_timing(
            action_horizon=16,
            mode="guided",
            delay=delay,
            exec_horizon=exec_horizon,
        )


def test_cold_start_has_no_base_plan_or_prefix():
    first = request()
    assert first.base_plan_id == -1
    assert first.rtc_prefix_len == 0
    assert first.prefix_array().shape == (0,)


def test_next_plan_requires_the_configured_physical_action_prefix():
    next_request = request(
        request_id=1,
        base_plan_id=0,
        install_step=12,
        prefix_len=2,
        prefix=[[1, 2, 3], [4, 5, 6]],
        expected_base_plan_id=0,
    )
    assert next_request.prefix_array().dtype == np.float32
    assert next_request.prefix_array().shape == (2, 3)


def test_wrong_base_plan_is_rejected():
    with pytest.raises(ValueError, match="does not match active plan"):
        request(base_plan_id=3, expected_base_plan_id=2)


@pytest.mark.parametrize(
    ("prefix_len", "prefix", "message"),
    [
        (1, [[1, 2, 3]], "must be 2"),
        (2, [[1, 2, 3]], "has 1 rows"),
        (2, [[1, 2], [3, 4]], "expected 3"),
        (2, [[1, 2, float("nan")], [3, 4, 5]], "must be finite"),
    ],
)
def test_bad_prefix_is_rejected(prefix_len, prefix, message):
    with pytest.raises(ValueError, match=message):
        request(
            request_id=1,
            base_plan_id=0,
            prefix_len=prefix_len,
            prefix=prefix,
            expected_base_plan_id=0,
        )


def test_mailbox_is_single_flight_and_monotonic():
    mailbox = RTCRequestMailbox()
    first = request()
    mailbox.offer(first)
    with pytest.raises(ValueError, match="still pending"):
        mailbox.offer(first)
    assert mailbox.take() is first

    second = request(
        request_id=1,
        base_plan_id=0,
        prefix_len=2,
        prefix=[[1, 2, 3], [4, 5, 6]],
        expected_base_plan_id=0,
    )
    mailbox.offer(second)
    assert mailbox.take() is second
    with pytest.raises(ValueError, match="stale"):
        mailbox.offer(first)

    mailbox.clear()
    mailbox.offer(first)
    assert mailbox.take() is first
