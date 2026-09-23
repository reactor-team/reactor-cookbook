# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Client-side stats formatting and polling without a hosted model."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

from reactor_sdk import ConnectionStats, ReactorError, ReactorStatus

from main import Client, format_webrtc_stats, parse_args, report_webrtc_stats


def empty_stats() -> ConnectionStats:
    return ConnectionStats(
        rtt_ms=None, jitter_s=None, packet_loss_ratio=None,
        incoming_bitrate_bps=None, outgoing_bitrate_bps=None,
        available_incoming_bitrate_bps=None, available_outgoing_bitrate_bps=None,
        target_bitrate_bps=None, frames_per_second=None,
        candidate_type=None, relay_protocol=None, candidate_pair_state=None,
        packets_received=0, packets_lost=0, packets_sent=0,
        bytes_received=0, bytes_sent=0, timestamp_ms=0,
    )


class StatsFormattingTest(unittest.TestCase):
    def test_periodic_line_prints_only_current_rtt(self) -> None:
        for rtt, expected in ((None, "N/A"), (float("nan"), "N/A"),
                              (0.0, "0.00"), (12.5, "12.50")):
            with self.subTest(rtt=rtt):
                self.assertEqual(
                    format_webrtc_stats(replace(empty_stats(), rtt_ms=rtt)),
                    f"[webrtc] rtt_ms={expected}",
                )

    def test_final_summary_prints_average_once_or_unavailable(self) -> None:
        for samples, expected in (
            ([0.0, 30.0, 60.0], "WebRTC RTT ms: avg=30.00 (from 3 samples)"),
            ([], "WebRTC RTT: unavailable — no RTT samples collected"),
        ):
            with self.subTest(samples=samples), patch("main.Reactor"), \
                    patch("builtins.print") as output:
                client = Client(Mock())
                client.webrtc_rtt_ms = samples
                client.summary(False, False)
            rtt_lines = [call.args[0] for call in output.call_args_list
                         if call.args[0].startswith("WebRTC RTT")]
            self.assertEqual(rtt_lines, [expected])

    def test_interval_accepts_disabled_and_rejects_invalid_windows(self) -> None:
        for value in ("0", "0.2", "2", "5"):
            with self.subTest(value=value), patch(
                "sys.argv", ["main.py", "--api-key", "test", "--stats-interval", value]
            ):
                self.assertEqual(parse_args().stats_interval, float(value))
        for value in ("-1", "0.1", "nan", "inf"):
            with self.subTest(value=value), patch(
                "sys.argv", ["main.py", "--api-key", "test", "--stats-interval", value]
            ), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()


class StatsPollingTest(unittest.IsolatedAsyncioTestCase):
    async def test_collects_valid_rtt_for_summary_without_printing_average(self) -> None:
        samples: list[float] = []
        reactor = Mock()
        reactor.get_status.return_value = ReactorStatus.READY
        reactor.get_stats = AsyncMock(side_effect=[
            replace(empty_stats(), rtt_ms=rtt)
            for rtt in (None, 0.0, None, 30.0, float("nan"), 60.0)
        ])
        with patch("main.asyncio.sleep", side_effect=[None] * 5 + [asyncio.CancelledError]), \
                patch("builtins.print") as output:
            with self.assertRaises(asyncio.CancelledError):
                await report_webrtc_stats(reactor, 10.0, samples)
        self.assertEqual(samples, [0.0, 30.0, 60.0])
        expected = ("N/A", "0.00", "N/A", "30.00", "N/A", "60.00")
        self.assertEqual(output.call_count, len(expected))
        for call, rtt in zip(output.call_args_list, expected):
            self.assertEqual(call.args[0], f"[webrtc] rtt_ms={rtt}")

    async def test_skips_not_ready_connection(self) -> None:
        reactor = Mock()
        reactor.get_status.return_value = ReactorStatus.CONNECTING
        reactor.get_stats = AsyncMock()
        with patch("main.asyncio.sleep", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await report_webrtc_stats(reactor, 2.0, [])
        reactor.get_stats.assert_not_awaited()

    async def test_stats_error_does_not_stop_subsequent_reports(self) -> None:
        reactor = Mock()
        reactor.get_status.return_value = ReactorStatus.READY
        reactor.get_stats = AsyncMock(side_effect=[
            ReactorError("connection changed"), empty_stats(),
        ])
        with patch("main.asyncio.sleep", side_effect=[None, asyncio.CancelledError]), \
                patch("builtins.print") as output:
            with self.assertRaises(asyncio.CancelledError):
                await report_webrtc_stats(reactor, 2.0, [])
        self.assertEqual(reactor.get_stats.await_count, 2)
        self.assertIn("connection changed", output.call_args_list[0].args[0])
        output.assert_any_call(format_webrtc_stats(empty_stats()))

    async def test_cancellation_interrupts_an_inflight_stats_request(self) -> None:
        entered = asyncio.Event()

        async def pending_stats() -> ConnectionStats:
            entered.set()
            await asyncio.Future()

        reactor = Mock()
        reactor.get_status.return_value = ReactorStatus.READY
        reactor.get_stats = AsyncMock(side_effect=pending_stats)
        task = asyncio.create_task(report_webrtc_stats(reactor, 2.0, []))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
