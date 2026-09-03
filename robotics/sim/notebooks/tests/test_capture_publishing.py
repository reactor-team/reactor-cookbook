# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Regression coverage for sender-authored camera capture timestamps."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from dreamzero_yam_bridge_i2rt import CapturedFrame, publish_camera_frames
from reactor_robotics.session import ReactorSession
from reactor_robotics.track import RepeatingFrameTrack


class FakeTrack:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int | None]] = []

    def push_frame(
        self,
        data: np.ndarray,
        *,
        capture_time_us: int | None = None,
    ) -> None:
        self.calls.append((data, capture_time_us))


class StaticCamera:
    def __init__(self, frame: CapturedFrame) -> None:
        self.frame = frame

    def read_rgb(self) -> CapturedFrame:
        return self.frame

    def close(self) -> None:
        pass


class RepeatingFrameTrackTest(unittest.TestCase):
    def test_repeated_pushes_keep_the_observation_capture_stamp(self) -> None:
        publisher = RepeatingFrameTrack("top", size=(2, 3))
        sdk_track = FakeTrack()
        frame = np.full((2, 3, 3), 7, dtype=np.uint8)

        publisher.bind(sdk_track)
        publisher.set_frame(frame, capture_time_us=123_456)
        publisher.push_current()
        publisher.push_current()

        self.assertEqual([stamp for _, stamp in sdk_track.calls], [123_456, 123_456])
        self.assertTrue(all(sent is frame for sent, _ in sdk_track.calls))

    def test_session_assigns_one_capture_stamp_to_every_view(self) -> None:
        session = object.__new__(ReactorSession)
        session.tracks = {
            "top": RepeatingFrameTrack("top", size=(2, 3)),
            "left": RepeatingFrameTrack("left", size=(2, 3)),
        }
        frames = {
            name: np.zeros((2, 3, 3), dtype=np.uint8) for name in session.tracks
        }

        with patch("reactor_sdk.time_micros", return_value=987_654):
            session.set_frames(frames)

        for publisher in session.tracks.values():
            sdk_track = FakeTrack()
            publisher.bind(sdk_track)
            publisher.push_current()
            self.assertEqual(sdk_track.calls[0][1], 987_654)


class CameraPublisherTest(unittest.IsolatedAsyncioTestCase):
    async def test_i2rt_publisher_rejects_nonpositive_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "poll_hz must be positive"):
            await publish_camera_frames({}, {}, poll_hz=0)

    async def test_i2rt_publisher_forwards_stamp_and_skips_duplicate_capture(self) -> None:
        data = np.zeros((2, 3, 3), dtype=np.uint8)
        source = StaticCamera(CapturedFrame(data, 42))
        track = FakeTrack()
        task = asyncio.create_task(
            publish_camera_frames({"top": track}, {"top": source}, poll_hz=1_000)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(len(track.calls), 1)
        self.assertIs(track.calls[0][0], data)
        self.assertEqual(track.calls[0][1], 42)


if __name__ == "__main__":
    unittest.main()
