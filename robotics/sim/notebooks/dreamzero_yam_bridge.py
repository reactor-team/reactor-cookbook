"""Minimal robot bridge for ``reactor/dreamzero-yam-molmoact2``.

Streams three camera views and both arms' measured state. Replace
``CameraTrack.read_frame``, ``RobotInterface.get_state``, and
``RobotInterface.execute_chunk`` to connect real hardware. See
``dreamzero_yam_bridge.md`` for the wire contract and safety boundary.
"""

import asyncio
import os
from typing import Any

import numpy as np
from reactor_sdk import Reactor, ReactorStatus, time_micros

MODEL = "reactor/dreamzero-yam-molmoact2"
# Model >= 1.1.0: ask the server to align the three cameras' frames by their
# capture stamps for this session (default off server-side; changes model
# inputs). Older deployments reject the unknown command.
PAIR_BY_CAPTURE_TIME = os.environ.get("PAIR_BY_CAPTURE_TIME", "") == "1"
API_URL = os.environ.get("REACTOR_API_URL", "https://api.reactor.inc")
CAMERAS = ["top", "left", "right"]  # model expects exactly these three views
STATE_HZ = 10.0
CAMERA_HZ = 15.0


class CameraSource:
    """Captures one camera view and its sender-clock timestamp.

    Replace ``read_frame`` with your camera driver. Return the timestamp taken
    with ``reactor_sdk.time_micros()`` as close as possible to sensor capture.
    The model was trained on 176x320 RGB; sending at/near that resolution
    avoids wasting bandwidth.
    """

    def __init__(self, camera_name: str):
        self.camera_name = camera_name

    def read_frame(self) -> tuple[np.ndarray, int]:
        # TODO: return the latest RGB frame from your camera as HxWx3 uint8.
        frame = np.zeros((176, 320, 3), dtype=np.uint8)
        return frame, time_micros()


async def publish_cameras(
    tracks: dict[str, Any],
    sources: dict[str, CameraSource],
) -> None:
    """Push captured frames with their sender-authored capture timestamps."""
    period = 1.0 / CAMERA_HZ
    while True:
        started = asyncio.get_running_loop().time()
        for name, source in sources.items():
            frame, capture_time_us = source.read_frame()
            tracks[name].push_frame(
                frame,
                capture_time_us=capture_time_us,
            )
        await asyncio.sleep(
            max(0.0, period - (asyncio.get_running_loop().time() - started))
        )


class RobotInterface:
    """Replace with your robot driver."""

    def get_state(self) -> dict:
        # TODO: return measured joint positions (rad) and grippers (0..1).
        return {
            "left_joint_pos": [0.0] * 6,
            "left_gripper_pos": 0.0,
            "right_joint_pos": [0.0] * 6,
            "right_gripper_pos": 0.0,
        }

    def execute_chunk(self, actions: np.ndarray) -> None:
        # TODO: replace the pending action buffer with `actions` (24x14) and
        # feed it to your controller. Columns: 0-5 left joints (absolute rad
        # targets when you stream state), 6 left gripper, 7-12 right joints,
        # 13 right gripper. Blend / rate-limit at the seam with the chunk
        # you were executing.
        pass


async def main() -> None:
    robot = RobotInterface()
    ready = asyncio.Event()
    camera_sources = {name: CameraSource(name) for name in CAMERAS}
    camera_task: asyncio.Task | None = None
    connected = False

    reactor = Reactor(MODEL, api_key=os.environ["REACTOR_API_KEY"], api_url=API_URL)

    @reactor.on_status
    def on_status(status):
        print(f"status: {status}")
        if status == ReactorStatus.READY:
            ready.set()

    @reactor.on_message
    def on_message(message):
        data = message.get("data", {}) if isinstance(message, dict) else {}
        if message.get("type") == "action_chunk":
            actions = np.asarray(data["actions"], dtype=np.float64)  # (24, 14)
            robot.execute_chunk(actions)
            skew = data.get("view_skew_us")  # model >= 1.1.0
            print(
                f"chunk {data['chunk_index']}: inference {data['inference_seconds']:.3f}s"
                + (f" | view_skew {skew}us" if skew is not None else "")
            )
        else:
            print(f"message: {message}")

    try:
        await reactor.connect()
        connected = True
        await asyncio.wait_for(ready.wait(), timeout=120)

        tracks = {name: await reactor.publish_track(name) for name in CAMERAS}
        camera_task = asyncio.create_task(publish_cameras(tracks, camera_sources))

        # Set the task once; the episode starts when prompt + state are in.
        await reactor.send_command(
            "set_prompt", {"prompt": "fold the towel neatly with both arms"}
        )
        if PAIR_BY_CAPTURE_TIME:
            await reactor.send_command(
                "set_pair_by_capture_time", {"pair_by_capture_time": True}
            )

        # Closed loop: stream measured state continuously. This is what paces
        # the model — every chunk is conditioned on the latest state.
        while True:
            state = robot.get_state()
            await reactor.send_command("set_left_joint_pos", {"left_joint_pos": state["left_joint_pos"]})
            await reactor.send_command("set_left_gripper_pos", {"left_gripper_pos": state["left_gripper_pos"]})
            await reactor.send_command("set_right_joint_pos", {"right_joint_pos": state["right_joint_pos"]})
            await reactor.send_command("set_right_gripper_pos", {"right_gripper_pos": state["right_gripper_pos"]})
            await asyncio.sleep(1.0 / STATE_HZ)
    finally:
        if camera_task is not None:
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                pass
        if connected:
            await reactor.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
