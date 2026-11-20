"""Receive real video through the Reactor SDK and verify camera/reset behavior."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import threading

import av
import numpy as np
from reactor_sdk import Reactor


class FrameLedger:
    """Account for actual tagged media independently of completion messages.

    Runtime boundary frames have no tags. Gap-fill repeats retain their tags
    and count once. Pixel brightness is never an identity or validity test.
    """

    def __init__(self):
        self.frames = {}
        self.expected = {}
        self.untagged = 0
        self.repeats = 0
        self.lock = threading.Lock()

    def expect(self, message):
        key = (int(message["world_id"]), int(message["chunk_index"]))
        count = int(message["frames"])
        if count != 33:
            raise ValueError(f"Unexpected chunk frame count: {count}")
        with self.lock:
            self.expected[key] = count
            self._check_indices(key)
        return key

    def receive(self, frame, user_data):
        with self.lock:
            if not user_data:
                self.untagged += 1
                return
            tag = json.loads(user_data) if isinstance(user_data, (bytes, str)) else user_data
            values = [tag[name] for name in ("world_id", "chunk_index", "frame_index")]
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
                raise ValueError("Invalid frame identity")
            key = tuple(values)
            if frame.shape != (384, 640, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Invalid model frame: {frame.shape}, {frame.dtype}")
            if not 0 <= key[2] < 33:
                raise ValueError("Frame index outside chunk")
            if key in self.frames:
                self.repeats += 1
            else:
                self.frames[key] = frame.copy()

    def _check_indices(self, key):
        found = {k[2] for k in self.frames if k[:2] == key}
        if found - set(range(self.expected[key])):
            raise ValueError("Unexpected frame index")
        return found == set(range(self.expected[key]))

    def complete(self, key):
        with self.lock:
            return key in self.expected and self._check_indices(key)

    def report(self):
        with self.lock:
            required = {(w, c, f) for (w, c), n in self.expected.items() for f in range(n)}
            return dict(expected_frames=len(required), received_frames=len(self.frames),
                        missing=sorted(required - self.frames.keys()),
                        unexpected=sorted(self.frames.keys() - required),
                        untagged_frames=self.untagged, repeated_frames=self.repeats)

    def finish(self):
        with self.lock:
            required = {(w, c, f) for (w, c), n in self.expected.items() for f in range(n)}
            missing = required - self.frames.keys()
            unexpected = self.frames.keys() - required
            if not required or missing or unexpected:
                raise AssertionError(f"Incomplete media: {len(missing)} missing, {len(unexpected)} unexpected")
            keys = sorted(required)
            return [self.frames[k] for k in keys], dict(
                expected_frames=len(required), received_frames=len(self.frames),
                untagged_frames=self.untagged, repeated_frames=self.repeats,
                chunks=[dict(world_id=w, chunk_index=c, expected_frames=n,
                             received_frames=sum(k[:2] == (w, c) for k in self.frames))
                        for (w, c), n in self.expected.items()],
            )


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(dict(status="running")))
    messages = []
    ledger = FrameLedger()
    errors = []
    client = Reactor("worldcrafter-fast", local=True, api_url=args.url)
    viewer = None

    def message(value):
        messages.append(value)
        print("MESSAGE " + json.dumps(value), flush=True)

    client.on("message", message)
    client.on("error", lambda error: errors.append(str(error)))

    @client.track("main_video").on_frame
    def on_frame(frame, frame_id, timestamp_us, user_data):
        try:
            ledger.receive(frame, user_data)
        except Exception as exc:
            errors.append(str(exc))

    async def wait_chunk(world_id, index):
        deadline = time.monotonic() + args.timeout
        media_deadline = None
        while time.monotonic() < deadline:
            for value in messages:
                payload = value.get("data", value)
                if (payload.get("world_id") == world_id and payload.get("chunk_index") == index
                        and "camera_end" in payload):
                    key = ledger.expect(payload)
                    if ledger.complete(key):
                        return payload
                    if media_deadline is None:
                        media_deadline = time.monotonic() + args.media_timeout
            if media_deadline is not None and time.monotonic() >= media_deadline:
                raise TimeoutError(f"Incomplete media for chunk {world_id}/{index}: {ledger.report()}")
            if errors:
                raise RuntimeError(errors[-1])
            await asyncio.sleep(0.25)
        raise TimeoutError(f"No completed chunk {world_id}/{index}")

    try:
        await asyncio.wait_for(client.connect(), 90)
        upload = await client.upload_file(args.image)
        response = await client.send_command("start_world", dict(image=upload,
                prompt=args.prompt.read_text().strip(), seed=42, gravity_align=True))
        print("START " + json.dumps(response), flush=True)
        first = await wait_chunk(1, 0)
        route = [("forward", 1.0), ("yaw_left", 20.0), ("yaw_right", 20.0)] if args.full else [("yaw_left", 20.0)]
        chunks = [first]
        for index, (action, amount) in enumerate(route, 1):
            await client.send_command("move", dict(action=action, amount=amount, chunks=1))
            chunks.append(await wait_chunk(1, index))
        if args.multi_viewer:
            viewer = Reactor("worldcrafter-fast", local=True, api_url=args.url)
            await viewer.connect(session_id=client.session_id)
            await client.send_command("pause", {})
            await client.send_command("move", dict(action="forward", amount=.25, chunks=4))
            await client.send_command("resume", {})
            await viewer.disconnect()
            viewer.close()
            viewer = None
            for index in range(len(chunks), len(chunks) + 4):
                chunks.append(await wait_chunk(1, index))
        await client.send_command("reset", {})
        reset = await wait_chunk(2, 0)
        assert np.allclose(first["camera_end"], reset["camera_end"], atol=1e-6)
        assert not np.allclose(first["camera_end"], chunks[-1]["camera_end"], atol=1e-4)
        assert all(item["gpu_count"] == args.gpus for item in chunks + [reset])
        frames, media = ledger.finish()
        print("MEDIA", json.dumps(media), flush=True)
        print("STATS", await client.get_stats(), flush=True)
        assert len(frames) == 33 * (len(chunks) + 1)
        video = args.output / "reactor-client.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=16)
            stream.width, stream.height, stream.pix_fmt = 640, 384, "yuv420p"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == len(frames)
        from PIL import Image
        for name, frame in (("first", frames[0]), ("last", frames[-1])):
            Image.fromarray(frame).save(args.output / f"{name}.png")
        report = dict(status="passed", gpu_count=args.gpus, received_frames=len(frames), media=media,
                      decoded_frames=decoded, chunks=chunks, reset=reset,
                      camera_changes_verified=True, reset_pose_verified=True,
                      multi_viewer_verified=args.multi_viewer,
                      transport="Reactor SDK 1.10.0 / WebRTC", video=str(video))
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
        print("VALIDATION " + json.dumps(report), flush=True)
    except Exception as exc:
        (args.output / "validation.json").write_text(json.dumps(
            dict(status="failed", error=str(exc), media=ledger.report()), indent=2))
        raise
    finally:
        (args.output / "media.json").write_text(json.dumps(ledger.report(), indent=2))
        (args.output / "messages.json").write_text(json.dumps(messages, indent=2))
        if viewer is not None:
            await viewer.disconnect()
            viewer.close()
        await client.disconnect()
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--media-timeout", type=float, default=30,
                        help="Seconds to receive all frames after a chunk completion message.")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--multi-viewer", action="store_true", help="Verify accepted motion survives a second viewer disconnecting.")
    asyncio.run(main(parser.parse_args()))
