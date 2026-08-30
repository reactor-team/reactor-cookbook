"""Drive the fasth3 clip queue end to end with the Reactor Python SDK.

Connects to a served fasth3 model, exercises the whole queue contract —
enqueue with metadata, watching the queue turn ready, playing a clip to the
end, the hold on black, playing a specific clip by UUID, and stopping one
mid-play — and writes everything received to disk: one .mp4 per clip (video
plus synchronized audio), the full message log, and a small timing report.

Run against a local `reactor run` (the default), or against a hosted session
with --api-key. Requires ffmpeg on PATH for the .mp4 encode.

Usage:
    python client.py                          # local runtime on :8080
    python client.py --api-key rk_...        # hosted session
    python client.py --seconds 8 --out ./out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
from reactor_sdk import Reactor

FPS = 24
AUDIO_RATE = 48_000

PROMPTS = [
    "A lighthouse on a rocky coast in heavy fog, slow cinematic drone orbit, waves crashing",
    "A neon-lit alley in the rain at night, puddles reflecting signs, slow dolly forward",
]


def payload(reply):
    """Unwrap a send_command reply envelope ({"type", "data"}) to its data."""
    if isinstance(reply, dict) and "data" in reply and "type" in reply:
        return reply["data"]
    return reply


class ClipCapture:
    """Frames and audio received while one clip plays."""

    def __init__(self, clip: dict):
        self.clip = clip
        self.frames: list[np.ndarray] = []
        self.audio: list[np.ndarray] = []
        self.first_frame_at: float | None = None


class Session:
    """Message log, per-clip media capture, and wait-for helpers."""

    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.messages: list[dict] = []
        self.log = (out_dir / "messages.jsonl").open("w")
        self.captures: dict[str, ClipCapture] = {}
        self.current: ClipCapture | None = None
        self.stray_frames = 0  # frames received while nothing should play

    def on_message(self, message):
        stamp = time.monotonic()
        kind = message.get("type") if isinstance(message, dict) else None
        data = message.get("data", {}) if isinstance(message, dict) else {}
        self.messages.append({"t": stamp, "type": kind, "data": data})
        self.log.write(json.dumps({"t": stamp, "type": kind, "data": data}) + "\n")
        self.log.flush()
        clip = data.get("clip") if isinstance(data, dict) else None
        print(f"  msg {kind}" + (f" clip={clip['clip_id'][:8]}" if isinstance(clip, dict) else ""))

        if kind == "clip_started":
            capture = ClipCapture(data["clip"])
            self.captures[data["clip"]["clip_id"]] = capture
            self.current = capture
        elif kind in ("clip_finished", "clip_stopped"):
            self.current = None

    async def wait_for(self, kind: str, timeout: float, predicate=None) -> dict:
        """Wait for the next `kind` message (optionally matching predicate)."""
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for message in self.messages[seen:]:
                seen += 1
                if message["type"] == kind and (predicate is None or predicate(message["data"])):
                    return message["data"]
            if time.monotonic() > deadline:
                raise TimeoutError(f"no {kind} within {timeout}s")
            await asyncio.sleep(0.05)

    def on_video(self, frame: np.ndarray):
        if self.current is not None:
            if self.current.first_frame_at is None:
                self.current.first_frame_at = time.monotonic()
            self.current.frames.append(np.asarray(frame))
        else:
            self.stray_frames += 1

    def on_audio(self, frame, sample_rate=AUDIO_RATE, num_channels=1):
        if self.current is not None:
            self.current.audio.append(np.asarray(frame).reshape(-1))


def save_clip(out: Path, name: str, capture: ClipCapture) -> str:
    """Encode one captured clip to .mp4 (h264 + aac) via ffmpeg."""
    if not capture.frames:
        return "no frames captured"
    height, width = capture.frames[0].shape[:2]
    wav_path = out / f"{name}.wav"
    samples = (
        np.concatenate(capture.audio) if capture.audio else np.zeros(1, dtype=np.int16)
    ).astype(np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_RATE)
        handle.writeframes(samples.tobytes())

    mp4_path = out / f"{name}.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(FPS), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(mp4_path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in capture.frames:
        encoder.stdin.write(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes())
    encoder.stdin.close()
    encoder.wait()
    seconds = len(capture.frames) / FPS
    return (
        f"{mp4_path} ({len(capture.frames)} frames, {seconds:.2f}s video, "
        f"{len(samples) / AUDIO_RATE:.2f}s audio)"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fasth3")
    parser.add_argument("--api-key", default=None, help="hosted session; default is local mode")
    parser.add_argument("--seconds", type=float, default=None, help="clip length to set")
    parser.add_argument("--out", default="./fasth3_out")
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(out)

    if args.api_key:
        reactor = Reactor(args.model, api_key=args.api_key)
    else:
        reactor = Reactor(args.model, local=True)
    reactor.on("message", session.on_message)

    print("connecting ...")
    started = time.monotonic()
    await reactor.connect()
    print(f"status={reactor.status} after {time.monotonic() - started:.1f}s "
          f"session={reactor.session_id}")

    try:
        video = reactor.tracks.with_direction("recvonly").with_kind("video").one()
        audio = reactor.tracks.with_direction("recvonly").with_kind("audio").one()
        video.on_frame(session.on_video)
        audio.on_frame(session.on_audio)

        state = payload(await reactor.send_command("get_state", {}))
        print("state:", json.dumps(state))

        if args.seconds:
            reply = payload(
                await reactor.send_command("set_clip_seconds", {"seconds": args.seconds})
            )
            print("clip length accepted:", reply)

        # --- enqueue two clips, metadata carrying this client's own record ---
        clips = []
        for index, prompt in enumerate(PROMPTS):
            reply = payload(await reactor.send_command(
                "enqueue",
                {"prompt": prompt, "metadata": json.dumps({"requested_by": "queue-client", "n": index})},
            ))
            clip = reply["clip"]
            clips.append({"info": clip, "enqueued_at": time.monotonic()})
            print(f"enqueued {clip['clip_id'][:8]} frames={clip['frames']} "
                  f"seed={clip['seed']} ready={clip['ready']}")

        first_id = clips[0]["info"]["clip_id"]
        second_id = clips[1]["info"]["clip_id"]

        # --- wait for clip 1 to build, then play it to the end ---------------
        await session.wait_for(
            "queue_update",
            timeout=args.ready_timeout,
            predicate=lambda d: any(c["clip_id"] == first_id and c["ready"] for c in d["clips"]),
        )
        ready_after = time.monotonic() - clips[0]["enqueued_at"]
        print(f"clip 1 ready after {ready_after:.1f}s")

        play_at = time.monotonic()
        await reactor.send_command("play", {})
        started_data = await session.wait_for("clip_started", timeout=30)
        assert started_data["clip"]["clip_id"] == first_id, "play must take the oldest ready clip"
        finished = await session.wait_for(
            "clip_finished", timeout=clips[0]["info"]["seconds"] + 60
        )
        print(f"clip 1 finished; seconds_sent={finished['seconds_sent']}")

        # --- the stream holds on black: nothing may play on its own ----------
        stray_before = session.stray_frames
        await asyncio.sleep(2.0)
        print(f"frames while idle: {session.stray_frames - stray_before} (flush tail is tolerated)")

        # --- play clip 2 by UUID, then stop it mid-play -----------------------
        await session.wait_for(
            "queue_update",
            timeout=args.ready_timeout,
            predicate=lambda d: any(c["clip_id"] == second_id and c["ready"] for c in d["clips"]),
        )
        await reactor.send_command("play", {"clip_id": second_id})
        await session.wait_for(
            "clip_started", timeout=30, predicate=lambda d: d["clip"]["clip_id"] == second_id
        )
        await asyncio.sleep(4.0)
        await reactor.send_command("stop", {})
        stopped = await session.wait_for("clip_stopped", timeout=15)
        print(f"clip 2 stopped mid-play; seconds_sent={stopped['seconds_sent']}")

        queue = payload(await reactor.send_command("get_queue", {}))
        print("queue after run:", json.dumps(queue))

        # --- save what was received ------------------------------------------
        for index, clip_id in enumerate((first_id, second_id), start=1):
            capture = session.captures.get(clip_id)
            if capture:
                ttff = (
                    capture.first_frame_at - play_at
                    if index == 1 and capture.first_frame_at
                    else None
                )
                print(f"clip{index}: {save_clip(out, f'clip{index}', capture)}"
                      + (f" ttff={ttff:.2f}s" if ttff else ""))
        report = {
            "clip1_ready_after_s": round(ready_after, 2),
            "clip1_frames": len(session.captures[first_id].frames)
            if first_id in session.captures else 0,
            "clip2_frames": len(session.captures[second_id].frames)
            if second_id in session.captures else 0,
            "stray_frames_while_idle": session.stray_frames,
            "messages": len(session.messages),
        }
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print("report:", json.dumps(report))
    finally:
        await reactor.disconnect()
        session.log.close()


if __name__ == "__main__":
    asyncio.run(main())
