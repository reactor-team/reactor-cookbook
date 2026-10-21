"""Upload prepared control video, render, and save a sampled preview and metadata."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import av
from PIL import Image
from reactor_sdk import Reactor

CANVASES = {
    "16,9": (832, 480),
    "4,3": (736, 544),
    "1,1": (640, 640),
    "3,4": (544, 736),
    "9,16": (480, 832),
}


def validate(args):
    if args.control.suffix.lower() not in (".mp4", ".webm", ".mov"):
        raise ValueError("Use a video file with a recognized video extension")
    if args.control.stat().st_size > 128 * 1024**2:
        raise ValueError("Upload exceeds 128 MiB")
    if not 1 <= args.frames <= 600 or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Invalid frame count or seed")
    with av.open(str(args.control)) as source:
        if not source.streams.video:
            raise ValueError("No video stream")
        track = source.streams.video[0]
        if track.average_rate is None or abs(float(track.average_rate) - 30) > 0.1:
            raise ValueError("Prepare 30 or 29.97 FPS video")
        count = 0
        for frame in source.decode(track):
            count += 1
            if count >= args.frames:
                break
        if count < args.frames:
            raise ValueError("Not enough source frames")


async def run(args):
    validate(args)
    caption = args.prompt.read_text().strip()
    if not caption:
        raise ValueError("Prompt must not be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    client = Reactor(
        model_name="cosmos3-nano-sim-transfer",
        api_key=os.environ["REACTOR_API_KEY"],
        max_session_duration_seconds=300,
    )
    ready, terminal = asyncio.Event(), asyncio.Queue()
    received = 0
    latest = None
    first_frame = None
    started = None

    @client.on_message
    def message(value):
        kind, data = value.get("type"), value.get("data", {})
        if (
            kind == "state_update"
            and data.get("has_source")
            and not data.get("generating")
        ):
            ready.set()
        if kind in ("done", "failed"):
            terminal.put_nowait(value)

    async with client:
        await client.connect()

        @client.track("main_video").on_raw_frame
        def frame(data, width, height, *_):
            nonlocal received, latest, first_frame
            if started is None or (width, height) != CANVASES[args.aspect_ratio]:
                return
            if len(data) != width * height * 4:
                return
            received += 1
            latest = bytes(data), width, height
            if first_frame is None:
                first_frame = time.monotonic() - started

        await client.send_command("reset", {})
        ready.clear()
        await client.upload_file(str(args.control))
        # FileRef confirms transport only. Require fresh model acceptance.
        await asyncio.wait_for(ready.wait(), 30)
        started = time.monotonic()
        await client.send_command(
            "generate",
            {
                "prompt": caption,
                "control": args.kind,
                "num_frames": args.frames,
                "aspect_ratio": args.aspect_ratio,
                "seed": args.seed,
            },
        )
        result = await asyncio.wait_for(terminal.get(), 240)
        elapsed = time.monotonic() - started
        if result["type"] == "failed":
            raise RuntimeError(result["data"])
        if result["data"]["num_frames"] != args.frames:
            raise RuntimeError("Unexpected generated frame count")
        # done means queued, not fully received. This is a bounded preview sample.
        await asyncio.sleep(5)
        if latest is None:
            raise RuntimeError("No preview received")
        pixels, width, height = latest
        Image.frombytes("RGBA", (width, height), pixels, "raw", "BGRA").convert(
            "RGB"
        ).save(args.output / "preview.png")
        report = {
            "generated_frames": args.frames,
            "received_preview_frames": received,
            "first_preview_seconds": first_frame,
            "done_seconds": elapsed,
            "control": args.kind,
            "aspect_ratio": args.aspect_ratio,
        }
        await save_report_and_reset(client, args.output, report)
    print(json.dumps(report))


async def save_report_and_reset(client, output, report):
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    try:
        await asyncio.wait_for(client.send_command("reset", {}), 10)
    except Exception as exc:
        # Rendering succeeded; a cleanup failure must not erase that evidence.
        report["cleanup_warning"] = f"{type(exc).__name__}: {exc}"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Saved render; cleanup reset failed: {exc}", file=sys.stderr)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--kind", choices=("depth", "edge", "blur", "seg"), default="depth")
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--aspect-ratio", choices=CANVASES, default="16,9")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("output/render"))
    asyncio.run(run(p.parse_args()))
