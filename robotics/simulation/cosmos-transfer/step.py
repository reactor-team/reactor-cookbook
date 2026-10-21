"""Render prepared depth PNGs through the stock Reactor Python SDK.

This replays a fixture, not a simulator/policy loop. A simulator integration
would prepare each next four-frame input only after consuming the prior result.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

from PIL import Image
from reactor_sdk import Reactor


async def run(args):
    paths = sorted(args.depth_dir.glob("*.png"))
    if args.frames > len(paths) or not 1 <= args.frames <= 597 or (args.frames - 1) % 4:
        raise ValueError("Provide at least --frames PNGs; --frames must equal 1 + 4N.")
    args.output.mkdir(parents=True, exist_ok=True)
    client = Reactor(
        model_name=args.model,
        api_url=args.url,
        local=args.local,
        api_key=os.environ.get("REACTOR_API_KEY"),
        max_session_duration_seconds=600,
    )
    results = asyncio.Queue()

    @client.on_message
    def on_message(message):
        if message.get("type") in ("step_result", "step_failed"):
            results.put_nowait(message)

    async def receive(episode_id, step_index):
        async with asyncio.timeout(args.timeout):
            while True:
                message = await results.get()
                data = message["data"]
                if data["episode_id"] != episode_id or data["step_index"] != step_index:
                    continue
                if message["type"] == "step_failed":
                    raise RuntimeError(data)
                return data

    def encode(path):
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"{path}: use RGB PNG depth visualizations.")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    timings = []
    async with client:
        await client.connect()
        try:
            for index in range(1 + (args.frames - 1) // 4):
                start = 0 if index == 0 else 1 + (index - 1) * 4
                count = 1 if index == 0 else 4
                encoded = [encode(path) for path in paths[start : start + count]]
                started = time.monotonic()
                if index == 0:
                    accepted = await client.send_command(
                        "initialize",
                        {
                            "depth_png_b64": encoded[0],
                            "prompt": args.prompt.read_text(),
                            "total_frames": args.frames,
                            "aspect_ratio": args.aspect_ratio,
                            "seed": args.seed,
                            "preview": args.preview,
                        },
                    )
                    episode_id = accepted["data"]["episode_id"]
                else:
                    await client.send_command(
                        "advance",
                        {
                            "episode_id": episode_id,
                            "step_index": index,
                            "depth_png_b64": encoded,
                        },
                    )
                result = await receive(episode_id, index)
                if (
                    result["first_frame"] != start
                    or len(result["frames_png_b64"]) != count
                ):
                    raise ValueError("Unexpected result frame range")
                for offset, (encoded_png, expected) in enumerate(
                    zip(
                        result["frames_png_b64"],
                        result["sha256"],
                        strict=True,
                    )
                ):
                    raw = base64.b64decode(encoded_png, validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        actual = hashlib.sha256(
                            image.convert("RGB").tobytes()
                        ).hexdigest()
                    if actual != expected:
                        raise ValueError("Returned RGB frame hash mismatch.")
                    (args.output / f"{start + offset:06d}.png").write_bytes(raw)
                timing = {
                    "step_index": index,
                    "first_frame": start,
                    "frames": count,
                    "seconds": time.monotonic() - started,
                }
                timings.append(timing)
                print(json.dumps(timing), flush=True)
            if not result["complete"]:
                raise ValueError("Final result did not complete the episode")
        finally:
            await client.disconnect()
    (args.output / "timings.json").write_text(json.dumps(timings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Lexically ordered RGB PNGs at the selected output dimensions.",
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aspect-ratio", default="16,9")
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--url", default="https://api.reactor.inc")
    parser.add_argument("--model", default="cosmos3-nano-sim-transfer")
    parser.add_argument("--timeout", type=float, default=180)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
