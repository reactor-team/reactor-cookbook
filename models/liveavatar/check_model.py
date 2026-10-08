"""Upload explicit user-provided inputs, receive real SDK tracks and save a take."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import soundfile as sf
from reactor_sdk import Reactor


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    audio = []
    audio_formats = set()
    ended = asyncio.Event()
    failure = []
    messages = []
    async with Reactor(
        model_name="liveavatar", local=True, api_url=args.url
    ) as reactor:

        @reactor.track("main_video").on_frame
        async def video_frame(frame):
            frames.append(frame.copy())

        @reactor.track("main_audio").on_frame
        async def audio_frame(frame, sample_rate, num_channels):
            audio.append(frame.copy())
            audio_formats.add((sample_rate, num_channels))

        @reactor.on_message
        def message(msg):
            messages.append(msg)
            print(json.dumps(msg, ensure_ascii=False), flush=True)
            if msg["type"] == "generation_ended":
                if msg["data"]["reason"] != "complete":
                    failure.append(msg["data"]["reason"])
                ended.set()

        await asyncio.wait_for(reactor.connect(), 60)
        with args.image.open("rb") as uploaded:
            image = await reactor.upload_file(uploaded)
        await reactor.send_command("set_avatar_image", {"image": image})
        with args.audio.open("rb") as uploaded:
            driving_audio = await reactor.upload_file(uploaded)
        await reactor.send_command("set_audio", {"audio": driving_audio})
        await reactor.send_command(
            "set_prompt", {"prompt": args.prompt, "negative_prompt": ""}
        )
        await reactor.send_command(
            "set_generation_options", {"seed": args.seed, "max_chunks": args.chunks}
        )
        await reactor.send_command("start", {})
        await asyncio.wait_for(ended.wait(), args.timeout)
        await asyncio.sleep(args.chunks * 48 / 25 + 2)
        await reactor.send_command("stop", {})
    (args.output / "messages.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False)
    )
    if failure:
        raise RuntimeError("; ".join(failure))
    if not frames or not audio:
        raise RuntimeError(
            f"Missing SDK tracks: {len(frames)} video frames, {len(audio)} audio packets"
        )
    imageio.mimwrite(args.output / "video.mp4", frames, fps=25, macro_block_size=1)
    if len(audio_formats) != 1:
        raise RuntimeError(f"Audio format changed during take: {audio_formats}")
    rate, channels = next(iter(audio_formats))
    pcm = np.concatenate(audio, axis=0).reshape(-1, channels)
    sf.write(args.output / "audio.wav", pcm, rate)
    # Keep received PCM (including transport underrun silence) separately.
    # The gapless video uses model time, so pair it with the driving audio as
    # the upstream offline exporter does, not wall-clock transport silence.
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(args.output / "video.mp4"),
            "-i",
            str(args.audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-shortest",
            str(args.output / "take.mp4"),
        ],
        check=True,
    )
    print(f"Saved {len(frames)} received video frames to {args.output / 'take.mp4'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--url", default="http://127.0.0.1:8791")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
