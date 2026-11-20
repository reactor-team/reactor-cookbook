"""Measure LiveAvatar's block time on one GPU, offline, against real time.

Drives the model half block by block from a portrait and a speech file, with no
runtime and no client, and prints one JSON line with the time each stage took
per block. A block is 12 frames, 0.48 s of video: the model keeps real time
while a block renders in less than that.

    python bench.py --weights /path/to/weights --portrait face.png --audio speech.wav \
        --blocks 40 --compile --clip out.mp4
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Folder with wan2_2/ and liveavatar_lora/.",
    )
    parser.add_argument("--source", type=Path, default=Path("/opt/liveavatar-source"))
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument(
        "--audio", type=Path, required=True, help="Speech, any format ffmpeg reads."
    )
    parser.add_argument(
        "--blocks", type=int, default=40, help="Blocks to time after the warm-up."
    )
    parser.add_argument(
        "--warmup", type=int, default=8, help="Blocks rendered before timing starts."
    )
    parser.add_argument(
        "--silent", type=int, default=0, help="Silent blocks appended after the speech."
    )
    parser.add_argument(
        "--size", default="512x512", help="WIDTHxHEIGHT, multiples of 64."
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--fp8", action="store_true", help="FP8 projections in the transformer blocks."
    )
    parser.add_argument(
        "--clip", type=Path, help="Write the timed blocks as an MP4 with the speech."
    )
    args = parser.parse_args()

    from liveavatar_model import LiveAvatarModel, crop_portrait, read_audio

    width, height = (int(v) for v in args.size.lower().split("x"))
    model = LiveAvatarModel()
    model.load(
        args.weights,
        source=args.source,
        width=width,
        height=height,
        compile_dit=args.compile,
        fp8=args.fp8,
        warmup_blocks=0,
    )
    engine = model.engine
    portrait = crop_portrait(args.portrait.read_bytes(), width, height)
    engine.bind(portrait)

    speech = read_audio(args.audio)
    block = engine.frames_for_block(1) * 640
    total = args.warmup + args.blocks
    speech = np.concatenate(
        [speech, np.zeros(max(0, total * block - speech.size), np.float32)]
    )
    script = ["V"] * (total - args.silent) + ["S"] * args.silent
    rows, frames, played = [], [], []
    for n, kind in enumerate(script):
        audio = (
            speech[n * block : (n + 1) * block]
            if kind == "V"
            else np.zeros(block, np.float32)
        )
        out = engine.render(audio)
        if n >= args.warmup:
            t = engine.timing
            rows.append(
                {
                    "kind": kind,
                    "audio": t.audio_ms,
                    "dit": t.dit_ms,
                    "decode": t.decode_ms,
                    "total": t.total_ms,
                }
            )
            frames.append(out)
            played.append(audio[-out.shape[0] * 640 :])

    def stat(key: str) -> dict[str, float]:
        values = [r[key] for r in rows]
        return {
            "median": round(statistics.median(values), 1),
            "p90": round(float(np.percentile(values, 90)), 1),
            "max": round(max(values), 1),
        }

    budget_ms = 1000 * engine.frames_for_block(1) / 25
    total_med = statistics.median(r["total"] for r in rows)
    print(
        json.dumps(
            {
                "size": f"{width}x{height}",
                "compile": args.compile,
                "fp8": args.fp8,
                "cond_tokens": engine.fast.cond_tokens,
                "blocks": len(rows),
                "budget_ms": budget_ms,
                "audio_ms": stat("audio"),
                "dit_ms": stat("dit"),
                "step_ms_median": round(
                    statistics.median(r["dit"] for r in rows) / 4, 1
                ),
                "decode_ms": stat("decode"),
                "total_ms": stat("total"),
                "rtf": round(total_med / budget_ms, 3),
            }
        )
    )
    if args.clip:
        write_clip(args.clip, np.concatenate(frames), np.concatenate(played))


def write_clip(path: Path, frames: np.ndarray, audio16k: np.ndarray) -> None:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        sf.write(wav, audio16k, 16000)
        h, w = frames.shape[1:3]
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
            "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ]  # fmt: skip
        subprocess.run(cmd, input=frames.tobytes(), check=True)


if __name__ == "__main__":
    main()
