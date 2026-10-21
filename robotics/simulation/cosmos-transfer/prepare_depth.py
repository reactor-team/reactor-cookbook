"""Convert a sequence of metric camera-depth .npy arrays to RGB8 PNGs and MP4."""

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image

CANVASES = {(832, 480), (736, 544), (640, 640), (544, 736), (480, 832)}


def depth_to_rgb(depth, near, far):
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("Require finite 0 < near < far in meters")
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("Expected a 2D floating-point camera-depth array in meters")
    # Fixed bounds across the whole episode: near is white, far is black.
    valid = np.isfinite(depth) & (depth > 0)
    clipped = np.clip(np.where(valid, depth, far), near, far)
    gray = np.rint(255 * (far - clipped) / (far - near)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near", type=float, required=True, help="Near bound in meters")
    parser.add_argument("--far", type=float, required=True, help="Far bound in meters")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.npy"))
    if not 1 <= len(files) <= 600:
        parser.error("Provide 1–600 lexically ordered .npy frames")
    if args.output.exists():
        parser.error("Use a new output directory to avoid mixing episodes")
    first = depth_to_rgb(np.load(files[0], allow_pickle=False), args.near, args.far)
    height, width = first.shape[:2]
    if (width, height) not in CANVASES:
        parser.error("Render depth at a supported canvas size; see README")
    args.output.mkdir(parents=True)
    png_dir = args.output / "depth"
    png_dir.mkdir()
    with av.open(str(args.output / "depth.mp4"), "w") as video:
        stream = video.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index, path in enumerate(files):
            rgb = depth_to_rgb(np.load(path, allow_pickle=False), args.near, args.far)
            if rgb.shape != first.shape:
                raise ValueError(f"Depth dimensions changed at {path}")
            Image.fromarray(rgb).save(png_dir / f"{index:06d}.png")
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    print(f"Saved {len(files)} RGB depth PNGs and 30 FPS depth.mp4 to {args.output}")


if __name__ == "__main__":
    main()
