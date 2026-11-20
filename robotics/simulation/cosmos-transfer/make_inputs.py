"""Create synthetic control fixtures; these are not physical simulation ground truth."""

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("inputs"))
    p.add_argument("--frames", type=int, default=121)
    args = p.parse_args()
    if not 1 <= args.frames <= 600:
        p.error("--frames must be between 1 and 600")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "depth").mkdir(exist_ok=True)
    (args.output / "metric-depth").mkdir(exist_ok=True)
    for control in ("depth", "edge", "blur", "seg"):
        with av.open(str(args.output / f"{control}.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 832, 480, "yuv420p"
            for index in range(args.frames):
                image = Image.new(
                    "RGB",
                    (832, 480),
                    (60, 60, 60) if control == "depth" else (80, 100, 110),
                )
                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    (0, 200, 832, 480),
                    fill=(140, 140, 140) if control == "depth" else (160, 120, 80),
                )
                draw.rectangle(
                    (220, 280, 330, 370),
                    fill=(200, 200, 200) if control == "depth" else (220, 40, 40),
                )
                draw.rectangle(
                    (500, 260, 600, 350),
                    fill=(180, 180, 180) if control == "depth" else (40, 70, 210),
                )
                x = int(350 + 50 * np.sin(index / 20))
                draw.rectangle(
                    (x, 120, x + 40, 280),
                    fill=(230, 230, 230) if control == "depth" else (190, 190, 190),
                )
                if control == "depth":
                    image.save(args.output / "depth" / f"{index:06d}.png")
                    # Synthetic metric fixture using the README's fixed 0.2–2m bounds.
                    depth_m = 2.0 - np.asarray(image)[..., 0].astype(np.float32) / 255 * 1.8
                    np.save(args.output / "metric-depth" / f"{index:06d}.npy", depth_m)
                elif control == "edge":
                    image = (
                        image.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
                    )
                elif control == "blur":
                    image = image.filter(ImageFilter.GaussianBlur(12))
                # seg retains flat, stable region colors; not a canonical dataset palette.
                frame = av.VideoFrame.from_ndarray(np.array(image), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    (args.output / "prompt.txt").write_text(
        "A grey robot gripper above a wooden tabletop with red and blue blocks, realistic indoor lighting."
    )
    print(
        f"Wrote four {args.frames}-frame controls, RGB depth PNGs, and prompt.txt to {args.output}"
    )


if __name__ == "__main__":
    main()
