"""Python 3.11 inference worker; rank zero serves bounded local RPC requests."""
import argparse
import io
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback

WARMUP_PROMPT = "A first-person view of a quiet, softly lit courtyard."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    parser.add_argument("--warmup-chunks", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size not in (1, 8):
        raise ValueError("Expected one or eight inference ranks")
    if size == 8:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        if len(visible) != 8:
            raise ValueError("Eight visible GPUs are required")
        # Rank zero's VAE pipeline spans all GPUs; peer DiTs each own cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible) if rank == 0 else visible[rank]
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            os.environ[key] = str(Path(os.environ.get(key, args.root / key.lower())) / f"rank_{rank}")
        if rank:
            from demo.multigpu.serving import serve_peer
            serve_peer()
            return

    import numpy as np
    from PIL import Image, ImageOps
    if size == 8:
        from demo.multigpu.serving import ContextParallelEngine
        engine = ContextParallelEngine()
    else:
        from demo.engine import WorldCrafterEngine
        engine = WorldCrafterEngine()
    world_id = None
    gravity_aligned = False
    up = None
    cancel = threading.Event()
    warm_up(engine, args.warmup_chunks, args.max_chunks, cancel)
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(args.socket)
    stream = connection.makefile("rwb")

    def respond(value):
        stream.write((json.dumps(value) + "\n").encode())
        stream.flush()

    respond(dict(gpu_count=size, engine=engine.info))
    try:
        for line in stream:
            request = json.loads(line)
            identity = request["id"]
            try:
                if request["command"] == "reset":
                    engine.close_session()
                    world_id = None
                    gravity_aligned = False
                    up = None
                    respond(dict(id=identity))
                    continue
                if request["command"] != "generate":
                    raise ValueError("Unknown worker command")
                if request["world_id"] != world_id:
                    if not request["has_image"]:
                        raise ValueError("A new world requires an anchor image")
                    up = None
                    with Image.open(args.root / "anchor.image") as source:
                        image = native_image(ImageOps.exif_transpose(source).convert("RGB"))
                    if request["gravity_align"]:
                        from worldcrafter.calibration import calibrate_image
                        up = calibrate_image(image)["world_up"]
                    anchor = generation_image(image)
                    engine.initialize_session(anchor, request["prompt"],
                                              request["seed"], args.max_chunks, cancel)
                    world_id = request["world_id"]
                    gravity_aligned = up is not None
                if engine.index >= args.max_chunks:
                    raise ValueError("World chunk limit reached")
                with np.load(args.root / "poses.npz", allow_pickle=False) as poses:
                    if "path" in poses:
                        local, world = path_poses(poses["path"], up, engine.index)
                    else:
                        local, world = poses["local"], poses["world"]
                frames = 33 * (engine.index + 1)
                for value in (local, world):
                    if value.ndim != 3 or len(value) != frames or value.shape[1:] not in ((3, 4), (4, 4)) \
                            or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError("Invalid cumulative camera poses")
                result = engine.next_chunk(None, poses=(local, world))
                np.save(args.root / "frames.npy", result["frames"], allow_pickle=False)
                reply = dict(id=identity, world_id=world_id, chunk_index=result["chunk_index"],
                             camera_end=result["global_pose"][-1][:3].tolist(),
                             gravity_aligned=gravity_aligned, gpu_count=size,
                             world_up=up)
                print("REACTOR_CHUNK " + json.dumps(dict(reply,
                      engine_mode=engine.info["gpu_mode"], rgb_sha256=result["rgb_sha256"],
                      memory=result["memory"])), flush=True)
                respond(reply)
            except Exception as exc:
                traceback.print_exc()
                respond(dict(id=identity, error=f"{type(exc).__name__}: {exc}"))
                break
    finally:
        engine.close_session()
        if size == 8:
            engine.shutdown()
        stream.close()
        connection.close()


def warm_up(engine, chunks, max_chunks, cancel):
    """Pay compilation and GeoCalib start-up before reporting ready, on a throwaway world."""
    if chunks == 0:
        return
    import numpy as np
    from PIL import Image
    from worldcrafter.calibration import calibrate_image
    from worldcrafter_camera import CameraPlanner
    started = time.monotonic()
    gradient = np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None].repeat(384, 0).repeat(3, 2)
    image = Image.fromarray(gradient)
    calibrate_image(image)
    engine.initialize_session(generation_image(native_image(image)),
                              WARMUP_PROMPT, 0, max_chunks, cancel)
    planner = CameraPlanner()
    for index in range(chunks):
        poses = planner.plan("static", 0.0) if index == 0 else planner.plan("forward", 1.0)
        engine.next_chunk(None, poses=poses)
    engine.close_session()
    print(f"WorldCrafter warmup complete: {chunks} chunks in {time.monotonic() - started:.1f} s", flush=True)


def native_image(image):
    """Resize an upright RGB image exactly as upstream ``inference.py`` does.

    Upstream I2V stretches the input to 640x384 with PIL's default filter,
    ``load_image(path).resize((640, 384))``, and calibrates gravity on that
    same resized image (``worldcrafter/cli.py`` ``prepare_camera``).
    """
    return image.resize((640, 384))


def generation_image(image):
    """Pass the 640x384 image losslessly to the path-based engine.

    The engine's preprocessing resizes to the same dimensions, an identity.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def path_poses(path, up, index):
    """Return cumulative (local, world) poses for an uploaded camera path prefix.

    Mirrors upstream ``inference.py --camera-path``: ``align_trajectory`` when
    gravity calibration succeeded (``cli.py`` ``prepare_camera``), the aligned
    global c2w as float32, and chunk-relative poses from ``relative_poses`` on
    each 33-frame window (``ucpe.bridge._relative_pose_chunk``). Alignment is
    per pose relative to pose zero, so a prefix aligns as the full path does.
    """
    import numpy as np
    from worldcrafter.camera import relative_poses
    from worldcrafter.gravity import align_trajectory
    if path.shape != (33 * (index + 1), 4, 4) or path.dtype != np.float32 or not np.isfinite(path).all():
        raise ValueError("Invalid cumulative camera path")
    world = align_trajectory(path, up) if up is not None else path
    world = np.ascontiguousarray(world, dtype=np.float32)
    local = np.concatenate([relative_poses(world[k * 33:(k + 1) * 33]) for k in range(index + 1)])
    return local, world


if __name__ == "__main__":
    main()
