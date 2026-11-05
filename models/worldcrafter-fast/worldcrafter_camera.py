"""CPU camera planning in WorldCrafter's right/down/forward c2w convention."""
from dataclasses import dataclass
import json
import math

import numpy as np

FRAMES = 33
TRANSLATIONS = ("forward", "backward", "left", "right", "up", "down")
ROTATIONS = ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "orbit_left", "orbit_right",
             "orbit_up", "orbit_down")
ACTIONS = TRANSLATIONS + ROTATIONS
MAX_TRANSLATION = 5.0
MAX_ROTATION = 45.0
MAX_CHUNKS_PER_MOVE = 8
MAX_CAMERA_PATH_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CameraPath:
    """Finite global c2w path, float32 ``(33 * chunks, 4, 4)``; pose zero is the input image.

    These are the poses upstream ``inference.py --camera-path`` reads, unchanged:
    one per output frame, unaligned to gravity. The worker applies upstream's
    gravity alignment and chunk-relative pose derivation.
    """
    poses: np.ndarray

    @property
    def chunks(self) -> int:
        return len(self.poses) // FRAMES


def load_camera_path(data: bytes, max_chunks: int) -> CameraPath:
    """Parse ``{"poses": [33 * chunks, 4, 4]}``; raise ValueError for an unusable path.

    Rotation tolerances follow upstream ``load_camera``. Poses are kept as
    uploaded, as upstream keeps a ``camera.npy``: the model conditions on
    chunk-relative poses, and gravity alignment is about pose zero.
    """
    if len(data) > MAX_CAMERA_PATH_BYTES:
        raise ValueError("Camera path exceeds 16 MiB")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Camera path must be JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"poses"}:
        raise ValueError("Camera path must contain only poses")
    try:
        poses = np.asarray(payload["poses"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("poses must be numeric") from exc
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N, 4, 4]")
    if not FRAMES <= len(poses) <= max_chunks * FRAMES or len(poses) % FRAMES:
        raise ValueError(f"A path needs {FRAMES} * chunks poses, at most {max_chunks} chunks")
    if not np.isfinite(poses).all() or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("Camera poses must be finite homogeneous transforms")
    rotation = poses[:, :3, :3]
    if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=5e-3, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=5e-3, rtol=0)):
        raise ValueError("Camera poses require proper rotations")
    poses = np.ascontiguousarray(poses.astype(np.float32))
    poses.setflags(write=False)
    return CameraPath(poses)


def validate_motion(action: str, amount: float, chunks: int):
    """Raise ValueError, with a reason for the client, unless the motion is plannable."""
    if action not in ACTIONS:
        raise ValueError("Use a supported camera direction.")
    translation = action in TRANSLATIONS
    limit = MAX_TRANSLATION if translation else MAX_ROTATION
    if not math.isfinite(amount) or not 0.1 <= amount <= limit:
        unit = "units" if translation else "degrees"
        raise ValueError(f"{'Translation' if translation else 'Rotation'} must be 0.1 to {limit:g} {unit} per chunk.")
    if not 1 <= chunks <= MAX_CHUNKS_PER_MOVE:
        raise ValueError(f"Queue 1 to {MAX_CHUNKS_PER_MOVE} chunks per move.")


def gravity_basis(up):
    """Map the initial camera into a level world; UP is in camera coordinates."""
    up = np.asarray(up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("Gravity must contain three finite values")
    norm = np.linalg.norm(up)
    if abs(norm - 1) > .01 or abs(up[2] / norm) > math.sin(math.radians(75)):
        raise ValueError("Invalid calibrated gravity")
    down = -up / norm
    forward = np.array([0., 0., 1.]) - down[2] * down
    forward /= np.linalg.norm(forward)
    return np.stack([np.cross(down, forward), down, forward])


class CameraPlanner:
    """Own cumulative poses and logical endpoints, independent of model inference.

    A chunk samples fractions 0/33 through 32/33. The next chunk starts at the
    logical endpoint, not at the last sampled frame. Local poses restart at
    each chunk; retrieval poses remain relative to the initial camera.
    """

    def __init__(self):
        self.world = np.eye(4, dtype=np.float64)
        self.initial_inverse = np.eye(4, dtype=np.float64)
        self.local_chunks = []
        self.global_chunks = []

    def align_after_anchor(self, world_up):
        """Apply calibration after the identity anchor chunk, before any motion."""
        if len(self.global_chunks) != 1:
            raise ValueError("Gravity must be applied immediately after the anchor")
        if world_up is not None:
            self.world[:3, :3] = gravity_basis(world_up)
            self.initial_inverse = np.linalg.inv(self.world)

    def plan(self, action, amount):
        rotation = self.world[:3, :3]
        poses = np.repeat(self.world[None], FRAMES, axis=0)
        end = self.world.copy()
        alpha = np.arange(FRAMES, dtype=np.float64) / FRAMES
        turns = {
            "yaw_left": ("yaw", -1), "yaw_right": ("yaw", 1),
            "pitch_up": ("pitch", 1), "pitch_down": ("pitch", -1),
            "orbit_left": ("yaw", 1), "orbit_right": ("yaw", -1),
            "orbit_up": ("pitch", -1), "orbit_down": ("pitch", 1),
        }
        if action in turns:
            axis, sign = turns[action]

            def rotated(fraction):
                angle = math.radians(sign * amount * fraction)
                c, s = math.cos(angle), math.sin(angle)
                if axis == "yaw":
                    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rotation
                return rotation @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

            for i, fraction in enumerate(alpha):
                poses[i, :3, :3] = rotated(fraction)
            end[:3, :3] = rotated(1.)
            if action.startswith("orbit_"):
                center = self.world[:3, 3] + 2. * rotation[:, 2]
                poses[:, :3, 3] = center - 2. * poses[:, :3, 2]
                poses[0, :3, 3] = self.world[:3, 3]
                end[:3, 3] = center - 2. * end[:3, 2]
        elif action != "static":
            if action in ("up", "down"):
                direction = np.array([0., -1. if action == "up" else 1., 0.])
            elif action in ("forward", "backward", "left", "right"):
                forward = action in ("forward", "backward")
                direction = rotation[:, 2 if forward else 0].copy()
                direction[1] = 0
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    other = rotation[:, 0 if forward else 2]
                    direction = np.array([-other[2], 0., other[0]])
                    if not forward:
                        direction = -direction
                    norm = np.linalg.norm(direction)
                direction /= norm
                if action in ("backward", "left"):
                    direction = -direction
            else:
                raise ValueError("Unsupported camera action")
            delta = amount * direction
            poses[:, :3, 3] += alpha[:, None] * delta
            end[:3, 3] += delta
        global_pose = (self.initial_inverse @ poses)[:, :3].astype(np.float32)
        homogeneous = np.zeros((FRAMES, 4, 4), dtype=np.float64)
        homogeneous[:, :3, :4] = global_pose
        homogeneous[:, 3, 3] = 1
        local = (np.linalg.inv(homogeneous[0])[None] @ homogeneous)[:, :3].astype(np.float32)
        self.world = end
        self.local_chunks.append(local)
        self.global_chunks.append(global_pose)
        local, global_pose = np.concatenate(self.local_chunks), np.concatenate(self.global_chunks)
        local.setflags(write=False)
        global_pose.setflags(write=False)
        return local, global_pose
