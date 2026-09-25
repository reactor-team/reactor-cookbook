"""Load the native LingBot pipeline inside the Runner-owned worker process."""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WorkerSettings:
    """Plain paths and inference settings passed to the model worker."""

    source_path: Path
    checkpoint_dir: Path
    runtime_root: Path
    max_chunks: int
    context_latents: int
    max_area: int
    shift: float


class LingBotBackend:
    """Own one loaded Fast model and one optional causal rollout."""

    def __init__(
        self, settings: WorkerSettings, *, rank: int = 0, world_size: int = 1
    ) -> None:
        import sys

        sys.path.insert(0, str(settings.source_path))
        import torch
        import torch.distributed as dist
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.interactive_fast import InteractiveFastRollout

        self._torch = torch
        self._runtime_root = Path(settings.runtime_root).resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        config = WAN_CONFIGS["i2v-A14B"]
        context_latents = int(settings.context_latents)
        checkpoint_dir = Path(settings.checkpoint_dir).resolve()
        if "cam" not in checkpoint_dir.name:
            raise ValueError(
                "LingBot Fast camera checkpoints must use a path containing 'cam'"
            )
        if world_size not in (1, 2, 4) or not 0 <= rank < world_size:
            raise ValueError("LingBot supports one, two, or four ranks")
        if world_size > 1 and (
            not dist.is_initialized()
            or dist.get_world_size() != world_size
            or dist.get_rank() != rank
        ):
            raise RuntimeError(
                "DistributedRunner must initialize the matching process group"
            )
        if config.num_heads % world_size:
            raise ValueError("LingBot attention heads must divide evenly across ranks")
        torch.cuda.set_device(rank)
        self._pipe = wan.WanI2VFast(
            config=config,
            checkpoint_dir=str(checkpoint_dir),
            device_id=rank,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=world_size > 1,
            t5_cpu=False,
            init_on_cpu=False,
            convert_model_dtype=False,
            pipe_dtype=torch.bfloat16,
            local_attn_size=context_latents,
            sink_size=0,
        )
        self._rollout = InteractiveFastRollout(
            self._pipe,
            max_chunks=int(settings.max_chunks),
            context_latents=context_latents,
            chunk_size=3,
            max_area=int(settings.max_area),
            shift=float(settings.shift),
        )
        self._active = False

    def reset(
        self,
        *,
        seed: int,
        anchor_image: Path | bytes,
        suffix: str,
        intrinsics: Path,
        prompt: str,
    ) -> None:
        """Start a fresh image-conditioned causal rollout."""
        if not intrinsics.is_file():
            raise FileNotFoundError(f"LingBot intrinsics do not exist: {intrinsics}")
        with ExitStack() as stack:
            if isinstance(anchor_image, bytes):
                temporary = stack.enter_context(
                    tempfile.NamedTemporaryFile(suffix=suffix, dir=self._runtime_root)
                )
                temporary.write(anchor_image)
                temporary.flush()
                image_path = Path(temporary.name)
            else:
                image_path = anchor_image
            with Image.open(image_path) as image:
                self._rollout.reset(
                    prompt,
                    image.convert("RGB"),
                    np.load(intrinsics, allow_pickle=False),
                    seed,
                )
        self._active = True

    def generate_chunk(self, relative_c2ws: np.ndarray, prompt: str) -> np.ndarray:
        """Return one native chunk as contiguous CPU RGB frames."""
        if not self._active:
            raise RuntimeError("reset LingBot before generating a chunk")
        poses = np.asarray(relative_c2ws, dtype=np.float32)
        if poses.shape != (3, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(
                "LingBot relative camera poses must be finite with shape (3, 4, 4)"
            )
        video = self._rollout.generate_chunk(poses, prompt)
        frames = (
            video.permute(1, 2, 3, 0)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .to(self._torch.uint8)
            .cpu()
            .numpy()
        )
        frames = np.ascontiguousarray(frames)
        expected = 9 if self._rollout.chunk_index == 1 else 12
        if frames.shape[0] != expected:
            raise RuntimeError(
                f"LingBot causal VAE decoded {frames.shape[0]} frames; expected {expected}"
            )
        return frames

    def end_session(self) -> None:
        """Release causal caches while preserving model weights."""
        self._rollout.end()
        self._active = False
