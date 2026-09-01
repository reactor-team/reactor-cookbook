"""Thin stateful bridge to NVIDIA's released Lyra-2 GUI inference backend."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import numpy as np

# TE's cuDNN fused-attention backend rejects hosts that expose more than one
# CUDA runtime (this DLAMI has system CUDA 13 beside PyTorch's CUDA 12). Lyra's
# own DotProductAttention supports the equivalent FlashAttention path.
os.environ.setdefault("NVTE_FUSED_ATTN", "0")
os.environ.setdefault("NVTE_FLASH_ATTN", "1")


class Lyra2Backend:
    """Retain upstream streaming VAE caches, latent history, DA3, and Sparse3DCache."""

    def __init__(self, config: dict):
        self.config = config
        source = Path(config["source_path"]).resolve()
        weights = Path(os.environ.get("REACTOR_WEIGHTS_PATH", source.parent)).resolve()
        mounted_checkpoints = weights / "source/Lyra-2/checkpoints"
        image_checkpoints = source / "checkpoints"
        if not image_checkpoints.exists():
            image_checkpoints.symlink_to(mounted_checkpoints, target_is_directory=True)
        os.chdir(source)
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / "gui/api"))
        os.environ["LYRA_GUI_USE_DMD"] = "1"
        os.environ["LYRA_GUI_OFFLOAD"] = "0"
        os.environ["LYRA_GUI_PROMPT"] = config["default_prompt"]
        # A manual Reactor prompt is always supplied, so the GUI's optional Qwen
        # dynamic captioner must not reserve memory or download unrelated weights.
        import gui.api.lyra_persistent as persistent
        class _UnusedCaptioner:
            def __init__(self, *args, **kwargs): pass
        persistent.QwenCaptioner = _UnusedCaptioner
        self.model = persistent.Lyra2PersistentModel(
            checkpoint_path=str(weights / "source/Lyra-2/checkpoints/model"),
            output_root=str(weights / config["output_path"]),
        )
        # The released GUI wrapper is tuned for smaller cards and contains
        # additional hard-coded CPU moves beyond args.offload. This deployment
        # has a 183 GiB B200, so keep diffusion, CLIP, DA3 and UMT5 resident.
        self.model.args.offload = False
        self.model.args.offload_vae = False
        self.model.args.offload_da3_diffusion = False
        self.model.args.offload_da3_model = False
        self.model._video_net_on_cpu = lambda **_: nullcontext()

        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding

        def _embed_prompt_resident(model, caption: str):
            embedding = get_umt5_embedding(caption, device=str(model.device))
            return embedding.to(dtype=model.dtype).unsqueeze(0) if embedding.dim() == 2 else embedding.to(dtype=model.dtype)

        self.model._embed_prompt = MethodType(_embed_prompt_resident, self.model)
        self.model.da3_model.to(self.model.device)
        self.model.manual_t5 = self.model._embed_prompt(config["default_prompt"])
        self._intrinsics: np.ndarray | None = None

    def reset(self, image: np.ndarray, *, prompt: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        self.model.args.seed = seed
        self.model.manual_prompt = prompt
        self.model.manual_t5 = None
        identity = np.eye(4, dtype=np.float32)[None]
        result = self.model.seed_model_from_values(
            images_np=image[None], depths_np=None, masks_np=None,
            world_to_cameras_np=identity,
            focal_lengths_np=np.array([[1.0, 1.0]], dtype=np.float32),
            principal_point_rel_np=np.array([[0.5, 0.5]], dtype=np.float32),
            resolutions=np.array([[image.shape[1], image.shape[0]]], dtype=np.int32),
            request_id="reactor",
        )
        # seed_model_from_values() unconditionally parks DA3 on CPU for the GUI;
        # restore it immediately for subsequent spatial-memory updates.
        self.model.da3_model.to(self.model.device)
        fx, fy = result["focal_lengths"][0]
        cx, cy = result["principal_points"][0] * result["resolutions"][0]
        self._intrinsics = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3] = result["cameras_to_world"][0]
        return c2w, self._intrinsics.copy()

    def generate_chunk(self, w2c: np.ndarray, intrinsics: np.ndarray, *, prompt: str, chunk: int) -> tuple[np.ndarray, np.ndarray | None]:
        self.model.manual_prompt = prompt
        self.model.manual_t5 = None
        result = self.model.inference_on_cameras(
            w2c, intrinsics, fps=float(self.model.pipeline.fps.item()), save_buffer=False,
            request_id=f"reactor_{chunk:04d}",
        )
        # Upstream returns [B, T, C, H, W]; Reactor video tracks require
        # contiguous [T, H, W, C] RGB frames.
        video = result["video_no_overlap"][0].permute(0, 2, 3, 1).numpy()
        frames = np.clip((video + 1.0) * 127.5, 0, 255).astype(np.uint8)
        corrected = result.get("updated_last_camera_c2w")
        return np.ascontiguousarray(frames), corrected

    @property
    def intrinsics(self) -> np.ndarray:
        if self._intrinsics is None: raise RuntimeError("Lyra-2 is not seeded")
        return self._intrinsics

    def clear(self) -> None:
        self.model.clear_cache()
