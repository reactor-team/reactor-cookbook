"""Thin stateful bridge to NVIDIA's released Lyra-2 GUI inference backend."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import numpy as np

# lyra2_custom_traj_inference.py argparse defaults (nv-tlabs/lyra@9fffc9a,
# :138-157). The Reactor recipe serves the official script's settings rather
# than the GUI's (gui/api/lyra_persistent.py:77-78,114-117).
HEIGHT, WIDTH = 480, 832
DA3_FRAME_INTERVAL = 8
DA3_MAX_HISTORY_FRAMES = 10


class Lyra2Backend:
    """Retain upstream streaming VAE caches, latent history, DA3, and Sparse3DCache."""

    def __init__(self, config: dict):
        self.config = config
        source = Path(config["source_path"]).resolve()
        os.chdir(source)
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / "gui/api"))
        os.environ["LYRA_GUI_PROMPT"] = config["default_prompt"]
        # A manual Reactor prompt is always supplied, so the GUI's optional Qwen
        # dynamic captioner must not reserve memory or download unrelated weights.
        import gui.api.lyra_persistent as persistent

        class _UnusedCaptioner:
            def __init__(self, *args, **kwargs):
                pass

        persistent.QwenCaptioner = _UnusedCaptioner
        self.model = persistent.Lyra2PersistentModel(
            checkpoint_path=str(Path(config["checkpoints_path"]) / "model"),
            output_root=config["output_path"],
        )
        # The released GUI wrapper is tuned for smaller cards and contains
        # additional hard-coded CPU moves beyond args.offload. This B200 recipe
        # keeps diffusion, CLIP, DA3 and UMT5 resident.
        self.model.args.offload = False
        self.model.args.offload_vae = False
        self.model.args.offload_da3_diffusion = False
        self.model.args.offload_da3_model = False
        self.model.target_h, self.model.target_w = HEIGHT, WIDTH
        self.model.args.da3_frame_interval = DA3_FRAME_INTERVAL
        self.model.args.da3_max_history_frames = DA3_MAX_HISTORY_FRAMES
        self.model.args.da3_include_ar_chunk_last_frames = False
        self.model.args.da3_use_predicted_pose = False
        self.model.args.da3_predicted_pose_continuation = False
        self.model._video_net_on_cpu = lambda **_: nullcontext()

        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding

        def _embed_prompt_resident(model, caption: str):
            embedding = get_umt5_embedding(caption, device=str(model.device))
            return (
                embedding.to(dtype=model.dtype).unsqueeze(0)
                if embedding.dim() == 2
                else embedding.to(dtype=model.dtype)
            )

        self.model._embed_prompt = MethodType(_embed_prompt_resident, self.model)
        # The GUI wrapper re-enables cuDNN; the released inference scripts run
        # every convolution outside VAE decode on PyTorch's native path.
        import torch

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        self.model.da3_model.to(self.model.device)
        # --use_moge_scale (default on): MoGe loads after DA3, as in the script.
        from lyra_2._src.inference.depth_utils import load_moge_model

        self.moge_model = load_moge_model(self.model.device)
        self.moge_model.eval()
        self._seed_intrinsics: np.ndarray | None = None
        self._da3_single = persistent._da3_infer_depth_intrinsics_single
        persistent._da3_infer_depth_intrinsics_single = self._seed_depth
        self.model.manual_t5 = self.model._embed_prompt(config["default_prompt"])
        self._prompt = config["default_prompt"]

    def _seed_depth(self, da3_model, img_rgb_uint8, target_hw):
        """DA3 seed depth, rescaled to MoGe, with the trajectory's first K.

        Mirrors lyra2_custom_traj_inference.py:343-395. The seed image is
        already at the target size, so both helpers' INTER_LINEAR resizes are
        identity copies of the script's resize of the original image.
        """
        import gc

        import torch
        from lyra_2._src.inference.depth_utils import moge_infer_depth_intrinsics

        image_chw01, depth_hw, intrinsics, mask_hw = self._da3_single(
            da3_model, img_rgb_uint8, target_hw
        )
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                self.moge_model,
                img_rgb_uint8,
                depth_pred_hw=tuple(target_hw),
                target_hw=tuple(target_hw),
            )
        da3_d = depth_hw.to(moge_depth_hw.device)
        da3_m = mask_hw.to(moge_mask_hw.device)
        valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)
        if valid_mask.sum() > 10:
            inv_da3 = 1.0 / (da3_d[valid_mask] + 1e-6)
            inv_moge = 1.0 / (moge_depth_hw[valid_mask] + 1e-6)
            numerator = (inv_da3 * inv_moge).sum()
            denominator = (inv_da3 * inv_da3).sum()
            if denominator > 1e-8:
                scale = numerator / denominator
                if scale > 1e-6:
                    depth_hw = depth_hw / scale.to(depth_hw.device)
        del moge_depth_hw, moge_mask_hw, da3_d, da3_m
        torch.cuda.empty_cache()
        gc.collect()
        if self._seed_intrinsics is not None:
            # The script seeds its cache with the trajectory's K, not DA3's.
            intrinsics = torch.from_numpy(
                np.asarray(self._seed_intrinsics, dtype=np.float32).copy()
            )
        return image_chw01, depth_hw, intrinsics, mask_hw

    def reset(
        self,
        image: np.ndarray,
        *,
        prompt: str,
        seed: int,
        intrinsics: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from lyra_2._ext.imaginaire.utils import misc

        self.model.args.seed = seed
        # DMD re-noises steps 2-4 from the global RNG; seed it per world as
        # lyra2_custom_traj_inference does before each sample.
        misc.set_random_seed(seed=seed, by_rank=True)
        self.model.manual_prompt = prompt
        self._set_prompt(prompt)
        self._seed_intrinsics = intrinsics
        identity = np.eye(4, dtype=np.float32)[None]
        result = self.model.seed_model_from_values(
            images_np=image[None],
            depths_np=None,
            masks_np=None,
            world_to_cameras_np=identity,
            focal_lengths_np=np.array([[1.0, 1.0]], dtype=np.float32),
            principal_point_rel_np=np.array([[0.5, 0.5]], dtype=np.float32),
            resolutions=np.array([[image.shape[1], image.shape[0]]], dtype=np.int32),
            request_id="reactor",
        )
        # seed_model_from_values() unconditionally parks DA3 on CPU for the GUI;
        # restore it immediately for subsequent spatial-memory updates.
        self.model.da3_model.to(self.model.device)
        # Reuse the exact seed matrix (the trajectory's first K when supplied,
        # otherwise DA3's); the GUI's relative principal point does not
        # round-trip through float32.
        intrinsics = self.model.seed_intrinsics[0].float().cpu().numpy()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3] = result["cameras_to_world"][0]
        return c2w, intrinsics

    def generate_chunk(
        self, w2c: np.ndarray, intrinsics: np.ndarray, *, prompt: str, chunk: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.model.manual_prompt = prompt
        self._set_prompt(prompt)
        result = self.model.inference_on_cameras(
            w2c,
            intrinsics,
            fps=float(self.model.pipeline.fps.item()),
            save_buffer=False,
            request_id=f"reactor_{chunk:04d}",
        )
        # Upstream returns [B, T, C, H, W]; Reactor video tracks require
        # contiguous [T, H, W, C] RGB frames.
        video = (
            result["video_no_overlap"][0]
            .permute(0, 2, 3, 1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        frames = np.clip((video + 1.0) * 127.5, 0, 255).astype(np.uint8)
        corrected = result.get("updated_last_camera_c2w")
        if corrected is not None:
            if not isinstance(corrected, np.ndarray):
                corrected = corrected.detach().float().cpu().numpy()
            corrected = np.asarray(corrected, dtype=np.float32)
            if corrected.shape == (3, 4):
                corrected = np.vstack((corrected, np.array([0, 0, 0, 1], np.float32)))
            if corrected.shape != (4, 4) or not np.isfinite(corrected).all():
                raise ValueError("Lyra returned an invalid corrected camera pose")
        return np.ascontiguousarray(frames), corrected

    def _set_prompt(self, prompt: str) -> None:
        if prompt != self._prompt or self.model.manual_t5 is None:
            self.model.manual_t5 = self.model._embed_prompt(prompt)
            self._prompt = prompt

    def clear(self) -> None:
        self.model.clear_cache()
