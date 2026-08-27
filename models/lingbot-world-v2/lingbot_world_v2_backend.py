"""Run the public LingBot causal-fast model one native chunk at a time."""

from __future__ import annotations

import hashlib
import importlib
import io
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from reactor_runtime import UploadedFile
from reactor_runtime.log import get_logger

from lingbot_world_v2_assets import LingBotConfig

logger = get_logger(__name__)

RGB_FPS = 16
TEMPORAL_STRIDE = 4
FIRST_CHUNK_FRAMES = 13
STEADY_CHUNK_FRAMES = 16


class LingBotBackend:
    """Keep public model weights and native causal state resident on one GPU.

    The public ``generate`` method owns the complete rollout loop. This backend
    lifts that loop into ``reset`` and ``generate_chunk`` without altering the
    upstream checkout. Self-attention KV, cross-attention KV, scheduler RNG,
    image-conditioning VAE state, and decoder state survive between calls.
    """

    def __init__(self, config: LingBotConfig) -> None:
        self._config = config
        modules = _load_upstream(config.source_path)
        self._torch = modules["torch"]
        self._tf = modules["torchvision_functional"]
        self._rearrange = modules["rearrange"]
        self._get_ks_transformed = modules["get_ks_transformed"]
        self._get_plucker_embeddings = modules["get_plucker_embeddings"]

        torch = self._torch
        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-World-V2 requires a CUDA GPU")
        torch.cuda.set_device(0)
        self._device = torch.device("cuda:0")
        upstream_config = modules["wan_configs"]["i2v-A14B"]
        self._pipe = modules["pipeline_type"](
            config=upstream_config,
            checkpoint_dir=str(config.checkpoint_path),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=False,
            convert_model_dtype=False,
            pipe_dtype=torch.bfloat16,
            local_attn_size=config.local_attention_frames,
            sink_size=config.attention_sink_frames,
            infer_mode="causal_fast",
        )
        self._pipe.model.to(self._device)
        self._session_ready = False
        self._chunk_index = 0
        self._prompt = ""
        self._context: list[Any] | None = None
        self._self_kv: list[dict[str, Any]] | None = None
        self._cross_kv: list[dict[str, Any]] | None = None
        self._base_noise: Any | None = None
        self._seed_generator: Any | None = None
        self._anchor: Any | None = None
        self._intrinsics: np.ndarray | None = None
        self._lat_h = 0
        self._lat_w = 0
        self._height = 0
        self._width = 0
        self._frame_seqlen = 0
        self._kv_size = 0
        self._timesteps: Any | None = None
        logger.info(
            "LingBot-World-V2 weights loaded",
            source_revision=config.source_revision,
            checkpoint_revision=config.checkpoint_revision,
            local_attention_frames=config.local_attention_frames,
            attention_sink_frames=config.attention_sink_frames,
        )

    def reset(
        self,
        *,
        image: Path | UploadedFile,
        prompt: str,
        seed: int,
        intrinsics: np.ndarray,
    ) -> None:
        """Start a fresh native causal rollout from an image and prompt.

        Args:
            image: Public example path or validated Reactor upload.
            prompt: Non-empty text condition for the first chunk.
            seed: Deterministic scheduler seed for the complete rollout.
            intrinsics: Packed ``[fx, fy, cx, cy]`` camera calibration.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("LingBot prompt must not be empty")
        self.end_session()
        torch = self._torch
        image_tensor = self._tf.to_tensor(_open_image(image)).sub_(0.5).div_(0.5)
        image_height, image_width = image_tensor.shape[1:]
        aspect_ratio = image_height / image_width
        self._lat_h = round(
            math.sqrt(self._config.max_area * aspect_ratio)
            // self._pipe.vae_stride[1]
            // self._pipe.patch_size[1]
            * self._pipe.patch_size[1]
        )
        self._lat_w = round(
            math.sqrt(self._config.max_area / aspect_ratio)
            // self._pipe.vae_stride[2]
            // self._pipe.patch_size[2]
            * self._pipe.patch_size[2]
        )
        if self._lat_h <= 0 or self._lat_w <= 0:
            raise ValueError(
                "anchor image aspect ratio produces an empty LingBot latent"
            )
        self._height = self._lat_h * self._pipe.vae_stride[1]
        self._width = self._lat_w * self._pipe.vae_stride[2]
        self._frame_seqlen = (
            self._lat_h
            * self._lat_w
            // (self._pipe.patch_size[1] * self._pipe.patch_size[2])
        )
        self._kv_size = self._frame_seqlen * self._config.local_attention_frames
        self._intrinsics = _normalize_intrinsics(intrinsics)
        self._anchor = (
            torch.nn.functional.interpolate(
                image_tensor[None],
                size=(self._height, self._width),
                mode="bicubic",
            )
            .transpose(0, 1)
            .to(self._device)
        )

        self._pipe.scheduler.set_timesteps(
            self._pipe.num_train_timesteps,
            shift=self._config.shift,
        )
        self._timesteps = self._pipe.scheduler.timesteps[list(self._config.timesteps)]
        self._seed_generator = torch.Generator(device=self._device)
        self._seed_generator.manual_seed(seed)
        total_latents = self._config.max_chunks * self._config.chunk_latents
        self._base_noise = torch.randn(
            16,
            total_latents,
            self._lat_h,
            self._lat_w,
            dtype=torch.float32,
            generator=self._seed_generator,
            device=self._device,
        )

        model_args = self._pipe.model.config
        head_dim = model_args.dim // model_args.num_heads
        self._self_kv = self._pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, self._kv_size, model_args.num_heads, head_dim],
            dtype=self._pipe.pipe_dtype,
            device=self._device,
        )
        self._set_prompt(prompt)
        self._pipe.vae.model.clear_cache()
        self._chunk_index = 0
        self._session_ready = True

    def generate_chunk(self, *, prompt: str, relative_poses: np.ndarray) -> np.ndarray:
        """Generate and decode exactly one native four-latent causal chunk.

        Args:
            prompt: Text condition sampled at this chunk boundary.
            relative_poses: Four framewise OpenCV transforms from the camera planner.

        Returns:
            RGB uint8 frames in ``THWC`` layout: 13 for chunk one, then 16.
        """
        if not self._session_ready:
            raise RuntimeError("LingBot rollout must be reset before generation")
        if self._chunk_index >= self._config.max_chunks:
            raise RuntimeError("LingBot rollout reached its native positional limit")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("LingBot prompt must not be empty")
        if prompt != self._prompt:
            self._set_prompt(prompt)

        torch = self._torch
        start = self._chunk_index * self._config.chunk_latents
        current_latent = self._base_noise[
            :, start : start + self._config.chunk_latents
        ].clone()
        condition = self._next_condition()
        camera = self._camera_embedding(relative_poses)
        kwargs = {
            "context": [self._context[0]],
            "seq_len": self._config.chunk_latents * self._frame_seqlen,
            "y": [condition],
            "dit_cond_dict": {"c2ws_plucker_emb": camera.chunk(1, dim=0)},
            "kv_cache": self._self_kv,
            "crossattn_cache": self._cross_kv,
            "current_start": start * self._frame_seqlen,
            "max_attention_size": self._kv_size,
            "frame_seqlen": self._frame_seqlen,
        }

        no_sync = getattr(self._pipe.model, "no_sync", nullcontext)
        with (
            torch.amp.autocast("cuda", dtype=self._pipe.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            for timestep_index, current_timestep in enumerate(self._timesteps):
                timestep = current_timestep.reshape(1).to(self._device)
                noise_pred = self._pipe.model(
                    x=[current_latent],
                    t=timestep,
                    cross_attn_first_call=not self._pipe._cross_attn_initialized,
                    **kwargs,
                )[0]
                self._pipe._cross_attn_initialized = True
                x0 = self._pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=current_latent,
                    timestep=current_timestep,
                    scheduler=self._pipe.scheduler,
                )
                if timestep_index < len(self._timesteps) - 1:
                    next_timestep = self._timesteps[timestep_index + 1]
                    noise = torch.randn(
                        x0.shape,
                        generator=self._seed_generator,
                        device=x0.device,
                        dtype=x0.dtype,
                    )
                    current_latent = self._pipe.scheduler.add_noise(
                        x0, noise, next_timestep
                    )

            zero_timestep = self._timesteps[-1].new_zeros(1).to(self._device)
            self._pipe.model(
                x=[x0],
                t=zero_timestep,
                cross_attn_first_call=False,
                **kwargs,
            )

        frames = self._decode_chunk(x0)
        expected = FIRST_CHUNK_FRAMES if self._chunk_index == 0 else STEADY_CHUNK_FRAMES
        if frames.shape[0] != expected:
            raise RuntimeError(
                f"LingBot decoded {frames.shape[0]} frames for chunk {self._chunk_index + 1}; "
                f"expected {expected}"
            )
        self._chunk_index += 1
        return frames

    def end_session(self) -> None:
        """Release rollout caches while keeping public model weights resident."""
        self._session_ready = False
        self._chunk_index = 0
        self._prompt = ""
        self._context = None
        self._self_kv = None
        self._cross_kv = None
        self._base_noise = None
        self._seed_generator = None
        self._anchor = None
        self._intrinsics = None
        self._pipe._cross_attn_initialized = False
        self._pipe.vae.model.clear_cache()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _set_prompt(self, prompt: str) -> None:
        """Encode a prompt and reset only the cross-attention cache."""
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if cache_key in self._pipe._t5_cache:
            context = self._pipe._t5_cache[cache_key]
        else:
            self._pipe.text_encoder.model.to(self._device)
            context = self._pipe.text_encoder([prompt], self._device)
            self._pipe._t5_cache[cache_key] = context
        model_args = self._pipe.model.config
        head_dim = model_args.dim // model_args.num_heads
        self._cross_kv = self._pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[1, 512, model_args.num_heads, head_dim],
            dtype=self._pipe.pipe_dtype,
            device=self._device,
        )
        self._pipe._cross_attn_initialized = False
        self._prompt = prompt
        self._context = context
        logger.info("LingBot prompt ready", prompt_sha256=cache_key)

    def _next_condition(self) -> Any:
        """Encode the anchor or causal zero continuation into four VAE latents."""
        torch = self._torch
        outputs = []
        first_chunk = self._chunk_index == 0
        if first_chunk:
            outputs.append(self._encode_condition_block(self._anchor))
            zero_blocks = self._config.chunk_latents - 1
        else:
            zero_blocks = self._config.chunk_latents
        for _ in range(zero_blocks):
            zeros = torch.zeros(
                3,
                TEMPORAL_STRIDE,
                self._height,
                self._width,
                device=self._device,
            )
            outputs.append(self._encode_condition_block(zeros))
        latents = torch.cat(outputs, dim=1)
        mask = torch.zeros(
            4,
            self._config.chunk_latents,
            self._lat_h,
            self._lat_w,
            device=self._device,
        )
        if first_chunk:
            mask[:, 0] = 1
        return torch.cat([mask, latents])

    def _encode_condition_block(self, frames: Any) -> Any:
        """Advance the upstream causal VAE encoder without clearing its cache."""
        model = self._pipe.vae.model
        model._enc_conv_idx = [0]
        with self._vae_autocast():
            encoded = model.encoder(
                frames.unsqueeze(0),
                feat_cache=model._enc_feat_map,
                feat_idx=model._enc_conv_idx,
            )
            mean, _ = model.conv1(encoded).chunk(2, dim=1)
            scale = self._pipe.vae.scale
            mean = (mean - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(
                1, model.z_dim, 1, 1, 1
            )
        return mean.float().squeeze(0)

    def _decode_chunk(self, latents: Any) -> np.ndarray:
        """Advance the upstream causal VAE decoder and return only new RGB frames."""
        torch = self._torch
        model = self._pipe.vae.model
        scale = self._pipe.vae.scale
        with self._vae_autocast():
            value = latents.unsqueeze(0)
            value = value / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(
                1, model.z_dim, 1, 1, 1
            )
            value = model.conv2(value)
            decoded = []
            for index in range(value.shape[2]):
                model._conv_idx = [0]
                decoded.append(
                    model.decoder(
                        value[:, :, index : index + 1],
                        feat_cache=model._feat_map,
                        feat_idx=model._conv_idx,
                    )
                )
            video = torch.cat(decoded, dim=2).float().clamp_(-1, 1).squeeze(0)
        return (
            ((video + 1.0) * 127.5)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

    def _camera_embedding(self, relative_poses: np.ndarray) -> Any:
        """Convert four framewise poses into upstream Pluecker conditioning."""
        torch = self._torch
        poses = np.asarray(relative_poses, dtype=np.float32)
        expected = (self._config.chunk_latents, 4, 4)
        if poses.shape != expected or not np.isfinite(poses).all():
            raise ValueError(f"relative camera poses must have shape {expected}")
        intrinsics = torch.from_numpy(self._intrinsics[None]).float()
        intrinsics = self._get_ks_transformed(
            intrinsics,
            height_org=480,
            width_org=832,
            height_resize=self._height,
            width_resize=self._width,
            height_final=self._height,
            width_final=self._width,
        )[0]
        intrinsics = intrinsics.repeat(self._config.chunk_latents, 1).to(self._device)
        pose_tensor = torch.from_numpy(poses).to(self._device)
        plucker = self._get_plucker_embeddings(
            pose_tensor,
            intrinsics,
            self._height,
            self._width,
        )
        plucker = self._rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=self._height // self._lat_h,
            c2=self._width // self._lat_w,
        )[None]
        return self._rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=self._config.chunk_latents,
            h=self._lat_h,
            w=self._lat_w,
        ).to(self._pipe.param_dtype)

    def _vae_autocast(self) -> Any:
        """Return the dtype context used by the upstream VAE wrapper."""
        dtype = self._pipe.vae.dtype
        if dtype not in (self._torch.float16, self._torch.bfloat16):
            return nullcontext()
        return self._torch.amp.autocast("cuda", dtype=dtype)


def _load_upstream(source_path: Path) -> dict[str, Any]:
    """Import the pinned upstream modules without modifying their checkout."""
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    wan = importlib.import_module("wan")
    origin = Path(wan.__file__ or "").resolve()
    if source_path.resolve() not in origin.parents:
        raise RuntimeError(f"imported wan from {origin}, expected {source_path}")
    torch = importlib.import_module("torch")
    attention_module = importlib.import_module("wan.modules.attention")
    if not (
        attention_module.FLASH_ATTN_2_AVAILABLE
        or attention_module.FLASH_ATTN_3_AVAILABLE
    ):
        model_module = importlib.import_module("wan.modules.model_fast")
        model_module.flash_attention = attention_module.attention
        logger.info("LingBot cross-attention uses the upstream PyTorch SDPA fallback")
    return {
        "torch": torch,
        "torchvision_functional": importlib.import_module(
            "torchvision.transforms.functional"
        ),
        "rearrange": importlib.import_module("einops").rearrange,
        "pipeline_type": wan.WanI2VCausal,
        "wan_configs": importlib.import_module("wan.configs").WAN_CONFIGS,
        "get_ks_transformed": importlib.import_module(
            "wan.utils.cam_utils"
        ).get_Ks_transformed,
        "get_plucker_embeddings": importlib.import_module(
            "wan.utils.cam_utils"
        ).get_plucker_embeddings,
    }


def _open_image(value: Path | UploadedFile) -> Image.Image:
    """Decode a selected image and return an EXIF-oriented RGB copy."""
    source: Path | io.BytesIO = (
        value if isinstance(value, Path) else io.BytesIO(value.data)
    )
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def _normalize_intrinsics(value: np.ndarray) -> np.ndarray:
    """Return a finite packed camera calibration vector."""
    intrinsics = np.asarray(value, dtype=np.float32)
    if intrinsics.ndim == 2 and intrinsics.shape[1:] == (4,):
        intrinsics = intrinsics[0]
    if intrinsics.shape != (4,):
        raise ValueError(
            f"intrinsics must have shape (4,) or (N, 4), got {intrinsics.shape}"
        )
    if not np.isfinite(intrinsics).all() or bool(np.any(intrinsics[:2] <= 0)):
        raise ValueError("intrinsics must be finite with positive focal lengths")
    return np.ascontiguousarray(intrinsics)
