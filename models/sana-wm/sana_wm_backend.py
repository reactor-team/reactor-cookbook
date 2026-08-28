"""Drive SANA-WM's native streaming stages one autoregressive chunk at a time."""

from __future__ import annotations

import gc
import importlib
import io
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from reactor_runtime import UploadedFile

from sana_wm_assets import resolve_model_assets, resolve_pi3x_assets
from sana_wm_types import Control, SanaWMConfig

FPS = 16
PIXEL_FRAMES_PER_CHUNK = 24
LATENT_FRAMES_PER_CHUNK = 3
_SINK_FRAMES = 1
_VAE_TIME_STRIDE = 8
_DENOISING_STEPS = [1000, 960, 889, 727, 0]


class TrajectoryCompleteError(RuntimeError):
    """Signal that a finite uploaded trajectory has no complete chunk left."""


def _prepend_import_path(path: Path) -> None:
    """Put one public source checkout first on the Python import path."""
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _open_image(source: Path | UploadedFile) -> Image.Image:
    """Decode a path or Reactor upload as an oriented RGB image."""
    from PIL import ImageOps

    if isinstance(source, UploadedFile):
        stream: str | io.BytesIO = io.BytesIO(source.data)
    else:
        stream = str(source)
    with Image.open(stream) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_numpy_upload(file: UploadedFile) -> np.ndarray:
    """Decode one upload as a non-pickled NumPy array."""
    try:
        return np.load(io.BytesIO(file.data), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{file.name} is not a valid NumPy .npy file") from exc


class SanaStreamingBackend:
    """Own model weights and the three persistent upstream streaming caches."""

    def __init__(self, config: SanaWMConfig) -> None:
        os.environ.setdefault("DISABLE_XFORMERS", "1")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("DPM_TQDM", "True")
        _prepend_import_path(config.source_path)

        self._config = config
        self._torch = importlib.import_module("torch")
        if not self._torch.cuda.is_available():
            raise RuntimeError("SANA-WM requires a CUDA accelerator")
        self._device = self._torch.device("cuda")
        self._wm = importlib.import_module(
            "inference_video_scripts.wm.inference_sana_wm"
        )
        self._camera_control = importlib.import_module(
            "inference_video_scripts.wm.camera_control"
        )
        self._sampler_module = importlib.import_module(
            "diffusion.scheduler.self_forcing_flow_euler_sampler"
        )
        self._refiner_module = importlib.import_module(
            "diffusion.refiner.diffusers_ltx2_refiner"
        )
        self._vae_streaming_module = importlib.import_module("diffusion.model.ltx2")
        self._chunk_module = importlib.import_module("diffusion.utils.chunk_utils")
        self._pyrallis = importlib.import_module("pyrallis")
        self._transforms = importlib.import_module("torchvision.transforms")

        streaming_root, text_encoder_root = resolve_model_assets(config)
        self._install_pinned_text_encoder(text_encoder_root)
        upstream_config = self._pyrallis.parse(
            config_class=self._wm.InferenceConfig,
            config_path=config.upstream_config,
            args=[],
        )
        upstream_config.vae.vae_pretrained = str(streaming_root / "ltx2_causal_vae")
        refiner = self._wm.RefinerSettings(
            root=streaming_root / "refiner_diffusers",
            gemma_root=streaming_root / "gemma3_12b",
            sink_size=_SINK_FRAMES,
            seed=config.seed,
            block_size=LATENT_FRAMES_PER_CHUNK,
            kv_max_frames=config.refiner_kv_max_frames,
        )
        self._pipeline = self._wm.SanaWMPipeline(
            config=upstream_config,
            model_path=streaming_root / "sana_dit" / "model.pt",
            device=self._device,
            refiner=refiner,
        )
        self._base_forward_long = self._pipeline.model.forward_long

        self._total_frames = 1 + config.max_chunks * PIXEL_FRAMES_PER_CHUNK
        self._stage1_iter: Any = None
        self._refiner_runner: Any = None
        self._vae_decoder: Any = None
        self._z: Any = None
        self._raymap: Any = None
        self._chunk_plucker: Any = None
        self._intrinsics: np.ndarray | None = None
        self._poses: list[np.ndarray] = []
        self._trajectory: np.ndarray | None = None
        self._trajectory_cursor = 1
        self._integrator: Any = None
        self._velocity: Any = None
        self._last_controls: set[str] = set()
        self._chunk_index = 0

    @property
    def chunk_index(self) -> int:
        """Return the count of chunks completed in the active rollout."""
        return self._chunk_index

    @property
    def trajectory_frames(self) -> int | None:
        """Return the finite trajectory length, or null for live controls."""
        if self._trajectory is None:
            return None
        return int(self._trajectory.shape[0])

    def _install_pinned_text_encoder(self, root: Path) -> None:
        """Make the upstream builder load its Gemma encoder from a pinned snapshot."""
        torch = self._torch

        def get_text_encoder(
            name: str = "gemma-2-2b-it", device: str = "cuda"
        ) -> tuple[Any, Any]:
            if name != "gemma-2-2b-it":
                raise ValueError(
                    f"SANA-WM streaming requires gemma-2-2b-it, got {name!r}"
                )
            transformers = importlib.import_module("transformers")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                root, local_files_only=True
            )
            tokenizer.padding_side = "right"
            encoder = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    root,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
                .get_decoder()
                .to(device)
            )
            return tokenizer, encoder

        self._wm.get_tokenizer_and_text_encoder = get_text_encoder

    def reset(
        self,
        image_source: Path | UploadedFile,
        prompt: str,
        seed: int,
        *,
        intrinsics_source: Path | UploadedFile | None,
        trajectory: np.ndarray | None,
    ) -> None:
        """Initialize fresh upstream caches from an image, prompt, and camera source."""
        with self._torch.inference_mode():
            self._initialize_rollout(
                image_source,
                prompt,
                seed,
                intrinsics_source=intrinsics_source,
                trajectory=trajectory,
            )

    def _initialize_rollout(
        self,
        image_source: Path | UploadedFile,
        prompt: str,
        seed: int,
        *,
        intrinsics_source: Path | UploadedFile | None,
        trajectory: np.ndarray | None,
    ) -> None:
        """Build per-world state while the public reset holds inference mode."""
        self._release_rollout()
        image = _open_image(image_source)
        cropped, src_size, resized_size, crop_offset = self._wm.resize_and_center_crop(
            image
        )
        intrinsics = self._resolve_intrinsics(image, intrinsics_source)
        intrinsics = self._wm.transform_intrinsics_for_crop(
            intrinsics,
            src_size,
            resized_size,
            crop_offset,
        )
        self._intrinsics = self._fit_intrinsics(intrinsics, self._total_frames)
        self._trajectory = self._normalize_trajectory(trajectory)
        self._trajectory_cursor = 1
        self._poses = [np.eye(4, dtype=np.float32)]
        self._integrator = self._camera_control.CameraPoseIntegrator(
            math.radians(self._config.pitch_limit_degrees)
        )
        self._velocity = self._camera_control.VelocityState()
        self._last_controls = set()
        self._chunk_index = 0

        refiner_prompt, refiner_mask = self._pipeline._get_streaming_refiner_prompt(
            prompt
        )
        vae_encoder = getattr(self._pipeline.vae, "encoder", None)
        if vae_encoder is not None:
            vae_encoder.to(self._device)
        image_tensor = (
            (self._transforms.ToTensor()(cropped) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
        )
        first_latent = self._wm.vae_encode(
            self._pipeline.config.vae.vae_type,
            self._pipeline.vae,
            image_tensor.to(self._device, dtype=self._pipeline.vae_dtype),
            device=self._device,
        ).to(self._pipeline.weight_dtype)
        self._pipeline._offload_vae_encoder_for_streaming()
        cond, cond_mask, neg, _neg_mask = self._pipeline._get_streaming_stage1_prompt(
            prompt, ""
        )

        latent_t = 1 + self._config.max_chunks * LATENT_FRAMES_PER_CHUNK
        latent_h = self._wm.TARGET_HEIGHT // self._pipeline.config.vae.vae_stride[-1]
        latent_w = self._wm.TARGET_WIDTH // self._pipeline.config.vae.vae_stride[-1]
        generator = self._torch.Generator(device=self._device).manual_seed(seed)
        self._z = self._torch.randn(
            1,
            int(first_latent.shape[1]),
            latent_t,
            latent_h,
            latent_w,
            dtype=self._pipeline.weight_dtype,
            device=self._device,
            generator=generator,
        )
        self._z[:, :, :1] = first_latent
        self._raymap = self._torch.zeros(
            1,
            latent_t,
            20,
            dtype=self._pipeline.weight_dtype,
            device=self._device,
        )
        self._chunk_plucker = self._torch.zeros(
            1,
            6 * _VAE_TIME_STRIDE,
            latent_t,
            latent_h,
            latent_w,
            dtype=self._pipeline.weight_dtype,
            device=self._device,
        )
        chunk_index = self._chunk_module.get_chunk_index_from_config(
            self._pipeline.config, num_frames=latent_t
        )
        model_kwargs: dict[str, Any] = {
            "data_info": {
                "img_hw": self._torch.tensor(
                    [[self._wm.TARGET_HEIGHT, self._wm.TARGET_WIDTH]],
                    dtype=self._torch.float,
                    device=self._device,
                ),
                "condition_frame_info": {0: 0.0},
            },
            "mask": cond_mask,
            "camera_conditions": self._raymap,
            "chunk_plucker": self._chunk_plucker,
        }
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index
        self._pipeline.model.forward_long = self._base_forward_long
        solver = self._sampler_module.SelfForcingFlowEulerCamCtrl(
            self._pipeline.model,
            condition=cond,
            uncondition=neg,
            cfg_scale=1.0,
            flow_shift=8.0,
            model_kwargs=model_kwargs,
            base_chunk_frames=LATENT_FRAMES_PER_CHUNK,
            num_cached_blocks=self._config.num_cached_blocks,
            sink_token=True,
            use_softmax_attention=True,
        )
        self._stage1_iter = solver.sample_chunks(
            self._z,
            steps=4,
            generator=generator,
            denoising_step_list=list(_DENOISING_STEPS),
        )

        self._pipeline.refiner.move_video_modules(self._device)
        self._pipeline.refiner.offload_video_unused_audio_modules("cpu")
        self._pipeline.model.to(self._device)
        self._pipeline._move_vae_decoder_for_streaming(self._device)
        sigmas = self._torch.tensor(
            self._refiner_module.STAGE_2_DISTILLED_SIGMA_VALUES,
            dtype=self._torch.float32,
            device=self._device,
        )
        self._refiner_runner = self._refiner_module.RefinerChunkRunner(
            self._pipeline.refiner,
            prompt_embeds=refiner_prompt,
            prompt_attention_mask=refiner_mask,
            fps=float(FPS),
            sigmas=sigmas,
            source_sink_frames=_SINK_FRAMES,
            block_size=LATENT_FRAMES_PER_CHUNK,
            kv_max_frames=self._config.refiner_kv_max_frames,
            seed=seed,
            spatial_shape=(latent_h, latent_w),
            n_active_frames=latent_t - _SINK_FRAMES,
            latent_channels=int(self._z.shape[1]),
            batch_size=1,
        )
        self._vae_decoder = self._vae_streaming_module.CausalVaeStreamingDecoder(
            self._pipeline.vae
        )
        self._vae_decoder.reset()

    def _resolve_intrinsics(
        self,
        image: Image.Image,
        source: Path | UploadedFile | None,
    ) -> np.ndarray:
        """Load native NumPy calibration or estimate it with pinned Pi3X."""
        if source is None:
            return self._estimate_intrinsics(image)
        if isinstance(source, UploadedFile):
            array = load_numpy_upload(source).astype(np.float32)
            return self._intrinsics_array_to_vec4(array)
        return self._wm.load_intrinsics(source, 1)

    def _intrinsics_array_to_vec4(self, array: np.ndarray) -> np.ndarray:
        """Convert every upstream-supported calibration shape to `(F, 4)`."""
        if array.shape == (4,):
            return array[None]
        if array.shape == (3, 3):
            return np.array(
                [[array[0, 0], array[1, 1], array[0, 2], array[1, 2]]],
                dtype=np.float32,
            )
        if array.ndim == 2 and array.shape[1] == 4:
            return array
        if array.ndim == 3 and array.shape[1:] == (3, 3):
            return np.stack(
                [
                    array[:, 0, 0],
                    array[:, 1, 1],
                    array[:, 0, 2],
                    array[:, 1, 2],
                ],
                axis=1,
            )
        raise ValueError(
            "intrinsics must have shape (4,), (F,4), (3,3), or (F,3,3); "
            f"got {array.shape}"
        )

    def _fit_intrinsics(self, array: np.ndarray, frames: int) -> np.ndarray:
        """Fit calibration to the bounded rollout using the upstream interpolation."""
        return self._wm._fit_intrinsics_sequence(
            np.asarray(array, dtype=np.float32), frames
        )

    def _estimate_intrinsics(self, image: Image.Image) -> np.ndarray:
        """Estimate calibration with the pinned Pi3X implementation and weights."""
        source, weights = resolve_pi3x_assets(self._config)
        _prepend_import_path(source)
        pi3x_module = importlib.import_module("pi3.models.pi3x")
        geometry = importlib.import_module("pi3.utils.geometry")
        width, height = image.size
        pixel_limit = 255_000
        scale = math.sqrt(pixel_limit / (width * height))
        target_w, target_h = width * scale, height * scale
        columns, rows = max(1, round(target_w / 14)), max(1, round(target_h / 14))
        while (columns * 14) * (rows * 14) > pixel_limit:
            if columns / rows > target_w / target_h:
                columns -= 1
            else:
                rows -= 1
        model_w, model_h = max(1, columns) * 14, max(1, rows) * 14
        resized = image.resize((model_w, model_h), Image.Resampling.LANCZOS)
        tensor = self._transforms.ToTensor()(resized).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self._device)
        model = pi3x_module.Pi3X.from_pretrained(weights).to(self._device).eval()
        model.disable_multimodal()
        model.requires_grad_(False)
        with (
            self._torch.no_grad(),
            self._torch.amp.autocast("cuda", dtype=self._torch.bfloat16),
        ):
            output = model(imgs=tensor)
        directions = self._torch.nn.functional.normalize(output["local_points"], dim=-1)
        matrix = (
            geometry.recover_intrinsic_from_rays_d(
                directions, force_center_principal_point=True
            )[0, 0]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        sx, sy = width / model_w, height / model_h
        fx, fy = float(matrix[0, 0] * sx), float(matrix[1, 1] * sy)
        cx, cy = float(matrix[0, 2] * sx), float(matrix[1, 2] * sy)
        fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        fov_y = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
        del model, output, directions, tensor
        self._torch.cuda.empty_cache()
        gc.collect()
        if not (
            self._wm.MIN_FOV_DEG < fov_x < self._wm.MAX_FOV_DEG
            and self._wm.MIN_FOV_DEG < fov_y < self._wm.MAX_FOV_DEG
        ):
            raise ValueError(
                f"Pi3X estimated an implausible field of view ({fov_x:.1f}°, "
                f"{fov_y:.1f}°); upload trusted intrinsics"
            )
        return np.array([[fx, fy, cx, cy]], dtype=np.float32)

    def _normalize_trajectory(self, trajectory: np.ndarray | None) -> np.ndarray | None:
        """Express a command-validated camera trajectory relative to its first pose."""
        if trajectory is None:
            return None
        values = np.asarray(trajectory, dtype=np.float32)
        first_inverse = np.linalg.inv(values[0]).astype(np.float32)
        return np.matmul(first_inverse[None], values).astype(np.float32)

    def generate_chunk(self, controls: set[Control]) -> np.ndarray:
        """Advance all three native caches once and return 24 RGB frames."""
        with self._torch.inference_mode():
            return self._generate_chunk(controls)

    def _generate_chunk(self, controls: set[Control]) -> np.ndarray:
        """Run one chunk while the public entry point holds inference mode."""
        if self._stage1_iter is None or self._refiner_runner is None:
            raise RuntimeError("SANA-WM rollout is not initialized")
        if self._chunk_index >= self._config.max_chunks:
            raise StopIteration("SANA-WM rollout reached its configured chunk bound")
        self._append_camera_poses(set(controls))
        self._write_camera_conditioning(self._chunk_index)
        chunk_idx, latent_view, start_f, end_f = next(self._stage1_iter)
        if int(chunk_idx) != self._chunk_index:
            raise RuntimeError(
                f"Stage-1 yielded chunk {chunk_idx}, expected {self._chunk_index}"
            )
        block_start = _SINK_FRAMES + self._chunk_index * LATENT_FRAMES_PER_CHUNK
        block_end = block_start + LATENT_FRAMES_PER_CHUNK
        if self._chunk_index == 0:
            clean_block = latent_view[:, :, _SINK_FRAMES:]
            sink = self._z[:, :, :_SINK_FRAMES]
        else:
            clean_block = latent_view
            sink = None
        if start_f > block_start or end_f < block_end:
            raise RuntimeError(
                f"Stage-1 latent range [{start_f}, {end_f}) does not cover "
                f"refiner block [{block_start}, {block_end})"
            )
        refined = self._refiner_runner.refine_block(
            block_idx=self._chunk_index,
            clean_block=clean_block,
            block_start=block_start,
            block_end=block_end,
            sink_seed_frames=sink,
        )
        decode_input = (
            self._torch.cat([self._z[:, :, :_SINK_FRAMES], refined], dim=2)
            if self._chunk_index == 0
            else refined
        )
        pixels = self._vae_decoder.decode_chunk(decode_input)
        frames = (
            (pixels.float() * 127.5 + 127.5)
            .clamp(0, 255)
            .to(self._torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .cpu()
            .numpy()[0]
        )
        if self._chunk_index == 0:
            frames = frames[1:]
        if frames.shape != (
            PIXEL_FRAMES_PER_CHUNK,
            self._wm.TARGET_HEIGHT,
            self._wm.TARGET_WIDTH,
            3,
        ):
            raise RuntimeError(f"unexpected decoded chunk shape: {frames.shape}")
        self._chunk_index += 1
        return frames

    def _append_camera_poses(self, controls: set[str]) -> None:
        """Append exactly one chunk of native camera-to-world poses."""
        if self._trajectory is not None:
            end = self._trajectory_cursor + PIXEL_FRAMES_PER_CHUNK
            if end > self._trajectory.shape[0]:
                raise TrajectoryCompleteError
            self._poses.extend(self._trajectory[self._trajectory_cursor : end])
            self._trajectory_cursor = end
            return
        target = self._camera_control.controls_to_target_velocity(
            controls,
            translation_speed=self._config.translation_speed,
            rotation_speed_rad=math.radians(self._config.rotation_speed_degrees),
        )
        for _ in range(PIXEL_FRAMES_PER_CHUNK):
            if controls - self._last_controls:
                self._velocity.snap_to(target)
            else:
                self._velocity.step_toward(target, 1.0 / FPS)
            self._last_controls = set(controls)
            self._poses.append(self._integrator.step(self._velocity).astype(np.float32))

    def _write_camera_conditioning(self, chunk_index: int) -> None:
        """Populate the current Stage-1 slice with upstream ray and Plücker features."""
        if self._intrinsics is None:
            raise RuntimeError("camera intrinsics are not initialized")
        poses = np.stack(self._poses, axis=0)
        latent_start = (
            0
            if chunk_index == 0
            else _SINK_FRAMES + chunk_index * LATENT_FRAMES_PER_CHUNK
        )
        latent_end = _SINK_FRAMES + (chunk_index + 1) * LATENT_FRAMES_PER_CHUNK
        latent_h = self._wm.TARGET_HEIGHT // self._pipeline.config.vae.vae_stride[-1]
        latent_w = self._wm.TARGET_WIDTH // self._pipeline.config.vae.vae_stride[-1]
        spatial_scale_x = latent_w / float(self._wm.TARGET_WIDTH)
        spatial_scale_y = latent_h / float(self._wm.TARGET_HEIGHT)
        for latent_index in range(latent_start, latent_end):
            pixel_index = latent_index * _VAE_TIME_STRIDE
            pose = poses[pixel_index]
            intrinsics = self._intrinsics[pixel_index].copy()
            intrinsics[[0, 2]] *= spatial_scale_x
            intrinsics[[1, 3]] *= spatial_scale_y
            ray = np.concatenate([pose.reshape(-1), intrinsics], axis=0)
            self._raymap[0, latent_index].copy_(
                self._torch.from_numpy(ray).to(
                    self._device, dtype=self._pipeline.weight_dtype
                )
            )

            start = max(0, pixel_index - (_VAE_TIME_STRIDE - 1))
            end = start + _VAE_TIME_STRIDE
            segment_poses = self._torch.from_numpy(poses[start:end]).float()
            segment_intrinsics = self._torch.from_numpy(
                self._intrinsics[start:end].copy()
            ).float()
            segment_intrinsics[:, [0, 2]] *= spatial_scale_x
            segment_intrinsics[:, [1, 3]] *= spatial_scale_y
            plucker = self._wm.compute_raymap(
                segment_intrinsics,
                segment_poses,
                latent_h,
                latent_w,
                use_plucker=True,
            )
            packed = plucker.permute(0, 3, 1, 2).reshape(
                6 * _VAE_TIME_STRIDE, latent_h, latent_w
            )
            self._chunk_plucker[0, :, latent_index].copy_(
                packed.to(self._device, dtype=self._pipeline.weight_dtype)
            )

    def _release_rollout(self) -> None:
        """Release per-world tensors while keeping loaded model weights resident."""
        iterator = self._stage1_iter
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()
        self._pipeline.model.forward_long = self._base_forward_long
        self._stage1_iter = None
        self._refiner_runner = None
        if self._vae_decoder is not None:
            self._vae_decoder.reset()
        self._vae_decoder = None
        self._z = None
        self._raymap = None
        self._chunk_plucker = None
        gc.collect()
        self._torch.cuda.empty_cache()

    def end_session(self) -> None:
        """Release the active rollout's autoregressive state."""
        self._release_rollout()
