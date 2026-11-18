"""Incremental MovieRoll-Distilled sampling using upstream window/KV/VAE code.

The orchestration follows MovieRollDMDPipeline._execute_rollout(SAMPLING).
There is no finite drain, so the same five-slot frontier continues until Stop.
"""

import logging
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from movieroll_history import LatentHistory
from movieroll_model import (
    CHUNK_SIZE,
    HEIGHT,
    MAX_PROMPTS,
    WIDTH,
    WINDOW_SIZE,
    Anchor,
    Settings,
)

logger = logging.getLogger(__name__)


class DeviceHistory:
    """Restore selected chunks with their original dtype for native recompute."""

    def __init__(self, settings: Settings, device):
        self.archive = LatentHistory(
            settings.sessions, max_bytes=settings.history_bytes
        )
        self.device = device

    def __len__(self):
        return len(self.archive)

    def append(self, chunk):
        import torch

        chunk = chunk.detach().cpu().contiguous()
        if chunk.dtype == torch.bfloat16:
            chunk = chunk.view(torch.uint16)
        elif chunk.dtype != torch.float32:
            raise ValueError("Expected native BF16 image or FP32 generated latents")
        self.archive.append(chunk.numpy())

    def __getitem__(self, index):
        import torch

        array = self.archive[index]
        tensor = torch.from_numpy(array)
        if array.dtype == np.uint16:
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(self.device)

    def close(self):
        self.archive.close()


@dataclass
class Run:
    pipe: object
    profile: dict
    generator: object
    history: DeviceHistory
    prompt: str
    condition: tuple
    context_noise: object
    first_size: int
    round_index: int = 0
    position: int = 0
    event_splits: list[int] = field(default_factory=list)
    decode_pending: object | None = None
    event_changed: bool = False


class MovieRollBackend:
    def __init__(self, settings: Settings, *, rank: int = 0):
        import torch

        from movieroll_attention import configure
        from movieroll_kernels import configure as configure_kernels
        from movieroll_optimizations import TimestepProjections, configure_layout

        configure()
        from movieroll.commons.parallel_states import (
            ParallelDims,
            initialize_sequence_parallel_state_v2,
            initialize_vae_parallel_group,
            nccl_info,
        )
        from movieroll.commons.profiles import load_model_dir
        from movieroll.commons.streaming_config import StreamingConfig
        from movieroll.models.autoencoders.vae_loader import load_vae
        from movieroll.models.text_encoders import PROMPT_TEMPLATE
        from movieroll.models.text_encoders.byT5 import load_glyph_byT5_v2
        from movieroll.pipelines.movieroll_dmd_pipeline import MovieRollDMDPipeline
        from movieroll.schedulers.scheduling_flow_match_streaming import (
            FlowMatchStreamingScheduler,
        )
        from movieroll.utils.load import load_text_encoder, load_transformer

        self.prefix_attention = configure_layout()
        configure_kernels()
        self.settings = settings
        self.rank = rank
        self.use_sp = settings.gpu_count > 1
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        if self.use_sp:
            dims = ParallelDims(1, 1, settings.gpu_count, settings.gpu_count)
            mesh = dims.build_mesh(device_type="cuda")
            initialize_sequence_parallel_state_v2(dims, mesh)
        vae_group = initialize_vae_parallel_group("sp" if self.use_sp else "none")
        weights = settings.weights
        self.vae, _, _, _ = load_vae(
            "fp16",
            str(weights / "vae"),
            logger=logger,
            device=self.device,
            streaming=True,
        )
        self.vae.eval().requires_grad_(False)
        if vae_group is not None:
            self.vae.enable_parallel_decode(vae_group)
        self.profiles = {
            mode: load_model_dir(str(weights / "movieroll/distilled" / mode))
            for mode in ("t2v", "i2v")
        }
        profile = self.profiles["t2v"]
        text = load_text_encoder(
            str(weights / "text_encoder/llm"),
            profile["text_len"],
            PROMPT_TEMPLATE[profile["prompt_template"]],
            PROMPT_TEMPLATE[profile["prompt_template_video"]],
            self.device,
            text_encoder_precision="fp16",
            logger=logger,
        )
        text.model.eval().requires_grad_(False)
        glyph = weights / "text_encoder/Glyph-SDXL-v2"
        font, color = (
            glyph / "assets/multilingual_10-lang_idx.json",
            glyph / "assets/color_idx.json",
        )
        byt5 = load_glyph_byT5_v2(
            self.device,
            byT5_google_path=str(weights / "text_encoder/byt5-small"),
            byT5_ckpt_path=str(glyph / "checkpoints/byt5_model.pt"),
            color_ann_path=str(color),
            font_ann_path=str(font),
            byt5_max_length=profile["byt5_max_length"],
        )
        byt5["byt5_model"].eval()
        self.pipelines = {}
        self.timestep_projections = []
        for mode, profile in self.profiles.items():
            streaming = StreamingConfig(
                **{
                    key: profile[key]
                    for key in (
                        "max_num_latent_frames",
                        "chunk_size",
                        "window_size",
                        "full_history_block_num",
                        "kv_recompute_enabled",
                        "kv_recompute_threshold",
                        "kv_recompute_recent_k",
                    )
                }
            )
            model = load_transformer(
                profile["model_type"],
                str(weights / "movieroll/distilled" / mode / "model.safetensors"),
                {
                    "device": self.device,
                    "dtype": torch.bfloat16,
                    "streaming_config": streaming,
                },
                logger,
            )
            model.to(self.device).eval().requires_grad_(False)
            self.timestep_projections.append(TimestepProjections(model))
            pipe = MovieRollDMDPipeline(
                vae=self.vae,
                text_encoder=text,
                transformer=model,
                scheduler=FlowMatchStreamingScheduler(
                    num_train_timesteps=1000, shift=profile["timestep_shift"]
                ),
                precision="bf16",
                vae_precision="fp16",
                text_len=profile["text_len"],
                prompt_format_font_path=str(font),
                prompt_format_color_path=str(color),
                byt5_kwarg=byt5,
                streaming_config=streaming,
                device=self.device,
                is_eval=False,
            )
            pipe.transformer.kv_cache_manager.verbose = False
            pipe._use_cfg = False
            if self.use_sp:
                pipe.set_sync_group(nccl_info.sp_group)
            self.pipelines[mode] = pipe
        self.run: Run | None = None
        logger.info("MovieRoll weights resident; attention=FA4")

    def reset(self) -> None:
        if self.run is not None:
            self.run.history.close()
            self.run = None
        for pipe in self.pipelines.values():
            pipe.transformer.kv_cache_manager.clear_kv_cache()
        for projections in self.timestep_projections:
            projections.clear()
        self.prefix_attention.clear()
        self.vae.clear_cache()

    def begin(self, anchor: Anchor, prompt: str) -> None:
        import torch
        from movieroll.commons.parallel_states import nccl_info
        from movieroll.models.transformers.modules.context_noise import (
            apply_context_noise,
        )
        from movieroll.models.transformers.modules.rolling_queue import RollingQueue
        from movieroll.schedulers.scheduling_flow_match_streaming import (
            StreamingSchedulerWrapper,
        )
        from movieroll.utils.communications import bcast_in_group

        if anchor.mode not in self.pipelines:
            raise ValueError("Unknown MovieRoll checkpoint")
        if anchor.mode == "i2v" and (
            anchor.image is None
            or anchor.image.shape != (HEIGHT, WIDTH, 3)
            or anchor.image.dtype != np.uint8
        ):
            raise ValueError("I2V requires a fitted uint8 RGB image")
        pipe, profile = self.pipelines[anchor.mode], self.profiles[anchor.mode]
        generator = torch.Generator(device="cpu").manual_seed(anchor.seed)
        pipe.reset_sync_random(generator)
        pipe.scheduler.set_custom_timesteps(
            profile["denoising_step_list"], device=self.device
        )
        pipe.scheduler_wrapper = StreamingSchedulerWrapper(
            pipe.streaming_config, pipe.scheduler, WINDOW_SIZE, WINDOW_SIZE
        )
        pipe.rolling_queue = RollingQueue()
        with torch.no_grad():
            condition = pipe.encode_text_condition(prompt)
        sigma = profile["context_noise"]
        self.run = Run(
            pipe=pipe,
            profile=profile,
            generator=generator,
            history=DeviceHistory(self.settings, self.device),
            prompt=prompt,
            condition=condition,
            context_noise=partial(
                apply_context_noise,
                sigma_max=sigma,
                force_equal_sigma=True,
                generator=generator,
            )
            if sigma > 0
            else None,
            first_size=CHUNK_SIZE + 1 if anchor.mode == "t2v" else CHUNK_SIZE,
        )
        if anchor.mode == "i2v":
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.float16),
            ):
                pixels = (
                    torch.from_numpy(anchor.image.copy())
                    .permute(2, 0, 1)
                    .float()
                    .div(255)
                    .sub(0.5)
                    .div(0.5)
                )
                latent = self.vae.encode(
                    pixels[None, :, None].to(self.device)
                ).latent_dist.mode()
            if self.vae.config.shift_factor is not None:
                latent = latent - float(self.vae.config.shift_factor)
            latent = latent * float(self.vae.config.scaling_factor)
            latent = latent.detach().to(dtype=torch.bfloat16, device=self.device)
            if self.use_sp:
                bcast_in_group(latent, nccl_info.sp_group)
            self.run.history.append(latent)
            with torch.no_grad():
                self.run.position = self._encode(latent)
            self.run.decode_pending = latent

    def _encode(self, latent):
        run = self.run
        assert run is not None
        return run.pipe._encode_clean_chunk_kv_with_context_noise(
            latent,
            kv_frame_position=run.position,
            target_dtype=run.pipe.target_dtype,
            autocast_enabled=True,
            context_noise_fn=run.context_noise,
            use_sp=self.use_sp,
        )

    def _round(self) -> None:
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        from movieroll.commons.parallel_states import nccl_info
        from movieroll.commons.streaming_config import RollingStageType
        from movieroll.utils.communications import bcast_in_group

        run = self.run
        assert run is not None
        pipe = run.pipe
        size = run.first_size if run.round_index == 0 else CHUNK_SIZE
        shape = (
            run.condition[0].shape[0],
            pipe.transformer.config.in_channels,
            size,
            HEIGHT // pipe.vae_scale_factor,
            WIDTH // pipe.vae_scale_factor,
        )
        if not self.use_sp or self.rank == 0:
            noise = randn_tensor(
                shape,
                generator=run.generator,
                device=self.device,
                dtype=pipe.target_dtype,
            )
        else:
            noise = torch.empty(shape, device=self.device, dtype=pipe.target_dtype)
        if self.use_sp:
            bcast_in_group(noise, nccl_info.sp_group)
        pipe.rolling_queue.add_new_noisy_chunk(noise)
        pipe._run_window_round(
            round_idx=run.round_index,
            rolling_stage=RollingStageType.PREFILL
            if run.round_index < WINDOW_SIZE
            else RollingStageType.DECODE,
            steps_per_window=1,
            event_cond=run.condition,
            kv_frame_position=run.position,
            timesteps=pipe.scheduler.timesteps,
            target_dtype=pipe.target_dtype,
            autocast_enabled=True,
            should_grad=False,
            compute_predicted_clean_window=False,
        )
        run.round_index += 1

    def step(self, prompt: str) -> np.ndarray:
        import torch

        from movieroll_optimizations import rgb_frames

        run = self.run
        assert run is not None
        pipe = run.pipe
        with torch.no_grad():
            if prompt != run.prompt:
                if len(run.event_splits) >= MAX_PROMPTS - 1:
                    raise ValueError("The native event context is full")
                condition = pipe.encode_text_condition(prompt)
                # At this boundary one previous-event chunk remains to be popped.
                # This is the same output-coordinate boundary as native prompt-list.
                run.event_splits.append(len(run.history) + 1)
                run.condition, run.prompt = condition, prompt
                run.event_changed = True
            while run.round_index < WINDOW_SIZE:
                self._round()
            clean = pipe.rolling_queue.pop_clean_chunk()
            run.history.append(clean)
            event_switch = run.event_changed
            do_recompute = (
                event_switch
                or pipe.transformer.kv_cache_manager.should_recompute(
                    pipe.streaming_config.kv_recompute_threshold
                )
            )
            if do_recompute:
                selected = pipe._select_recompute_chunks(run.history, run.event_splits)
                do_recompute = event_switch or len(selected) < len(run.history)
            if do_recompute:
                run.position = pipe._kv_recompute(
                    output_latents=run.history,
                    target_dtype=pipe.target_dtype,
                    autocast_enabled=True,
                    context_noise_fn=run.context_noise,
                    use_sp=self.use_sp,
                    trigger_reason="event_switch" if event_switch else "threshold",
                    selected_indices=selected,
                )
            else:
                run.position = self._encode(clean)
            run.event_changed = False
            self._round()
            # Match generate._streaming_decode_and_save: cast and unscale before
            # splitting into groups of four, retaining the causal T-cache.
            pending = (
                clean
                if run.decode_pending is None
                else torch.cat((run.decode_pending, clean), dim=2)
            )
            complete = pending.shape[2] // CHUNK_SIZE * CHUNK_SIZE
            ready, run.decode_pending = (
                pending[:, :, :complete],
                pending[:, :, complete:].clone(),
            )
            ready = ready.to(dtype=torch.float16) / float(
                self.vae.config.scaling_factor
            )
            if self.vae.config.shift_factor is not None:
                ready = ready + float(self.vae.config.shift_factor)
            frames = []
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for block in ready.split(CHUNK_SIZE, dim=2):
                    pixels = self.vae.decode(
                        block, return_dict=False, streaming_mode=True
                    )[0]
                    if self.rank == 0:
                        frames.append(rgb_frames(pixels))
            if self.rank != 0:
                return np.empty((0, HEIGHT, WIDTH, 3), dtype=np.uint8)
            return np.ascontiguousarray(np.concatenate(frames))
