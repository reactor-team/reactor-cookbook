"""StreamDiffusionV2 (1.3B, video-to-video), the model half.

A plain class with ``load`` / ``generate`` / ``reset`` that wraps the upstream
source (the pinned checkout ``streamdiffusionv2_assets`` prepares) and imports
nothing from ``reactor_runtime``. One ``generate()`` call is one iteration of the source's
own per-chunk loop (``streamv2v/inference.py``, ``run_inference``):

* the first step of a run takes 5 input frames, encodes the prompt, runs the
  source's ``prepare`` (both denoising steps on the first block) and returns 5
  output frames;
* every later step takes 4 input frames (one latent frame), adapts the noise
  strength to the motion between frames exactly as the source does, and runs
  one batched forward of the "stream batch": two latents at two noise levels.
  The output therefore trails the input by one chunk: step 1 returns nothing,
  step k >= 2 returns the 4 frames that rewrite input chunk k - 1.

One fix to the source's state handling: the rolling KV cache's eviction queue
(``self_attn.evict_idx``) is cleared at the start of every run; the source
never clears it, so its own web demo carries one session's queue into the next.

What the port adds to the source, both taken from the source's own streaming
code or architecture:

* the position refresh of ``SingleGPUInferencePipeline.run_stream_batch``
  (``t_refresh``), so a run can last indefinitely;
* a prompt change at a chunk boundary: the new text is encoded once and the
  cross-attention cache is rebuilt, the self-attention cache is kept.
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

@dataclass(frozen=True)
class StreamInput:
    """What one step needs.

    Attributes:
        frames: Input frames, uint8 ``(N, H, W, 3)`` RGB at any size, or a list of
            ``(H_i, W_i, 3)`` frames when sizes differ; each is resized to the serving
            resolution here. ``N`` is 5 on the first step of a
            run and 4 on every later step.
        prompt: The text the model reads at this chunk; may be empty.
        seed: The run's seed; read on the first step of a run only.
        noise_scale: The run's starting noise strength, 0..1; read on the
            first step of a run only.
        run_id: Identifies the run. A value the model has not seen starts a
            new run from these frames.
    """

    frames: np.ndarray | list
    prompt: str
    seed: int
    noise_scale: float
    run_id: int


@dataclass(frozen=True)
class StreamResult:
    """What one step produced.

    Attributes:
        frames: Output frames, uint8 ``(n, 480, 832, 3)`` RGB, or ``None`` on
            the step that fills the stream batch.
        index: The step index within the run, from 0.
        run_id: The run the step belongs to.
        noise_scale: The noise strength this step used (adapted per chunk).
        prompt: The prompt applied at this step.
    """

    frames: np.ndarray | None
    index: int
    run_id: int
    noise_scale: float
    prompt: str


class StreamDiffusionV2Model:
    """The vendored single-GPU stream pipeline behind ``load`` / ``generate`` / ``reset``."""

    FIRST_CHUNK_FRAMES = 5
    CHUNK_FRAMES = 4

    def __init__(self) -> None:
        self._gpu: ThreadPoolExecutor | None = None
        self.pm = None
        self._run: _Run | None = None
        self._prompt_cache: dict[str, dict] = {}

    def load(self, config_path: str, source_path: str, weights_root: str) -> None:
        """Load every weight from *weights_root* and the code from *source_path*, then warm up.

        All GPU work runs on one thread this class owns. PyTorch keeps per-thread GPU
        handles and plan caches (measured: a run on a thread other than the warm-up's
        paid 1.2-2.2 s per chunk for its first 6 chunks, against 0.19 s), and the
        runtime calls load() and generate() from different threads; one thread for
        both makes the warm-up count for the first session.
        """
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sdv2-gpu")
        self._gpu.submit(self._load, Path(config_path), Path(source_path), Path(weights_root)).result()

    def _load(self, config_path: Path, source_path: Path, weights_root: Path) -> None:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg = document["model"]
        ckpt_dir = weights_root / document["checkpoint"]["path"]
        # The source finds wan_models/ (text encoder, VAE, base config) through this variable.
        os.environ["STREAMDIFFUSIONV2_ROOT"] = str(weights_root)
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

        import torch
        import torchvision.transforms.functional as TF
        from argparse import Namespace

        from streamv2v import inference as inf
        from streamv2v.inference_common import merge_cli_config

        self.torch, self.TF, self.inf = torch, TF, inf
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        self.height, self.width = int(cfg["height"]), int(cfg["width"])
        args = Namespace(
            config_path=str(source_path / cfg["source_config"]), step=int(cfg["step"]),
            height=self.height, width=self.width, model_type=cfg["model_type"],
            use_taehv=False, use_tensorrt=False, fast=False, normalize_latents=False,
            profile=False, t2v=False, target_fps=None, fixed_noise_scale=False,
        )
        self.device = torch.device("cuda")
        config = merge_cli_config(args.config_path, args)
        self.pm = inf.SingleGPUInferencePipeline(config, self.device)
        self.pm.load_model(str(ckpt_dir))
        self.pipe = self.pm.pipeline
        # inference_stream writes the adapted timestep into denoising_step_list[0]
        # (pipeline.timestep aliases it), so each run restores the schedule first.
        self.schedule = self.pipe.denoising_step_list.clone()
        self.num_steps = len(self.schedule)
        self.t_refresh = int(cfg["t_refresh"])
        self.fsl = self.pipe.frame_seq_length
        self.warmup(int(cfg["warmup_steps"]))

    # -- the step ---------------------------------------------------------------

    def generate(self, input: StreamInput) -> StreamResult:
        """Run one chunk: the first chunk of a new run, or the next chunk of this one."""
        # Grad mode is per thread: load() and the runtime's step loop run on different
        # threads, so the source's global set_grad_enabled(False) does not reach here.
        return self._gpu.submit(self._generate, input).result()

    def _generate(self, input: StreamInput) -> StreamResult:
        with self.torch.no_grad():
            if self._run is None or self._run.run_id != input.run_id:
                return self._start_run(input)
            return self._next_chunk(input)

    def reset(self) -> None:
        """Forget the run: caches, stream batch, prompt encodings."""
        if self._gpu is not None:
            self._gpu.submit(self._reset).result()

    def _reset(self) -> None:
        self._run = None
        self._prompt_cache.clear()
        if self.pm is None:
            return
        with self.torch.no_grad():
            self.pm.reset_stream_state(reset_vae_flags=True)
            self.pipe.denoising_step_list.copy_(self.schedule)
        # The rolling KV cache's eviction queue lives on each attention module and the
        # source never clears it (models/wan/causal_model.py:194, set once at construction),
        # so a second run would inherit the first run's queue. Clearing it gives every run
        # the state a fresh process has, which is what the baseline ran from.
        for block in self.pipe.generator.model.blocks:
            block.self_attn.evict_idx = None

    # -- internals --------------------------------------------------------------

    def _images(self, frames):
        """uint8 (N,H,W,3), or a list of (H_i,W_i,3) frames, -> bf16 (1,3,N,h,w) in [-1,1].

        Each frame is resized on its own, as load_mp4_as_tensor does (TF.resize per frame on uint8 CHW),
        so a list of frames of different sizes goes through the same operation as one array.
        """
        torch, TF = self.torch, self.TF
        per_frame = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1) for f in frames]  # c h w, uint8
        video = torch.stack(
            [TF.resize(f, (self.height, self.width), antialias=True) for f in per_frame],
            dim=1,
        )
        video = video.float() / 127.5 - 1.0
        return video.unsqueeze(0).to(dtype=torch.bfloat16).to(self.device)

    def _noisy_latents(self, images, noise_scale: float):
        torch = self.torch
        latents = self.pm._timed_stream_encode(images)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        return noise * noise_scale + latents * (1 - noise_scale)

    def _decode(self, pred) -> np.ndarray:
        video = self.pm._timed_stream_decode(pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        return video.detach().cpu().float().numpy()

    @staticmethod
    def to_uint8(frames: np.ndarray) -> np.ndarray:
        """float [0,1] -> uint8 exactly as the source's writer (diffusers export_to_video) does."""
        return (frames * 255).astype(np.uint8)

    def _start_run(self, input: StreamInput) -> StreamResult:
        torch = self.torch
        n = len(input.frames)
        if n != self.FIRST_CHUNK_FRAMES:
            raise ValueError(f"the first step of a run takes {self.FIRST_CHUNK_FRAMES} frames, got {n}")
        self._reset()
        # set_seed() of models/util.py, at the point the source's first draw follows it.
        random.seed(input.seed)
        np.random.seed(input.seed)
        torch.manual_seed(input.seed)
        torch.cuda.manual_seed_all(input.seed)
        images = self._images(input.frames)
        noisy = self._noisy_latents(images, input.noise_scale)
        current_end = self.fsl * (1 + (n - 1) // 4)
        pred = self.pm.prepare_pipeline(
            text_prompts=[input.prompt], noise=noisy, current_start=0, current_end=current_end
        )
        self._prompt_cache[input.prompt] = self.pipe.conditional_dict
        out = self._decode(pred)
        self._run = _Run(
            run_id=input.run_id, prompt=input.prompt, init_noise_scale=float(input.noise_scale),
            noise_scale=float(input.noise_scale), current_end=current_end,
            last_image=images[:, :, [-1]], processed=0, index=0,
        )
        return StreamResult(self.to_uint8(out), 0, input.run_id, float(input.noise_scale), input.prompt)

    def _apply_prompt(self, prompt: str) -> None:
        """Encode a new prompt once and rebuild the cross-attention cache at this boundary."""
        cond = self._prompt_cache.get(prompt)
        if cond is None:
            cond = self.pipe.text_encoder(text_prompts=[prompt])
            self._prompt_cache[prompt] = cond
        embeds = cond["prompt_embeds"]
        if embeds.shape[0] == 1:
            embeds = embeds.repeat(self.pipe.batch_size, 1, 1)
        self.pipe.conditional_dict = {**cond, "prompt_embeds": embeds}
        for block in self.pipe.crossattn_cache:
            block["is_init"] = False
        self._run.prompt = prompt

    def _next_chunk(self, input: StreamInput) -> StreamResult:
        run, pipe = self._run, self.pipe
        n = len(input.frames)
        if n != self.CHUNK_FRAMES:
            raise ValueError(f"a step after the first takes {self.CHUNK_FRAMES} frames, got {n}")
        if input.prompt != run.prompt:
            self._apply_prompt(input.prompt)
        current_start = run.current_end
        current_end = run.current_end + (self.CHUNK_FRAMES // 4) * self.fsl
        # Position refresh from run_stream_batch: keeps RoPE positions bounded on a long run.
        if current_start // self.fsl >= self.t_refresh:
            current_start = pipe.kv_cache_length - self.fsl
            current_end = current_start + (self.CHUNK_FRAMES // 4) * self.fsl
        images = self._images(input.frames)
        noise_scale, current_step = self.inf.compute_noise_scale_and_step(
            input_video_original=self.torch.cat([run.last_image, images], dim=2),
            end_idx=self.CHUNK_FRAMES + 1, chunk_size=self.CHUNK_FRAMES,
            noise_scale=run.noise_scale, init_noise_scale=run.init_noise_scale,
        )
        noisy = self._noisy_latents(images, noise_scale)
        pred = pipe.inference_stream(
            noise=noisy, current_start=current_start, current_end=current_end, current_step=current_step
        )
        run.processed += 1
        run.index += 1
        out = None
        if run.processed >= self.num_steps:
            out = self.to_uint8(self._decode(pred[[-1]]))
        run.current_end = current_end
        run.last_image = images[:, :, [-1]]
        run.noise_scale = noise_scale
        return StreamResult(out, run.index, run.run_id, float(noise_scale), run.prompt)

    def warmup(self, steps: int) -> None:
        """Run a throwaway run at a client-like size so kernels and caches are ready, then forget it."""
        if steps <= 0:
            return
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, size=(self.FIRST_CHUNK_FRAMES, 480, 640, 3), dtype=np.uint8)
        self._generate(StreamInput(frames, "warm up", 0, 0.8, run_id=-1))
        for i in range(steps):
            f = rng.integers(0, 256, size=(self.CHUNK_FRAMES, 480, 640, 3), dtype=np.uint8)
            self._generate(StreamInput(f, "warm up" if i < steps // 2 else "warm up two", 0, 0.8, run_id=-1))
        self.torch.cuda.synchronize()
        self._reset()


@dataclass
class _Run:
    run_id: int
    prompt: str
    init_noise_scale: float
    noise_scale: float
    current_end: int
    last_image: object
    processed: int
    index: int
