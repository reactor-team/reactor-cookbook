"""The FastVideo engine behind FastH3: GPU work, and nothing client-facing.

One :class:`FastH3Backend` owns the eight-GPU ``VideoGenerator``, the
environment profile it must be built under, the load-time warm-up, and a single
persistent worker thread that builds clips one at a time. ``fasth3.py`` submits
work with :meth:`FastH3Backend.submit` and polls the returned
:class:`ClipJob`; nothing in here knows about commands, tracks, or the queue.

torch, torchaudio and fastvideo are imported lazily inside methods, so
importing this module — which rendering the schema does — needs none of them.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from reactor_runtime.log import get_logger

import fasth3_clip_plan as clip_plan
from fasth3_assets import FastH3Config

logger = get_logger(__name__)

FRAME_RATE = clip_plan.FPS

# WebRTC-native rate every clip's waveform is resampled to. The checkpoint's
# audio decoder is 32 kHz; the wire is 48 kHz.
OUTPUT_SAMPLE_RATE = 48_000
NATIVE_SAMPLE_RATE = 32_000

# How often a blocking wait on the worker re-checks. Used only during load().
_WORKER_POLL_SECONDS = 0.1

# What the warm-up builds. Never reaches a client: warm-up output is discarded,
# and its only job is to be a syntactically ordinary prompt.
WARMUP_PROMPT = "A slow cinematic shot of sunlight moving across a quiet room."

# Every prompt is padded (or token-truncated) to exactly this many tokens
# before it reaches the engine. Regional torch.compile is keyed on the packed
# sequence length, which includes the prompt's token count, so a novel prompt
# length would otherwise recompile — measured at ~23 s against ~15 s for the
# clip itself. One fixed length means one compiled shape, captured by the
# warm-up and reused by every clip. 256 comfortably holds the 800-character
# prompt cap.
PROMPT_TOKENS = 256


class ClipJob:
    """The handle to one submitted build: its inputs, outcome, and completion.

    The error is carried back rather than only logged, so the submitter can
    report the failed clip to clients. ``cancelled`` set before the worker
    reaches the job skips the build entirely; set after, the build runs to
    completion and the submitter discards the result.
    """

    __slots__ = ("cancelled", "done", "error", "fn", "result")

    def __init__(self, fn) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.result: tuple[list[Any], Any] | None = None
        self.cancelled = False


class FastH3Backend:
    """Build FastH3 clips on demand, serialised on one worker thread.

    The GPU work itself lives in the engine processes FastVideo spawns; the
    thread exists to serialise submissions and to give teardown a single handle
    to wait on.
    """

    def __init__(self, config: FastH3Config, model_path: Path) -> None:
        """Remember the recipe and the weights location; nothing loads yet."""
        self._config = config
        self._model_path = model_path
        self._jobs: queue.Queue[ClipJob] = queue.Queue()
        self._worker: threading.Thread | None = None
        self.generator: Any = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Build the eight-GPU generator and warm every configured clip shape.

        Runs once at startup. The caller's ``load()`` returning is what marks
        the pod ready, so everything that can fail — missing kernels, a broken
        native linkage, a cold compile — must fail here, not on the first
        client's clip.
        """
        # Must happen before the generator is built: the engine spawns worker
        # processes, which inherit os.environ, and these select the attention
        # backend and the sparse kernel.
        self._apply_profile_environment()
        self._validate_profile_dependencies()

        runtime = self._config.runtime
        num_gpus = int(runtime.get("num_gpus", 8))
        logger.info(
            "building fasth3 generator",
            model_path=str(self._model_path),
            num_gpus=num_gpus,
            clip_frames=self._config.clip_frames,
        )

        from fastvideo import VideoGenerator

        self.generator = VideoGenerator.from_config(self._generator_config())
        self._load_tokenizer()

        self._worker = threading.Thread(
            target=self._worker_loop, name="fasth3-generation", daemon=True
        )
        self._worker.start()
        self._preload_native_imports()
        self._run_blocking(self._warmup)
        logger.info("fasth3 backend loaded")

    def _load_tokenizer(self) -> None:
        """Load the bundle's tokenizer and calibrate the one-token pad filler.

        Padding must land on an exact token count, so the filler is verified to
        cost exactly one token at load rather than assumed.
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path / "tokenizer"))
        for candidate in (" .", ".", " a"):
            base = len(self._tokenizer.encode(WARMUP_PROMPT, add_special_tokens=False))
            padded = len(
                self._tokenizer.encode(WARMUP_PROMPT + candidate, add_special_tokens=False)
            )
            if padded == base + 1:
                self._pad_filler = candidate
                return
        raise RuntimeError("no single-token pad filler found for this tokenizer")

    def _pad_prompt(self, prompt: str) -> str:
        """Return *prompt* at exactly ``PROMPT_TOKENS`` tokens.

        Shorter prompts gain trailing filler tokens; a longer one (past the
        800-character cap only in pathological tokenizations) is truncated at
        the token boundary. The client-facing prompt — what `ClipInfo` echoes —
        is the original; only the engine sees this form.
        """
        encode = lambda text: len(self._tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if ids and len(ids) >= PROMPT_TOKENS:
            return self._tokenizer.decode(ids[:PROMPT_TOKENS])
        padded = prompt + self._pad_filler * (PROMPT_TOKENS - len(ids))
        # Filler cost is calibrated, but a prompt's own tail can merge with the
        # first filler token; correct by measurement rather than assumption.
        while encode(padded) > PROMPT_TOKENS:
            padded = padded[: -len(self._pad_filler)]
        while encode(padded) < PROMPT_TOKENS:
            padded += self._pad_filler
        if encode(padded) != PROMPT_TOKENS:
            logger.warning(f"prompt padded to {encode(padded)} tokens, not {PROMPT_TOKENS}")
        return padded

    @staticmethod
    def _preload_native_imports() -> None:
        """Touch every deferred native import the build path needs.

        Lazy imports mean the first import would otherwise happen on the first
        real clip — after load, after warm-up, after the pod reports ready —
        where a linkage failure is a dead session rather than a startup error.
        The resample below is a real call, so it fails here or not at all.
        """
        import numpy  # noqa: F401
        import torch
        import torchaudio.functional as AF

        AF.resample(torch.zeros(2, NATIVE_SAMPLE_RATE // 10), NATIVE_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

    # --------------------------------------------------------------- profile

    def _apply_profile_environment(self) -> None:
        """Set the FastH3 profile environment, exactly as the reference CLI does.

        Mirrors ``examples/inference/basic/basic_fasth3.py:profile_environment``.
        Values are explicit even for disabled features, so a shell's inherited
        experiment settings cannot silently change the profile a pod serves.
        """
        cfg = self._config.inference
        vsa_kernel = str(cfg.get("vsa_kernel", "sm100a"))
        fusions = "all" if bool(cfg.get("h3_fusions", True)) else "0"
        environment: dict[str, str | None] = {
            "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
            "FASTVIDEO_VSA_SM100A": "1" if vsa_kernel == "sm100a" else "0",
            "FASTVIDEO_VSA_CUTEDSL": "0",
            # A non-empty path enables the diagnostic probe; it must stay unset.
            "FASTVIDEO_H3_VSA_PROBE": None,
            "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
            "FASTVIDEO_FA4": "1" if bool(cfg.get("fa4", True)) else "0",
            "FASTVIDEO_NVFP4_FA4": "0",
            "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": fusions,
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": (
                "1" if bool(cfg.get("inference_torch_compile", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_DECODE": (
                "1" if bool(cfg.get("vae_parallel_decode", True)) else "0"
            ),
            "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
            "FASTVIDEO_ULYSSES_A2A": str(cfg.get("ulysses_a2a", "off")),
            "FASTVIDEO_STAGE_LOGGING": "1",
        }
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        logger.info("fasth3 profile", **{k: (v or "<unset>") for k, v in environment.items()})

    def _validate_profile_dependencies(self) -> None:
        """Fail before the 148 GB load when the selected fast route is absent."""
        import importlib.util

        cfg = self._config.inference
        if bool(cfg.get("fa4", True)):
            try:
                present = importlib.util.find_spec("flash_attn.cute") is not None
            except (ImportError, ModuleNotFoundError):
                present = False
            if not present:
                raise RuntimeError(
                    "FastH3's FA4 route needs the pinned flash-attn-4 package. Install it, "
                    "or set inference.fa4: false in fasth3.yaml."
                )
        if str(cfg.get("vsa_kernel", "sm100a")) == "sm100a":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
            except ImportError:
                present = False
            else:
                present = bool(getattr(block_sparse_attn_sm100a, "_HAS_VSA_SM100A", False))
            if not present:
                raise RuntimeError(
                    "FastH3's sm100a route needs fastvideo-kernel built with the Blackwell VSA "
                    "extension. Install a matching wheel, or set inference.vsa_kernel: triton."
                )

    def _generator_config(self):
        """The engine shape, mirroring ``basic_fasth3.py:build_generator_config``."""
        from fastvideo.api import (
            CompileConfig,
            ComponentConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            ParallelismConfig,
            PipelineSelection,
        )

        cfg = self._config.inference
        runtime = self._config.runtime
        num_gpus = int(runtime.get("num_gpus", 8))
        # The checkpoint's own contract (fastvideo_inference.json) shards the
        # transformer with FSDP. Sharding is what frees the VRAM to keep the
        # text encoder resident, which is the deployment this model wants.
        replicated_dit = bool(runtime.get("replicated_dit", False))
        return GeneratorConfig(
            model_path=str(self._model_path),
            pipeline=PipelineSelection(
                components=ComponentConfig(),
                experimental={
                    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                    "VSA_sparsity": float(cfg.get("vsa_sparsity", 0.9)),
                    "VSA_tile_size": int(cfg.get("vsa_tile_size", 64)),
                    "inference_torch_compile": bool(cfg.get("inference_torch_compile", True)),
                    "vae_parallel_decode": bool(cfg.get("vae_parallel_decode", True)),
                    "vae_parallel_decode_strategy": "gather",
                },
            ),
            engine=EngineConfig(
                num_gpus=num_gpus,
                use_fsdp_inference=num_gpus > 1 and not replicated_dit,
                parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=bool(runtime.get("offload_text_encoder", False)),
                    vae=bool(runtime.get("offload_vae", False)),
                    pin_cpu_memory=bool(runtime.get("pin_cpu_memory", False)),
                ),
                compile=CompileConfig(
                    enabled=False,
                    mode=None,
                    vae_enabled=bool(cfg.get("compile_vae", True)),
                ),
            ),
        )

    # ---------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        """Run submitted jobs, one at a time, forever.

        The waiter is always released, even when the job died: a completion
        event that never arrives is indistinguishable from a hang, and this
        thread is the only one that will ever set it.
        """
        logger.info("generation worker ready")
        while True:
            job = self._jobs.get()
            try:
                if not job.cancelled:
                    job.result = job.fn()
            except BaseException as error:  # noqa: BLE001 — handed to the submitter
                job.error = error
                logger.exception("generation worker job raised")
            finally:
                job.done.set()

    def submit(self, *, frames: int, prompt: str, seed: int, height: int, width: int) -> ClipJob:
        """Queue one clip build and hand back its job handle.

        Returns immediately; the caller polls ``job.done`` and reads
        ``job.result`` — ``(frames_list, samples)``, RGB uint8 frames and an
        int16 ``[1, samples]`` waveform at the wire rate — or ``job.error``.
        """
        job = ClipJob(
            lambda: self._generate_clip(
                frames=frames, prompt=prompt, seed=seed, height=height, width=width
            )
        )
        self._jobs.put(job)
        return job

    def _run_blocking(self, fn) -> None:
        """Run work on the worker, block until it finishes, and re-raise its failure.

        Used only by ``load()``. Blocking here is the point: the runtime marks
        the pod ready when the model's ``load()`` returns, so a failed warm-up
        has to stop startup rather than being discovered by the first client.
        """
        job = ClipJob(fn)
        self._jobs.put(job)
        while not job.done.wait(timeout=_WORKER_POLL_SECONDS):
            pass
        if job.error is not None:
            raise job.error

    # --------------------------------------------------------------- warm-up

    def _warmup(self) -> None:
        """Build one throwaway clip per shape, before the pod reports ready.

        Every distinct frame count and canvas is a separate one-time cost —
        regional compile, sparse-kernel autotune, allocator growth — and paying
        it here means the first real clip builds at warm speed. Results are
        discarded: ``return_frames=False, save_video=False`` skips the whole
        post-decode path, so a warm-up costs generation time and nothing else.
        """
        aspects = self._config.warmup_aspects
        cold = [a for a in clip_plan.ASPECT_CHOICES if a not in aspects]
        if cold:
            logger.info(
                "aspects left cold; their first clip pays a one-off compile stall", aspects=cold
            )
        for aspect in aspects:
            height, width = clip_plan.canvas_for_choice(aspect)
            started = time.monotonic()
            self.generator.generate(
                self._request(
                    frames=self._config.clip_frames,
                    prompt=WARMUP_PROMPT,
                    seed=self._config.seed,
                    height=height,
                    width=width,
                    keep_output=False,
                )
            )
            logger.info(
                "warmed clip shape",
                aspect=aspect,
                frames=self._config.clip_frames,
                height=height,
                width=width,
                seconds=round(time.monotonic() - started, 2),
            )

    # ------------------------------------------------------------ generation

    def _request(
        self,
        *,
        frames: int,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        keep_output: bool,
    ):
        """Build one generation request.

        Mirrors ``basic_fasth3.py:build_request``. ``keep_output=False`` is the
        warm-up shape: it skips the whole post-decode path, so a warm-up costs
        generation time and nothing else.
        """
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        return GenerationRequest(
            # Padded to the fixed token length so one compiled shape serves
            # every prompt; ClipInfo keeps echoing the original text.
            prompt=self._pad_prompt(prompt),
            # MiniMax-H3 is guidance-distilled, so there is no negative branch
            # to steer and no CFG pass to pay for.
            negative_prompt="",
            sampling=SamplingConfig(
                height=height,
                width=width,
                num_frames=frames,
                fps=FRAME_RATE,
                num_inference_steps=self._config.num_inference_steps,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=seed,
            ),
            output=OutputConfig(save_video=False, return_frames=keep_output),
        )

    def _generate_clip(self, *, frames: int, prompt: str, seed: int, height: int, width: int):
        """Build one clip and convert it to what the output tracks want.

        Returns ``(frames_list, samples)``: a list of RGB uint8 ``[h, w, 3]``
        arrays and int16 ``[1, samples]`` at 48 kHz, trimmed to exactly
        ``len(frames_list) / 24`` seconds so the two tracks stay in lockstep.
        """
        started = time.monotonic()
        result = self.generator.generate(
            self._request(
                frames=frames,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                keep_output=True,
            )
        )
        built = time.monotonic() - started

        frames_list = result.frames
        if not frames_list:
            raise RuntimeError("the generator returned no frames")
        samples = self._to_wire_audio(result.audio, result.audio_sample_rate, len(frames_list))
        # The line to evaluate the deployment by: build seconds against content
        # seconds (realtime_x > 1 means the clip built faster than it plays) on
        # the GPU count that produced it, with the per-stage split. The numbers
        # live in the message itself so every log formatter carries them.
        content = len(frames_list) / FRAME_RATE
        gpus = int(self._config.runtime.get("num_gpus", 8))
        logger.info(
            f"clip built: {len(frames_list)}f ({content:.2f}s content) in {built:.2f}s "
            f"= {content / built:.2f}x realtime on {gpus} gpus, "
            f"stages={self._stage_times(result)}"
        )
        return frames_list, samples

    @staticmethod
    def _stage_times(result) -> dict:
        """Per-stage seconds from the generator, for the clip log line.

        This is where a regression shows up first: post-decode frame processing
        scales with resolution x frames and competes with the build budget.
        """
        try:
            stages = getattr(getattr(result, "logging_info", None), "stages", None)
            if not stages:
                return {}
            return {
                name: round(float(metrics["execution_time"]), 3)
                for name, metrics in stages.items()
                if metrics.get("execution_time") is not None
            }
        except Exception:  # noqa: BLE001 — a log line must never fail a clip
            logger.exception("could not read the generator stage timings")
            return {}

    def _to_wire_audio(self, audio, sample_rate, frames: int):
        """Resample, downmix and quantize one clip's waveform for the wire.

        Mono at the source is deliberate: the transport mean-downmixes before
        the wire anyway, and the runtime recorder flattens two channels by
        concatenation, so a stereo emit only corrupts recordings. Averaging here,
        in float and before the int16 scale, is the same downmix one step
        earlier.
        """
        import torch
        import torchaudio.functional as AF

        if audio is None:
            raise RuntimeError("the generator returned no audio")
        waveform = audio if torch.is_tensor(audio) else torch.as_tensor(audio)
        waveform = waveform.detach().float().cpu()
        # The decoder hands back [samples, channels]; the wire wants channel-major.
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.contiguous()

        rate = int(sample_rate or NATIVE_SAMPLE_RATE)
        if rate != OUTPUT_SAMPLE_RATE:
            waveform = AF.resample(waveform, rate, OUTPUT_SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        want = round(frames / FRAME_RATE * OUTPUT_SAMPLE_RATE)
        if waveform.shape[-1] > want:
            waveform = waveform[:, :want]
        elif waveform.shape[-1] < want:
            pad = torch.zeros(
                (waveform.shape[0], want - waveform.shape[-1]), dtype=waveform.dtype
            )
            waveform = torch.cat([waveform, pad], dim=-1)
        return (waveform.clamp(-1, 1) * 32767).to(torch.int16).numpy()


__all__ = ["OUTPUT_SAMPLE_RATE", "ClipJob", "FastH3Backend"]
