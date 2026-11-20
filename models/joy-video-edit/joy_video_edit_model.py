# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""JoyAI-Video-Edit, the model half.

A plain class the application in ``joy_video_edit.py`` drives one step at a
time through ``load()``, ``generate()`` and ``reset()``. It imports nothing
from ``reactor_runtime`` and knows nothing about clients, tracks, or commands.
It owns the JoyOmni runtime (16B MMDiT, causal VAE, MiMo-VL-7B text encoder),
the live streaming session, and the one compute thread every model call runs
on.

The two halves meet on :class:`JoyVideoEditInput` and
:class:`JoyVideoEditResult`. A run is one identity (``run_id``) and one
:class:`RunConditioning` (prompt, reference image, seed), which the step that
opens the run carries and which stays fixed for the run. Inside a run the
model replaces its streaming session with a fresh one every
``kv_reset_frames`` camera frames, which bounds drift. A session's first chunk
takes one camera frame and every later chunk takes eight, so each result says
how many frames the next step must carry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joy_video_edit_assets import read_config, resolve_checkpoints
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Frames per chunk after a session's first: the causal VAE's temporal compression factor.
# load() checks it against the loaded VAE.
CHUNK_FRAMES = 8


# ---------------------------------------------------------------------------
# The inner contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConditioning:
    """What a run is conditioned on. Fixed from the step that opens the run to its end."""

    prompt: str
    # (H, W, 3) uint8 RGB reference image, or None for plain instruction editing.
    reference: np.ndarray | None
    seed: int


@dataclass(frozen=True)
class JoyVideoEditInput:
    """What one step needs."""

    # (H, W, 3) uint8 RGB camera frames, as many as the last result's ``frames_wanted``.
    frames: list[np.ndarray]
    run_id: int
    # Carried on the step that opens run ``run_id``; None on every later step of the run.
    conditioning: RunConditioning | None


@dataclass(frozen=True)
class JoyVideoEditResult:
    """What one step produced."""

    # [T, H, W, 3] uint8 edited frames.
    frames: np.ndarray
    # The run the model holds after this step.
    run_id: int
    # How many camera frames the next step must carry.
    frames_wanted: int


class NoConditioning(Exception):
    """The step opens a new run and carries no conditioning to start it from."""


class WrongFrameCount(Exception):
    """The step carries a different number of camera frames than the last result asked for."""


class ChunkFailed(Exception):
    """The chunk raised part-way. The run's session is unusable; ``reset()`` before stepping again."""


class RunBroken(Exception):
    """The step continues a run whose last chunk failed, without a ``reset()`` in between."""


@dataclass
class _Run:
    conditioning: RunConditioning
    session: Any
    frames_since_session: int = 0
    failed: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _close(session) -> None:
    """Close a streaming session; a failure is logged, not raised."""
    try:
        session.close()
    except Exception:
        logger.exception("session.close() failed (non-fatal)")


# ---------------------------------------------------------------------------
# The model half
# ---------------------------------------------------------------------------


class JoyVideoEditModel:
    """JoyAI-Video-Edit: live camera frames and an edit instruction in, edited frames out."""

    def __init__(self) -> None:
        self._run: _Run | None = None
        self._run_id: int | None = None

    # ---------------------------------------------------------------------------
    # load()
    # ---------------------------------------------------------------------------

    def load(self, config_path: Path | None, weights_root: Path | str) -> None:
        """Load the weights, warm every shape a session can meet, and capture its CUDA graphs.

        Args:
            config_path: ``joy_video_edit.yaml``; None runs on the defaults.
            weights_root: The weights directory. The pinned checkpoints are downloaded
                into it on first load, and the compile caches live under it.
        """
        config = read_config(config_path)

        # ---- Device / multi-GPU pipeline parallelism ----
        # GPU 0 always hosts the DiT (main bottleneck).  Additional GPUs host
        # VAE stages separately (intra-process device split, no NCCL needed).
        self.device = torch.device(f"cuda:{int(config.get('device_id', 0))}")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Per-stage devices.  Unset means the DiT device; every stage still gets
        # its own VAE object (weights are shared when devices coincide), because
        # the VAE keeps per-call streaming state on the instance.
        def _dev(key: str) -> str:
            """Return a device string from config, or the DiT device if absent/null.

            A stage pinned to a GPU index beyond the visible count falls back to the
            DiT device at load(), instead of failing deep in a warm-up forward pass.
            """
            v = config.get(key)
            if not v:
                return str(self.device)
            d = torch.device(v)
            if d.type == "cuda" and d.index is not None and d.index >= visible:
                logger.warning(
                    "load(): %s=%r requested but only %d GPU(s) visible — "
                    "falling back to DiT device %s",
                    key, v, visible, self.device,
                )
                return str(self.device)
            return str(v)

        self._vae_encode_device = _dev("vae_encode_device")
        self._vae_decode_device = _dev("vae_decode_device")
        self._vae_pseudo_device = _dev("vae_pseudo_device")

        logger.info(
            "load(): visible GPUs=%d dit=%s vae_encode=%s vae_decode=%s vae_pseudo=%s",
            visible, self.device,
            self._vae_encode_device, self._vae_decode_device, self._vae_pseudo_device,
        )

        # ---- Generation geometry ----
        self.height = int(config.get("height", 720))
        self.width = int(config.get("width", 1248))
        self.warmup_height = int(config.get("warmup_height", self.height))
        self.warmup_width = int(config.get("warmup_width", self.width))
        # A run's session is replaced by a fresh one (new KV cache, prompt
        # re-encoded on the next frame) after this many camera frames; 0 = never.
        self._kv_reset_frames = max(0, int(config.get("kv_reset_frames", 600)))

        # Temporal RoPE positions of the attention window (see joy_video_edit.yaml).
        _mti = config.get("max_temporal_ids", None)
        self._max_temporal_ids: int | None = int(_mti) if _mti is not None else None

        # Freeze KV when input is static (saves one DiT forward per static chunk).
        self._freeze_kv_on_static: bool = bool(config.get("freeze_kv_on_static", True))
        self._static_diff_thresh: float = float(config.get("static_diff_thresh", 0.5))

        # Denoising steps per chunk: fixed by the DiT's step distillation (upstream: 2).
        self._num_inference_steps = int(config.get("num_inference_steps", 2))

        # Per-chunk stage timing breakdown (set true to profile).
        self._profile_timings: bool = bool(config.get("profile_timings", False))

        # ---- Weights ----
        checkpoints = resolve_checkpoints(config, weights_root)
        logger.info(
            "load(): dit=%s vae=%s text_encoder=%s",
            checkpoints.dit, checkpoints.vae, checkpoints.text_encoder,
        )

        # ---- JoyAI performance env vars (set before any xvideo import) ----
        # xvideo reads these at import time, so they are force-set from the config
        # BEFORE importing it; the config, not the image environment, decides.
        #
        # fp8_img: FP8 quantisation on DiT image-stream GEMMs (joyomni_ops).
        #   Requires joyomni_ops compiled with fp8_scaled_mm (CUTLASS).
        #   Default on; set fp8_img: false in joy_video_edit.yaml to disable.
        fp8_img_cfg = config.get("fp8_img", True)
        fp8_txt_cfg = config.get("fp8_txt", False)

        # Guard: verify joyomni_ops was compiled WITH the FP8 CUDA kernel before
        # enabling FP8 quantisation.  The Python wrapper (fp8_scaled_mm) always
        # exists; has_fp8() checks whether the C++ kernel is actually registered
        # in torch.ops.  If not, running with JOYOMNI_FP8_IMG=1 crashes at the
        # first DiT forward (_maybe_install_fp8_stream assertion).
        _fp8_kernel_ok = False
        try:
            import joyomni_ops as _joy_ops
            _fp8_kernel_ok = _joy_ops.has_fp8()
        except Exception:
            pass
        if fp8_img_cfg and not _fp8_kernel_ok:
            logger.warning(
                "load(): fp8_img=true in config but joyomni_ops.has_fp8()=False "
                "(image built without CUTLASS FP8 kernel — rebuild with vendor/cutlass). "
                "Disabling FP8 to avoid crash."
            )
            fp8_img_cfg = False
        if fp8_txt_cfg and not _fp8_kernel_ok:
            fp8_txt_cfg = False

        os.environ["JOYOMNI_FP8_IMG"] = "1" if fp8_img_cfg else "0"
        os.environ["JOYOMNI_FP8_TXT"] = "1" if fp8_txt_cfg else "0"

        # TorchInductor / Triton / CUDA graph cache.
        # `reactor run` mounts the weights directory, so the compile caches live
        # there and survive container restarts.
        # IMPORTANT: use os.environ[key] = (force-set), NOT setdefault, because
        # the Dockerfile sets ENV TORCHINDUCTOR_CACHE_DIR=/root/.cache/... which
        # would otherwise win over setdefault even though the weights dir IS mounted.
        # Must be set BEFORE JoyOmniRuntime.load() — warmup triggers compilation.
        _compile_cache = os.path.join(str(weights_root), ".compile_cache", "joy-video-edit")
        try:
            os.makedirs(_compile_cache, exist_ok=True)
        except OSError:
            _compile_cache = "/root/.cache"
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(_compile_cache, "torchinductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(_compile_cache, "triton")
        os.environ["CUDA_CACHE_PATH"] = os.path.join(_compile_cache, "nv_compute")
        logger.info(
            "load(): compile cache → %s (triton=%s)",
            _compile_cache,
            os.environ["TRITON_CACHE_DIR"],
        )

        # ---- JoyOmniRuntime (loads DiT + VAE + text encoder) ----
        # Import lazily: joyomni_streaming requires the vendored xvideo package
        # which is copied into /app at image-build time.
        from xvideo.serving.joyomni_streaming import (
            JoyOmniRuntime, StreamingSettings, compute_executor, ref_image_buckets,
        )
        self._StreamingSettings = StreamingSettings

        # A small fixed set of shapes for every session, so the load-time warm-up covers all of
        # them: the DiT text is padded to `text_tokens` (padding masked in attention), and every
        # reference image is fitted into the nearest of `ref_image_aspects` (width:height), each sized as
        # upstream's 768 reference bucketing sizes an image of that aspect.
        _tt = config.get("text_tokens", 640)
        self._text_tokens: int | None = int(_tt) if _tt else None
        _ra = config.get("ref_image_aspects", ["1:1", "4:3", "3:4"])
        self._ref_image_sizes = (
            ref_image_buckets(tuple((int(a.split(":")[1]), int(a.split(":")[0])) for a in _ra)) if _ra else None
        )
        logger.info("load(): text_tokens=%s ref_image_sizes (h, w)=%s", self._text_tokens, self._ref_image_sizes)

        # The one compute thread: load and warm-up run on it (JoyOmniRuntime.load dispatches
        # there itself), and generate() and reset() run every step's model work on it.
        self._compute = compute_executor()

        logger.info("load(): building JoyOmniRuntime ...")
        _runtime_kwargs: dict = dict(
            device=str(self.device),
            vae_encode_device=self._vae_encode_device,
            vae_decode_device=self._vae_decode_device,
            vae_pseudo_device=self._vae_pseudo_device,
            postprocess_device=self._vae_pseudo_device,
            warmup_height=self.warmup_height,
            warmup_width=self.warmup_width,
            session_settings=self._session_settings(seed=42),
        )

        self._runtime = JoyOmniRuntime.load(
            dit_ckpt=str(checkpoints.dit),
            vae_ckpt=str(checkpoints.vae),
            text_encoder_ckpt=str(checkpoints.text_encoder),
            **_runtime_kwargs,
        )
        _ft = int(self._runtime.pipeline.vae_scale_factor_temporal)
        if _ft != CHUNK_FRAMES:
            raise RuntimeError(f"VAE temporal factor is {_ft}, CHUNK_FRAMES assumes {CHUNK_FRAMES}")
        logger.info(
            "load(): JoyOmniRuntime ready. max_temporal_ids=%s freeze_kv=%s profile=%s fp8_img=%s "
            "| pipeline: dit=%s vae_enc=%s vae_dec=%s vae_pseudo=%s",
            self._max_temporal_ids,
            self._freeze_kv_on_static,
            self._profile_timings,
            os.environ.get("JOYOMNI_FP8_IMG", "0"),
            self.device,
            self._vae_encode_device or self.device,
            self._vae_decode_device or self.device,
            self._vae_pseudo_device or self.device,
        )

        # ---- torch.compile DiT double_blocks ----------------------------------------
        # Applies TorchInductor kernel fusion to the 40 MMDoubleStreamBlock forward
        # passes, reducing Python dispatch overhead (~52% of chunk latency).
        #
        # Why per-block (not whole transformer):
        #   The outer transformer.forward() manages Python-level KV-cache state via
        #   cache_context(), reset_inference_kv_cache(), etc.  Compiling at block
        #   granularity leaves that state management in eager Python and only fuses
        #   the heavy inner compute (norms + QKV GEMMs + SDPA + MLP).
        #
        # Why dynamic=True:
        #   The KV cache grows from 3 600 tokens (chunk 1) to 57 600 tokens (chunk 16)
        #   as temporal context accumulates.  dynamic=True uses symbolic shapes so
        #   Inductor emits a single kernel that handles all KV lengths without
        #   recompiling per chunk.
        #
        # Compiled Triton kernels are persisted by TORCHINDUCTOR_FX_GRAPH_CACHE → the
        # first container startup pays the compile cost; subsequent starts reuse cache.
        if config.get("torch_compile_dit", False):
            try:
                _transformer = self._runtime.pipeline.transformer
                _blocks = _transformer.double_blocks
                for _i, _blk in enumerate(_blocks):
                    _blocks[_i] = torch.compile(
                        _blk,
                        mode="default",
                        fullgraph=False,
                        dynamic=True,
                    )
                logger.info(
                    "load(): torch.compile applied to %d DiT double_blocks "
                    "(mode=default, dynamic=True, fullgraph=False)",
                    len(_blocks),
                )
            except Exception as _e:
                logger.warning(
                    "load(): torch.compile skipped, running uncompiled: %s", _e
                )

        # CPU thread cap for torch's intra-op pool, matching the image's OMP_NUM_THREADS.
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

        # Disable grad for the loading thread — all forward passes run inside
        # torch.inference_mode() contexts in the JoyAI session workers (the
        # @torch.no_grad() decorators on push_frame, _encode_reference_chunk,
        # _denoise_chunk, _decode_chunk_pixels, etc.).
        torch.inference_mode(True).__enter__()  # noqa: FLY002  (thread-wide)

    # ---------------------------------------------------------------------------
    # The step
    # ---------------------------------------------------------------------------

    def generate(self, input: JoyVideoEditInput) -> JoyVideoEditResult:
        """Edit one chunk of camera frames, opening run ``input.run_id`` first if it is new.

        Raises:
            NoConditioning: The step opens a new run and carries no conditioning.
            RunBroken: The run's last chunk failed and no ``reset()`` followed.
            WrongFrameCount: The step carries a different number of frames than asked for.
            ChunkFailed: The chunk raised part-way; the cause is chained.
        """
        return self._compute.submit(self._generate, input).result()

    def reset(self) -> None:
        """End the live run, if any, and release its session. The next step opens a new run."""
        self._compute.submit(self._end_run).result()

    def _generate(self, input: JoyVideoEditInput) -> JoyVideoEditResult:
        if input.run_id != self._run_id:
            if input.conditioning is None:
                raise NoConditioning(f"run {input.run_id} is new and the step carries no conditioning")
            self._end_run()
            self._run = _Run(conditioning=input.conditioning, session=self._new_session(input.conditioning))
            self._run_id = input.run_id
            logger.info(
                "run %d started: prompt=%.60s rv2v=%s steps=%d seed=%d",
                input.run_id, input.conditioning.prompt, input.conditioning.reference is not None,
                self._num_inference_steps, input.conditioning.seed,
            )
        run = self._run
        assert run is not None
        if run.failed:
            raise RunBroken(f"run {self._run_id} failed on its last chunk; reset() before stepping it again")

        # Periodic session replacement (bounds drift; upstream default 600 frames). Taken
        # between chunks, so nothing of the old session is lost.
        if self._session_is_due(run):
            logger.info("run %d: session replaced after %d frames", self._run_id, run.frames_since_session)
            _close(run.session)
            run.session = self._new_session(run.conditioning)
            run.frames_since_session = 0

        n = run.session.frames_per_next_chunk
        if len(input.frames) != n:
            raise WrongFrameCount(f"the next chunk takes {n} camera frames, the step carries {len(input.frames)}")
        try:
            chunk = run.session.push_chunk(list(input.frames))
        except Exception as exc:
            run.failed = True
            logger.exception("run %d: chunk failed part-way", self._run_id)
            raise ChunkFailed(f"run {self._run_id}: chunk failed part-way") from exc
        run.frames_since_session += n

        # One [T, H, W, 3] array: the session's own output array when it returns one
        # (no copy), otherwise the frames stacked.
        frames = chunk.frames if chunk.frames is not None else np.stack(chunk.jpegs, axis=0)
        if chunk.valid_count is not None:
            frames = frames[: chunk.valid_count]
        frames_wanted = 1 if self._session_is_due(run) else run.session.frames_per_next_chunk
        return JoyVideoEditResult(frames=frames, run_id=input.run_id, frames_wanted=frames_wanted)

    def _session_is_due(self, run: _Run) -> bool:
        return bool(self._kv_reset_frames) and run.frames_since_session >= self._kv_reset_frames

    def _new_session(self, conditioning: RunConditioning):
        reference = PILImage.fromarray(conditioning.reference) if conditioning.reference is not None else None
        return self._runtime.create_v2v_session(
            prompt=conditioning.prompt,
            settings=self._session_settings(seed=conditioning.seed),
            ref_image=reference,
        )

    def _end_run(self) -> None:
        run, self._run, self._run_id = self._run, None, None
        if run is not None:
            _close(run.session)

    def _session_settings(self, *, seed: int):
        """The StreamingSettings of a live session; the load-time warm-up runs with the same."""
        return self._StreamingSettings(
            height=self.height,
            width=self.width,
            num_inference_steps=self._num_inference_steps,
            seed=seed,
            max_temporal_ids=self._max_temporal_ids,
            freeze_kv_on_static=self._freeze_kv_on_static,
            static_diff_thresh=self._static_diff_thresh,
            profile_timings=self._profile_timings,
            # Raw uint8 frames: the Reactor transport encodes the video track itself.
            output_codec="h264",
            text_tokens=self._text_tokens,
            ref_image_sizes=self._ref_image_sizes,
        )
