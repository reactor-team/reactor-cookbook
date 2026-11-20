from __future__ import annotations

import concurrent.futures
import functools
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image

from xvideo.config import ExpConfig, generate_video_image_bucket
from xvideo.models.models import load_dit, load_pipeline, build_vae
from xvideo.models.pipeline import (
    PRECISION_TO_TYPE,
)
from xvideo.serving.graph_runner import GRAPH_WINDOW_CHUNKS, StreamingGraphRunner, graph_env_enabled
from xvideo.utils import _dynamic_resize_from_bucket, seed_everything

DEFAULT_REFERENCE_IMG_IV2V_BASESIZE = 768

try:
    from torchvision.io import encode_jpeg as _tv_encode_jpeg
    _NVJPEG_OK = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _tv_encode_jpeg = None
    _NVJPEG_OK = False

def _autocast_ctx(device_type: str, dtype: torch.dtype, enabled: bool):
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    return nullcontext()


_FULL_WARMUP_CHUNKS = 4
_GRAPH_CACHE_CAP = 1
_GRAPH_CAPTURE_MAX_FAILS = 2

# All model work -- load, warm-up, CUDA-graph capture and every session -- runs on one
# long-lived thread. cuDNN keeps its execution-plan cache per thread, so a shape first
# planned on one thread is planned again (slowly) on another; one thread also means one
# CUDA current device and one default stream for everything.
_compute_executor: concurrent.futures.ThreadPoolExecutor | None = None
_compute_thread_id: int | None = None
_compute_executor_lock = threading.Lock()


def _mark_compute_thread() -> None:
    global _compute_thread_id
    _compute_thread_id = threading.get_ident()


def compute_executor() -> concurrent.futures.ThreadPoolExecutor:
    """The single-thread executor every model call runs on (created on first use)."""
    global _compute_executor
    with _compute_executor_lock:
        if _compute_executor is None:
            _compute_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="joyomni-compute",
                initializer=_mark_compute_thread,
            )
        return _compute_executor


# Frames that are not already at the session size are resized with PIL on this pool, a chunk's
# frames in parallel: PIL releases the GIL while it resamples, and resizing a chunk's 1280x720
# camera frames one after another costs ~65 ms per 8-frame chunk on the compute thread.  The
# result is the same PIL resize, frame for frame.
_RESIZE_WORKERS = 8
_resize_executor: concurrent.futures.ThreadPoolExecutor | None = None


def resize_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _resize_executor
    with _compute_executor_lock:
        if _resize_executor is None:
            _resize_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_RESIZE_WORKERS, thread_name_prefix="joyomni-resize",
            )
        return _resize_executor


def on_compute_thread(fn):
    """Run `fn` on the compute thread, blocking the caller until it returns."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if threading.get_ident() == _compute_thread_id:
            return fn(*args, **kwargs)
        return compute_executor().submit(fn, *args, **kwargs).result()
    return wrapper

def _vae_compile_module():
    from xvideo.models.vae import vae_compile as module
    return module

@dataclass(frozen=True)
class _ProfileTimer:
    wall_start: float
    start_event: torch.cuda.Event | None = None

def _profile_timer_start(
    device: torch.device | str | None = None,
    *,
    use_cuda_event: bool = True,
) -> _ProfileTimer:
    if not use_cuda_event or device is None or not torch.cuda.is_available():
        return _ProfileTimer(wall_start=time.perf_counter())
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return _ProfileTimer(wall_start=time.perf_counter())
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(device_obj))
    return _ProfileTimer(wall_start=time.perf_counter(), start_event=start_event)

def _profile_timer_elapsed(started: _ProfileTimer | float) -> float:
    if isinstance(started, _ProfileTimer):
        if started.start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            return float(started.start_event.elapsed_time(end_event)) / 1000.0
        return time.perf_counter() - started.wall_start
    return time.perf_counter() - float(started)

@dataclass(frozen=True)
class StreamingSettings:
    height: int = 720
    width: int = 1248
    num_inference_steps: int = 2
    seed: int = 42
    max_sequence_length: int | None = None
    enable_denormalization: bool | None = None
    max_temporal_ids: int | None = None
    freeze_kv_on_static: bool = True
    static_diff_thresh: float = 0.5
    store_clean_self_only: bool = True
    profile_timings: bool = False
    output_codec: str = "mjpeg"
    # Pad the DiT's text tokens to this many and mask the padding in attention, so every prompt
    # up to this length runs at one set of attention shapes.  None = the prompt's own length.
    text_tokens: int | None = None
    # Fit every reference image into the nearest-aspect of these (height, width) sizes
    # (letterboxing the remainder) instead of its own aspect bucket, so references run at a
    # small fixed set of shapes.  None = upstream's per-aspect bucket.
    ref_image_sizes: tuple[tuple[int, int], ...] | None = None

@dataclass
class StreamingChunkResult:
    profile: dict[str, Any]
    source_metas: list[dict[str, Any]]
    elapsed: float
    jpegs: list[bytes] | None = None
    valid_count: int | None = None
    # Raw uint8 output as one [T, H, W, 3] array (the h264 output path); `jpegs` then lists
    # views of its frames.
    frames: np.ndarray | None = None

def _to_host(t: torch.Tensor) -> torch.Tensor:
    """Queue a device -> host copy into pinned memory (a direct DMA instead of a staged pageable
    copy) and return without waiting: the data is valid once the current stream reaches this point.

    A fresh pinned tensor per call: its block returns to the caching host allocator when the
    last array viewing it is released (and the copy has completed), so returned frames never
    alias a later chunk's.
    """
    if not t.is_cuda:
        return t
    out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    out.copy_(t, non_blocking=True)
    return out


def _as_pil(frame: Image.Image | np.ndarray) -> Image.Image:
    return Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame


def _module_device(module: torch.nn.Module) -> torch.device:
    if hasattr(module, "device"):
        return torch.device(module.device)
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _load_vae_for_device(cfg: ExpConfig, device: torch.device) -> torch.nn.Module:
    vae = build_vae(cfg, device)
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _clone_vae_shared_weights(src: torch.nn.Module) -> torch.nn.Module:
    """A second VAE handle sharing `src`'s weight tensors.

    The VAE keeps per-call streaming state on the instance, so each role needs
    its own module object -- but weights can be shared. Conv weights are
    converted to channels_last_3d up-front: vae_compile converts per instance,
    and Tensor.to(memory_format=...) returns self once the layout already
    matches, so every clone keeps sharing storage.
    """
    from xvideo.models.vae import XVAEChunkCausal

    for m in src.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
    with torch.device("meta"):
        clone = XVAEChunkCausal.from_config(src.config)
    clone.load_state_dict(src.state_dict(), assign=True)
    clone.requires_grad_(False)
    return clone.eval()

def ref_image_buckets(
    aspects_hw: tuple[tuple[int, int], ...] = ((1, 1), (4, 3), (3, 4)),
    basesize: int = DEFAULT_REFERENCE_IMG_IV2V_BASESIZE,
) -> tuple[tuple[int, int], ...]:
    """For each (h, w) aspect, the (height, width) that upstream's reference bucketing gives an
    image of exactly that aspect (area about basesize^2, sides aligned to the bucket step)."""
    buckets = generate_video_image_bucket(img_basesize=basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0)
    sizes = []
    for ah, aw in aspects_hw:
        probe = Image.new("RGB", (aw * 240, ah * 240))
        _, bucket = _dynamic_resize_from_bucket(
            probe, bucket_configs=buckets, num_frames=1, num_items=1, return_bucket=True,
        )
        sizes.append((int(bucket[-2]), int(bucket[-1])))
    return tuple(sizes)


def _nearest_aspect(image: Image.Image, sizes: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r = np.log(image.height / image.width)
    return min(sizes, key=lambda hw: abs(np.log(hw[0] / hw[1]) - r))


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit `image` inside (height, width) keeping its aspect; pad with mid-grey (0 after normalisation)."""
    h, w = size
    scale = min(h / image.height, w / image.width)
    rh, rw = max(1, round(image.height * scale)), max(1, round(image.width * scale))
    resized = image.resize((rw, rh), getattr(Image, "Resampling", Image).BICUBIC)
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    canvas.paste(resized, ((w - rw) // 2, (h - rh) // 2))
    return canvas


def _warmup_frame(index: int, width: int, height: int) -> Image.Image:
    """A deterministic frame whose content moves from one index to the next."""
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    r = (x + 7.0 * index) % 256.0
    g = (y + 5.0 * index) % 256.0
    b = (np.sin(x * 0.02 + index * 0.3) * np.cos(y * 0.015) * 0.5 + 0.5) * 255.0
    return Image.fromarray(np.stack(np.broadcast_arrays(r, g, b), axis=-1).astype(np.uint8), mode="RGB")


class JoyOmniRuntime:
    def __init__(
        self,
        cfg: ExpConfig,
        pipeline: Any,
        device: torch.device,
        *,
        decode_vae: torch.nn.Module | None = None,
        pseudo_encode_vae: torch.nn.Module | None = None,
        postprocess_device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.device = device
        self.decode_vae = decode_vae or pipeline.vae
        self.pseudo_encode_vae = pseudo_encode_vae or self.decode_vae
        self.postprocess_device = postprocess_device or _module_device(self.pseudo_encode_vae)
        self.graph_runners: dict[tuple, StreamingGraphRunner] = {}
        self.graph_capture_failures: dict[tuple, int] = {}
        # How many captured runners stay resident (each holds its own static KV pool).
        self.graph_cache_cap = _GRAPH_CACHE_CAP
        # Set once the load-time warm-up has captured every graph a session can use: from then on
        # a session never captures (a capture inside a live session is a multi-second stall).
        self.graphs_sealed = False
        self._graph_mem_pool = None
        self.output_quality = 60
        self.lossless_output = False

    @classmethod
    @on_compute_thread
    def load(
        cls,
        dit_ckpt: str | None,
        *,
        vae_ckpt: str | None = None,
        text_encoder_ckpt: str | None = None,
        device: str | torch.device | None = None,
        vae_device: str | torch.device | None = None,
        vae_encode_device: str | torch.device | None = None,
        vae_decode_device: str | torch.device | None = None,
        vae_pseudo_device: str | torch.device | None = None,
        postprocess_device: str | torch.device | None = None,
        seed: int = 42,
        warmup_height: int = 720,
        warmup_width: int = 1248,
        session_settings: StreamingSettings | None = None,
    ) -> "JoyOmniRuntime":
        """`session_settings`: the settings every live session uses.  When given, the load-time
        warm-up runs whole sessions with them (warmup_sessions) instead of warmup_full_pipeline,
        and the reference-image encoder is warmed at the fixed reference sizes only."""
        device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device_obj.type == "cuda" and device_obj.index is not None:
            torch.cuda.set_device(device_obj)
        encode_device_arg = vae_encode_device if vae_encode_device is not None else vae_device
        decode_device_arg = vae_decode_device if vae_decode_device is not None else vae_device
        pseudo_device_arg = vae_pseudo_device if vae_pseudo_device is not None else decode_device_arg
        vae_encode_device_obj = torch.device(encode_device_arg) if encode_device_arg is not None else None
        vae_decode_device_obj = torch.device(decode_device_arg) if decode_device_arg is not None else None
        vae_pseudo_device_obj = torch.device(pseudo_device_arg) if pseudo_device_arg is not None else None
        postprocess_device_obj = torch.device(postprocess_device) if postprocess_device is not None else vae_pseudo_device_obj
        seed_everything(seed)

        cfg = ExpConfig()
        if vae_ckpt is not None:
            cfg.vae_arch_config["pretrained"] = vae_ckpt
        if text_encoder_ckpt is not None:
            cfg.text_encoder_arch_config["params"]["text_encoder_ckpt"] = text_encoder_ckpt
        if dit_ckpt is not None:
            cfg.dit_ckpt = dit_ckpt


        dit = load_dit(cfg, device=device_obj)
        if getattr(dit.config, "causal", False):
            dit.config.use_inference_kv_cache = True
        dit.requires_grad_(False)
        dit.eval()

        from xvideo.models.dit import attention_backend
        print(f"#####[STREAM] attention backend: {attention_backend()}", flush=True)

        pipeline = load_pipeline(cfg, dit, device_obj)
        pipeline.vae.requires_grad_(False)
        pipeline.vae.eval()
        if vae_encode_device_obj is not None:
            pipeline.vae = pipeline.vae.to(vae_encode_device_obj)
            print(f"#####[STREAM] moved encode VAE to {vae_encode_device_obj}")

        def _vae_for_role(role: str, target_device: torch.device) -> torch.nn.Module:
            target = _canonical_device(target_device)
            for share_src in (pipeline.vae, decode_vae):
                if share_src is not None and _canonical_device(_module_device(share_src)) == target:
                    print(f"#####[STREAM] {role} VAE shares weights on {target_device}")
                    return _clone_vae_shared_weights(share_src)
            vae = _load_vae_for_device(cfg, target_device)
            print(f"#####[STREAM] loaded {role} VAE on {target_device}")
            return vae

        decode_vae = None
        if vae_decode_device_obj is not None:
            decode_vae = _vae_for_role("decode", vae_decode_device_obj)
        if decode_vae is None:
            decode_vae = pipeline.vae
        pseudo_encode_vae = decode_vae
        if vae_pseudo_device_obj is not None:
            pseudo_encode_vae = _vae_for_role("pseudo encode", vae_pseudo_device_obj)

        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.transformer.eval()

        _orientations = [(warmup_height, warmup_width)]
        if (warmup_width, warmup_height) != (warmup_height, warmup_width):
            _orientations.append((warmup_width, warmup_height))
        _stem_mod = pipeline.vae.stem

        try:
            _vc = _vae_compile_module()
            _lat_c = int(getattr(decode_vae, "latent_channels", 0) or 0)
            _fspa = int(getattr(decode_vae, "ffactor_spatial", 0) or 0)
            _ddev = _module_device(decode_vae)
            _ddt = PRECISION_TO_TYPE[cfg.vae_precision]
            if _lat_c > 0 and _fspa > 0:
                for (_wh, _ww) in _orientations:
                    _ch = _wh * _stem_mod.group // _stem_mod.stride
                    _cw = _ww * _stem_mod.group // _stem_mod.stride
                    _vc.warmup_decode(
                        decode_vae, _lat_c,
                        _ch // _fspa, _cw // _fspa,
                        device=_ddev, dtype=_ddt,
                    )
        except Exception as _vc_exc:
            print(f"#####[STREAM] VAE compile warmup skipped: {_vc_exc!r}")

        try:
            _vc = _vae_compile_module()
            _vae_dt = PRECISION_TO_TYPE[cfg.vae_precision]
            _vae_ac = (_vae_dt != torch.float32)

            _src_vae = pipeline.vae
            for (_wh, _ww) in _orientations:
                _vc.warmup_encode(
                    _src_vae, 3, _wh, _ww,
                    device=_module_device(_src_vae), dtype=_vae_dt,
                    temporal_lens=(1, 1 + int(getattr(_src_vae, "ffactor_temporal", 8) or 8)),
                    autocast=_vae_ac,
                )
                _vc.warmup_encode(
                    pseudo_encode_vae, 3, _wh, _ww,
                    device=_module_device(pseudo_encode_vae), dtype=_vae_dt,
                    temporal_lens=(1,),
                    autocast=_vae_ac,
                )

            _ref_basesize = getattr(cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
            _ref_cfgs = generate_video_image_bucket(
                img_basesize=_ref_basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0,
            )
            _ref_hw = sorted({(c[3], c[4]) for c in _ref_cfgs})
            if session_settings is not None and session_settings.ref_image_sizes:
                _ref_hw = sorted({tuple(hw) for hw in session_settings.ref_image_sizes})
            _vc.maybe_setup_encode_dynamic(_src_vae)
            _vc.warmup_encode_dynamic(
                _src_vae, 3, _ref_hw,
                device=_module_device(_src_vae), dtype=_vae_dt,
                temporal_lens=(1,),
                autocast=False,
            )
        except Exception as _ve_exc:
            print(f"#####[STREAM] VAE encode compile warmup skipped: {_ve_exc!r}")

        runtime = cls(
            cfg=cfg,
            pipeline=pipeline,
            device=device_obj,
            decode_vae=decode_vae,
            pseudo_encode_vae=pseudo_encode_vae,
            postprocess_device=postprocess_device_obj,
        )

        if os.environ.get("JOYOMNI_SKIP_LOAD_WARMUP", "0").lower() in {"1", "true", "yes", "on"}:
            print("#####[STREAM] load-time warmup skipped (JOYOMNI_SKIP_LOAD_WARMUP)")
        elif session_settings is not None:
            runtime.warmup_sessions(session_settings)
        else:
            for (_wh, _ww) in _orientations:
                runtime.warmup_full_pipeline(height=_wh, width=_ww)

        if device_obj.type == "cuda" and device_obj.index is not None:
            _free_b, _total_b = torch.cuda.mem_get_info(device_obj)
            print(
                f"#####[STREAM] runtime loaded; device memory used "
                f"{(_total_b - _free_b) / 2**30:.1f}/{_total_b / 2**30:.1f} GiB "
                f"(torch reserved {torch.cuda.memory_reserved(device_obj) / 2**30:.1f} GiB, "
                f"allocated {torch.cuda.memory_allocated(device_obj) / 2**30:.1f} GiB)",
                flush=True,
            )

        return runtime

    def graph_mem_pool(self):
        """One CUDA-graph memory pool shared by every runner (graphs replay one at a time)."""
        if self._graph_mem_pool is None:
            self._graph_mem_pool = torch.cuda.graph_pool_handle()
        return self._graph_mem_pool

    def create_v2v_session(
        self,
        prompt: str,
        *,
        settings: StreamingSettings | None = None,
        ref_image: Image.Image | None = None,
    ) -> "JoyOmniV2VStreamingSession":
        return JoyOmniV2VStreamingSession(
            runtime=self,
            prompt=prompt,
            settings=settings or StreamingSettings(),
            ref_image=ref_image,
        )

    @on_compute_thread
    @torch.no_grad()
    def capture_graphs(
        self,
        shapes: list[tuple[int, int]],
        *,
        settings: StreamingSettings | None = None,
    ) -> int:
        """Capture the steady-chunk CUDA graph for every (text length, reference tokens) pair.

        For a finite, known shape set (text padded to buckets, references at one size) this
        moves every capture to load time, on the compute thread, instead of the first chunk
        of the first session that meets a new shape. `ref_tokens` is 0 for no reference.
        All runners stay resident: the cache cap is raised to hold them. Returns how many
        runners are ready.
        """
        settings = settings or StreamingSettings()
        self.graph_cache_cap = max(self.graph_cache_cap, 3 * len(shapes))
        session = self.create_v2v_session(prompt="", settings=settings)
        for txt_len, ref_tokens in shapes:
            session._maybe_prepare_graph_runner(txt_len=txt_len, ref_tokens=ref_tokens)
        return sum(1 for r in self.graph_runners.values() if getattr(r, "ready", False))

    def warmup_sessions(self, settings: StreamingSettings) -> None:
        """Run whole sessions through the live entry point (create_v2v_session + push_chunk)
        so that no chunk of a real session meets anything for the first time.

        With the text padded to `settings.text_tokens` and references at the fixed `settings.ref_image_sizes`,
        a session's attention shapes depend only on the chunk's window (chunks 0, 1, then steady),
        on whether a reference is present, and on the freeze-on-static window.  The sessions below
        walk all of them: window growth, the steady graph path, a static stretch long enough to
        take the frozen window, motion resuming, and a restart after each (what the periodic
        session reset does).  Every CUDA graph a session can replay is captured here; afterwards
        the runtime is sealed and a session never captures.
        """
        t_all = time.perf_counter()
        ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        prompt = "Turn the video into a watercolor painting."
        # One reference per fixed size (each exactly that aspect, so it lands in that bucket).
        refs = [_warmup_frame(0, w // 2, h // 2) for h, w in (settings.ref_image_sizes or ())]
        # (reference image, chunk pattern): M = moving chunk, S = static chunk (repeats the last frame)
        full = "MMMMMSSSMM"
        restart = "MMMM"
        plan = [(None, full), (None, restart)]
        for ref in refs:
            plan += [(ref, full), (ref, restart)]
        # chunk-0, chunk-1 and steady graphs for no reference and for each reference size
        self.graph_cache_cap = max(self.graph_cache_cap, 3 * (1 + len(refs)))
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        try:
            for ref_image, pattern in plan:
                t0 = time.perf_counter()
                session = self.create_v2v_session(prompt=prompt, settings=settings, ref_image=ref_image)
                elapsed = []
                try:
                    index = 0
                    for c, kind in enumerate(pattern):
                        n = 1 if c == 0 else ffactor_t
                        frames = []
                        for _ in range(n):
                            if kind == "M":
                                index += 1
                            frames.append(_warmup_frame(index, settings.width, settings.height))
                        result = session.push_chunk(frames)
                        elapsed.append(result.elapsed)
                    graph_chunks = session.graph_chunks
                    text_tokens = session.text_tokens
                finally:
                    session.close()
                print(
                    f"#####[WARMUP] session ref={ref_image is not None} chunks={pattern} "
                    f"text_tokens={text_tokens} graph_chunks={graph_chunks} in {time.perf_counter() - t0:.1f}s; "
                    f"chunk s: {' '.join(f'{e:.3f}' for e in elapsed)}",
                    flush=True,
                )
        finally:
            torch.set_rng_state(rng[0])
            if rng[1] is not None:
                torch.cuda.set_rng_state_all(rng[1])
        from xvideo.models.dit.dit import attention_shapes, seal_attention_shapes
        self.graphs_sealed = True
        seal_attention_shapes()
        graphs = sorted(k for k, r in self.graph_runners.items() if getattr(r, "ready", False))
        print(f"#####[WARMUP] graphs captured (latent_h, latent_w, text, ref_tokens, mti, steps, masked): {graphs}",
              flush=True)
        print(f"#####[WARMUP] attention shapes (q_len, kv_len, text_masked): {attention_shapes()}", flush=True)
        print(f"#####[WARMUP] done in {time.perf_counter() - t_all:.1f}s", flush=True)

    def warmup_full_pipeline(
        self,
        *,
        height: int = 720,
        width: int = 1248,
        num_chunks: int = _FULL_WARMUP_CHUNKS,
    ) -> None:
        t0 = time.time()

        ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        n_frames = 1 + max(0, num_chunks - 1) * ffactor_t + ffactor_t
        print(f"#####[STREAM] full-pipeline warmup: {num_chunks} chunks (~{n_frames} frames) at {width}x{height}")
        session = None
        try:
            settings = StreamingSettings(
                height=height,
                width=width,
                max_temporal_ids=8,
                profile_timings=False,
            )
            session = self.create_v2v_session(
                prompt="warmup", settings=settings,
            )
            rng = np.random.default_rng(0)
            completed = 0
            for i in range(n_frames):
                arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                frame = Image.fromarray(arr, mode="RGB")
                results = session.push_frame(frame, frame_meta={"seq": i + 1, "t_capture_ms": 0.0})
                completed += len(results)
                if completed >= num_chunks:
                    break
            print(f"#####[STREAM] full-pipeline warmup done: {completed} chunks in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"#####[STREAM] full-pipeline warmup skipped/failed: {exc!r}")
        finally:
            if session is not None:
                session.close()

class JoyOmniV2VStreamingSession:
    _session_serial_counter = 0

    def __init__(
        self,
        *,
        runtime: JoyOmniRuntime,
        prompt: str,
        settings: StreamingSettings,
        ref_image: Image.Image | None = None,
    ):
        self.runtime = runtime
        self.pipeline = runtime.pipeline
        JoyOmniV2VStreamingSession._session_serial_counter += 1
        self._session_serial = JoyOmniV2VStreamingSession._session_serial_counter

        self.decode_vae = runtime.decode_vae
        self.pseudo_encode_vae = runtime.pseudo_encode_vae
        self.postprocess_device = runtime.postprocess_device
        self.cfg = runtime.cfg
        self.raw_prompt = prompt
        self.prompt = prompt
        self.settings = settings
        self.ref_image = ref_image.convert("RGB") if ref_image is not None else None
        self.ref_image_latent: torch.Tensor | None = None
        self.device = self.pipeline.transformer.device
        self.generator = torch.Generator(device=self.device).manual_seed(settings.seed)

        self._prev_static_gray: np.ndarray | None = None
        self._static_anchor_id: int | None = None

        self.target_dtype = PRECISION_TO_TYPE[self.cfg.dit_precision]
        self.vae_dtype = PRECISION_TO_TYPE[self.cfg.vae_precision]
        self.vae_autocast_enabled = self.vae_dtype != torch.float32
        self.autocast_enabled = self.target_dtype != torch.float32
        self.device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":", 1)[0]

        self.chunk_size = self.pipeline._resolve_streaming_chunk_size(None, 1)
        if self.chunk_size != 1:
            raise ValueError(
                "The online v2v session currently supports transformer chunk_size=1 only; "
                f"got {self.chunk_size}."
            )

        self.ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        self.latent_channels = int(self.pipeline.vae.config.latent_channels)
        _stem = self.pipeline.vae.stem
        assert settings.height % (_stem.stride * 8) == 0 and settings.width % (_stem.stride * 8) == 0, (
            f"session size must be a multiple of {_stem.stride * 8}, got {settings.width}x{settings.height}"
        )
        self.latent_h = settings.height * _stem.group // _stem.stride // int(self.pipeline.vae_scale_factor)
        self.latent_w = settings.width * _stem.group // _stem.stride // int(self.pipeline.vae_scale_factor)
        self.local_window_size = int(getattr(self.pipeline.transformer.config, "local_window_size", 1))
        self.global_sink_chunk = self.pipeline._resolve_global_sink_chunk(None, self.pipeline.transformer)
        self.enable_denormalization = (
            self.cfg.enable_denormalization
            if settings.enable_denormalization is None
            else settings.enable_denormalization
        )

        self.initialized = False
        self.chunk_idx = 0
        self.pending_frames: list[Image.Image] = []
        self.pending_metas: list[dict[str, Any]] = []
        self.prev_source_frame: torch.Tensor | None = None
        self.ref_image_kv_prefilled = False

        self.streaming_cond_embeds: torch.Tensor | None = None
        self.streaming_cond_mask: torch.Tensor | None = None
        self.last_chunk_profile: dict[str, Any] | None = None

        self.text_tokens: int | None = None   # the prompt's own token count
        self.text_masked = False              # text padded to settings.text_tokens, padding masked
        self.graph_chunks = 0

        # Pseudo latent re-encoded from the last decoded frame of chunk k, held for the
        # decode of chunk k + 1 (the index it is stored under).
        self._pseudo_latent: torch.Tensor | None = None
        self._pseudo_latent_chunk_idx: int | None = None

    def _timer_start(
        self,
        device: torch.device | str | None = None,
        *,
        use_cuda_event: bool = True,
    ) -> _ProfileTimer | float:
        if self.settings.profile_timings:
            return _profile_timer_start(device, use_cuda_event=use_cuda_event)
        return time.perf_counter()

    def _timer_record(
        self,
        profile: dict[str, float | int],
        key: str,
        started: float,
    ) -> None:
        if not self.settings.profile_timings:
            return
        profile[key] = float(profile.get(key, 0.0)) + _profile_timer_elapsed(started)

    @property
    def frames_per_next_chunk(self) -> int:
        return 1 if self.chunk_idx == 0 else self.ffactor_t

    def _clear_vae_feature_caches(self) -> None:
        seen: set[int] = set()
        for vae in (self.pipeline.vae, self.decode_vae, self.pseudo_encode_vae):
            if vae is None or id(vae) in seen:
                continue
            seen.add(id(vae))
            clear_cache = getattr(vae, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    @on_compute_thread
    @torch.no_grad()
    def push_frame(
        self,
        frame: Image.Image | np.ndarray,
        frame_meta: dict[str, Any] | None = None,
    ) -> list[StreamingChunkResult]:
        """Add one frame; when it completes a chunk, run that chunk and return its result.

        Synchronous: every chunk this call completes has been computed and is returned.
        """
        frame = self._resize_frame(frame)
        meta = frame_meta or {"seq": self.chunk_idx + len(self.pending_frames) + 1, "t_capture_ms": time.time() * 1000.0}
        if not self.initialized:
            self._initialize(frame)
        self.pending_frames.append(frame)
        self.pending_metas.append(meta)
        results = []
        while len(self.pending_frames) >= self.frames_per_next_chunk:
            n = self.frames_per_next_chunk
            chunk_frames = self.pending_frames[:n]
            chunk_metas = self.pending_metas[:n]
            del self.pending_frames[:n]
            del self.pending_metas[:n]
            results.append(self._run_chunk(chunk_frames, chunk_metas))
        return results

    @on_compute_thread
    @torch.no_grad()
    def push_chunk(
        self,
        frames: list[Image.Image | np.ndarray],
        frame_metas: list[dict[str, Any]] | None = None,
    ) -> StreamingChunkResult:
        """Run exactly one chunk from `frames_per_next_chunk` frames and return its result."""
        if self.pending_frames:
            raise RuntimeError("push_chunk() cannot follow a partial push_frame() chunk")
        if len(frames) != self.frames_per_next_chunk:
            raise ValueError(f"chunk {self.chunk_idx} takes {self.frames_per_next_chunk} frames, got {len(frames)}")
        metas = list(frame_metas) if frame_metas is not None else [None] * len(frames)
        if any(not self._at_session_size(f) for f in frames):
            frames = list(resize_executor().map(self._resize_frame, frames))
        results: list[StreamingChunkResult] = []
        for frame, meta in zip(frames, metas):
            results += self.push_frame(frame, meta)
        assert len(results) == 1, len(results)
        return results[0]

    @on_compute_thread
    @torch.no_grad()
    def flush_pending(self) -> list[StreamingChunkResult]:
        """Pad a partial final chunk with its last frame and run it; `valid_count` marks the real frames."""
        if not self.initialized or not self.pending_frames:
            return []
        valid = len(self.pending_frames)
        pad = self.ffactor_t - valid
        chunk_frames = self.pending_frames + [self.pending_frames[-1]] * pad
        chunk_metas = self.pending_metas + [self.pending_metas[-1]] * pad
        self.pending_frames = []
        self.pending_metas = []
        return [self._run_chunk(chunk_frames, chunk_metas, valid_count=valid)]

    @on_compute_thread
    def close(self) -> None:
        if torch.cuda.is_available():
            devices = {str(self.device), str(self.postprocess_device)}
            devices.update(str(_module_device(vae)) for vae in (
                self.pipeline.vae, self.decode_vae, self.pseudo_encode_vae,
            ) if vae is not None)
            for device in devices:
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()

    @torch.no_grad()
    def _initialize(self, first_frame: Image.Image) -> None:
        if self.initialized:
            return
        init_started = time.perf_counter()

        if not getattr(self.pipeline.transformer.config, "causal", False):
            raise ValueError("Online streaming requires a causal transformer config.")
        self.pipeline.transformer.config.use_inference_kv_cache = True

        self.ref_image_latent = self._encode_ref_image_latent()

        self._encode_streaming_prompt(first_frame)

        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()
        if self.ref_image_latent is not None:
            self.pipeline._prefill_static_reference_kv_cache(
                self.pipeline.transformer,
                prompt_embeds=self.streaming_cond_embeds,
                prompt_embeds_mask=self.streaming_cond_mask,
                reference_image_latents=self.ref_image_latent,
                transformer_dtype=self.target_dtype,
            )
            self.ref_image_kv_prefilled = True

        self._maybe_prepare_graph_runner()

        self.initialized = True
        print(
            f"#####[STREAM] session init {time.perf_counter() - init_started:.3f}s "
            f"(prompt encode + reference prefill) text_tokens={self.text_tokens} "
            f"padded_to={self.streaming_cond_embeds.shape[1]} masked={self.text_masked} "
            f"ref={self.ref_image_latent is not None}",
            flush=True,
        )

    @torch.no_grad()
    def _encode_streaming_prompt(self, anchor_frame: Image.Image) -> None:
        prompt_image = _as_pil(self._resize_frame(anchor_frame))
        prompt = f"<|im_start|>user\n<image>\n{self.prompt}<|im_end|>\n"

        max_sequence_length = self.settings.max_sequence_length or int(self.cfg.text_token_max_length)
        prompt_embeds, prompt_mask = self.pipeline.encode_prompt(
            prompt=[prompt],
            images=[prompt_image],
            device=self.device,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            template_type="video",
        )

        self.text_tokens = int(prompt_embeds.shape[1])
        target = self.settings.text_tokens
        if target is not None and self.text_tokens <= target:
            pad = target - self.text_tokens
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad))
            prompt_mask = torch.nn.functional.pad(prompt_mask, (0, pad))
            self.text_masked = True
        elif target is not None:
            print(
                f"#####[STREAM] WARNING: prompt is {self.text_tokens} text tokens, over the "
                f"{target} the runtime is warmed for; running it unpadded and unwarmed "
                f"(its first chunks build new attention plans, and no CUDA graph)",
                flush=True,
            )
        self.streaming_cond_embeds = prompt_embeds
        self.streaming_cond_mask = prompt_mask

    @torch.no_grad()
    def _encode_ref_image_latent(self) -> torch.Tensor | None:
        if self.ref_image is None:
            return None

        ref_image_basesize = getattr(self.cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
        ref_image_bucket_configs = generate_video_image_bucket(
            img_basesize=ref_image_basesize,
            bs_img=1,
            bs_vid=0,
            bs_mimg=0,
            bs_mvid=0,
        )
        if self.settings.ref_image_sizes:
            size = _nearest_aspect(self.ref_image, self.settings.ref_image_sizes)
            resized_image = _letterbox(self.ref_image, size)
            print(f"#####[STREAM] reference {self.ref_image.width}x{self.ref_image.height} -> "
                  f"{size[1]}x{size[0]} (letterboxed)", flush=True)
        else:
            resized_image, _ = _dynamic_resize_from_bucket(
                self.ref_image,
                bucket_configs=ref_image_bucket_configs,
                num_frames=1,
                num_items=1,
                return_bucket=True,
            )
        pixel = torch.from_numpy(np.array(resized_image))
        pixel = rearrange(pixel, "h w c -> c h w")
        if pixel.dtype != torch.uint8:
            pixel = pixel.clamp(0, 255).to(torch.uint8)
        pixel_normalized = pixel.to(torch.float32) / 127.5 - 1.0

        ref_img_tensor = rearrange(pixel_normalized, "c h w -> 1 c 1 h w")
        encode_device = _module_device(self.pipeline.vae)
        ref_img_encoded = ref_img_tensor.to(device=encode_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        encoded = _vc.encode_via_dynamic(self.pipeline.vae, ref_img_encoded)
        if not hasattr(encoded, "latent_dist"):
            raise TypeError(f"Unsupported VAE encode output type for ref image: {type(encoded)}")
        ref_img_latent = encoded.latent_dist.sample()
        if self.enable_denormalization:
            ref_img_latent = self.pipeline.normalize_latents(ref_img_latent)

        return ref_img_latent[:, :, :1].to(device=self.device, dtype=self.target_dtype)

    @staticmethod
    def _chunk_last_frame_gray(source_frames: list[Image.Image]) -> np.ndarray | None:
        if not source_frames:
            return None
        frame = _as_pil(source_frames[-1]).convert("L").resize((64, 36))
        return np.asarray(frame, dtype=np.float32)

    def _update_static_anchor(self, chunk_idx: int, source_frames: list[Image.Image]) -> int | None:
        if not self.settings.freeze_kv_on_static:
            return None
        gray = self._chunk_last_frame_gray(source_frames)
        if chunk_idx == 0 or gray is None:
            self._prev_static_gray = gray if gray is not None else self._prev_static_gray
            self._static_anchor_id = None
            return None

        anchor_id: int | None = None
        if self._prev_static_gray is not None:
            mad = float(np.mean(np.abs(gray - self._prev_static_gray)))
            if mad < self.settings.static_diff_thresh:
                if self._static_anchor_id is None:
                    self._static_anchor_id = chunk_idx - 1
                    print(
                        f"#####[FREEZE-KV] static detected at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f} < {self.settings.static_diff_thresh}), "
                        f"anchor={self._static_anchor_id}",
                        flush=True,
                    )
                anchor_id = self._static_anchor_id
            else:
                if self._static_anchor_id is not None:
                    print(
                        f"#####[FREEZE-KV] motion resumed at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f}), anchor released",
                        flush=True,
                    )
                self._static_anchor_id = None
                anchor_id = None

        self._prev_static_gray = gray
        return anchor_id

    def _new_profile(self, chunk_idx: int, input_frames: int) -> dict[str, Any]:
        return {
            "chunk_idx": chunk_idx,
            "input_frames": input_frames,
            "steps": int(self.settings.num_inference_steps),
            "profile_timings": int(self.settings.profile_timings),
            "dit_device": str(self.device),
            "vae_encode_device": str(_module_device(self.pipeline.vae)),
            "vae_decode_device": str(_module_device(self.decode_vae)),
            "vae_pseudo_device": str(_module_device(self.pseudo_encode_vae)),
            "postprocess_device": str(self.postprocess_device),
        }

    def _graph_runner_for_chunk(
        self,
        *,
        chunk_idx: int,
        selected_chunk_ids: list[int],
        gather_chunk_ids: list[int],
    ):
        mti = self.settings.max_temporal_ids
        mti = GRAPH_WINDOW_CHUNKS - 1 if mti is None else mti
        if not graph_env_enabled() or self.settings.num_inference_steps < 1:
            return None
        if self.chunk_size != 1 or not self.settings.store_clean_self_only:
            return None
        # chunk 0: [0]; chunk 1: [0, 1]; then [0, k-1, k] -- or, frozen on a static input,
        # KV window [0, anchor, k] at the positions of [0, k-1, k].
        if chunk_idx == 0:
            history = 0
            ok = selected_chunk_ids == gather_chunk_ids == [0]
        elif chunk_idx == 1:
            history = 1
            ok = selected_chunk_ids == gather_chunk_ids == [0, 1]
        else:
            history = 2
            ok = (gather_chunk_ids == [0, chunk_idx - 1, chunk_idx] and len(selected_chunk_ids) == 3
                  and selected_chunk_ids[0] == 0 and selected_chunk_ids[2] == chunk_idx)
        if not ok:
            return None

        runner = self.runtime.graph_runners.get(self._graph_key(history=history))
        if runner is None or not getattr(runner, "ready", False):
            return None

        if runner.bound_token != self._session_serial:
            runner.bind_session(self._session_serial, self.streaming_cond_embeds, self.streaming_cond_mask)
        if (history < 2 or runner.pool_dirty or runner.last_graph_chunk != chunk_idx - 1
                or selected_chunk_ids[1] != chunk_idx - 1):
            scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
            sink_mem_id = self.pipeline._kv_cache_memory_id("clean", 0) if history >= 1 else None
            prev1_mem_id = (
                self.pipeline._kv_cache_memory_id("clean", selected_chunk_ids[1]) if history == 2 else None
            )
            ref_mem_id = (
                self.pipeline._kv_cache_memory_id("ref_image")
                if self.ref_image_kv_prefilled else None
            )
            needed = [m for m in (sink_mem_id, prev1_mem_id, ref_mem_id) if m is not None]
            if any(mem_id not in scope_store for mem_id in needed):
                # History not in the python cache (session teardown drain / transition):
                # this chunk is simply not eligible; anything else raising below is a real bug.
                return None
            runner.seed_from_cache(
                scope_store,
                sink_mem_id=sink_mem_id,
                prev1_mem_id=prev1_mem_id,
                prev1_pos=min(chunk_idx - 1, int(mti) - 1),
                ref_mem_id=ref_mem_id,
            )
        runner.set_positions(chunk_idx)
        return runner

    def _ref_kv_tokens(self) -> int:
        if not self.ref_image_kv_prefilled:
            return 0
        scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
        store = scope_store[self.pipeline._kv_cache_memory_id("ref_image")]
        return int(store[0]["key"].shape[1])

    def _graph_key(self, txt_len: int | None = None, ref_tokens: int | None = None, history: int = 2) -> tuple:
        return (
            self.latent_h,
            self.latent_w,
            int(self.streaming_cond_embeds.shape[1]) if txt_len is None else int(txt_len),
            self._ref_kv_tokens() if ref_tokens is None else int(ref_tokens),
            int(self.settings.max_temporal_ids or 0),
            int(self.settings.num_inference_steps),
            self._text_masked_for(int(self.streaming_cond_embeds.shape[1]) if txt_len is None else int(txt_len)),
            int(history),
        )

    def _text_masked_for(self, txt_len: int) -> bool:
        return self.settings.text_tokens is not None and txt_len == int(self.settings.text_tokens)

    def _maybe_prepare_graph_runner(self, txt_len: int | None = None, ref_tokens: int | None = None) -> None:
        """Capture the chunk-0, chunk-1 and steady-chunk graphs for (text length, reference
        tokens), if not cached.

        Defaults to this session's own prompt and reference; JoyOmniRuntime.capture_graphs
        passes them explicitly to capture a known shape set ahead of any session.
        """
        for history in (0, 1, 2):
            self._maybe_prepare_graph_runner_phase(txt_len, ref_tokens, history)

    def _maybe_prepare_graph_runner_phase(self, txt_len: int | None, ref_tokens: int | None, history: int) -> None:
        txt_len = int(self.streaming_cond_embeds.shape[1]) if txt_len is None else int(txt_len)
        ref_tokens = self._ref_kv_tokens() if ref_tokens is None else int(ref_tokens)
        mti = self.settings.max_temporal_ids
        mti = GRAPH_WINDOW_CHUNKS - 1 if mti is None else mti
        if not graph_env_enabled() or self.chunk_size != 1:
            return
        if not self.settings.store_clean_self_only:
            return
        if not self.global_sink_chunk or self.local_window_size != GRAPH_WINDOW_CHUNKS:
            return
        key = self._graph_key(txt_len, ref_tokens, history)
        cache = self.runtime.graph_runners
        fails = self.runtime.graph_capture_failures
        existing = cache.get(key)
        if existing is not None and getattr(existing, "ready", False):
            return
        if fails.get(key, 0) >= _GRAPH_CAPTURE_MAX_FAILS:
            return
        if self.runtime.graphs_sealed:
            print(f"#####[GRAPH] no warmed graph for {key}; this session runs eager (no capture "
                  f"inside a live session)", flush=True)
            return
        torch.cuda.synchronize(self.device)
        live_keys = [k for k, v in cache.items() if v is not None]
        while len(live_keys) >= self.runtime.graph_cache_cap:
            del cache[live_keys.pop(0)]
        cache.pop(key, None)
        # No empty_cache: the evicted runner's freed blocks are reused for the
        # new static bufs; returning them to the driver only slows recapture.
        pos_table_ids = torch.arange(int(mti), device=self.device, dtype=torch.long)
        latent_shape = (1, self.latent_channels, self.chunk_size, self.latent_h, self.latent_w)
        runner = StreamingGraphRunner(
            self.pipeline.transformer,
            chunk_tokens=self.latent_h * self.latent_w,
            ref_tokens=ref_tokens,
            latent_shape=latent_shape,
            txt_len=txt_len,
            max_temporal_ids=int(mti),
            pos_freqs=self._graph_cached_freqs(pos_table_ids),
            device=self.device,
            dtype=self.target_dtype,
            autocast_ctx=lambda: _autocast_ctx(
                self.device_type, self.target_dtype, self.autocast_enabled
            ),
            mask_text_padding=self._text_masked_for(txt_len),
            history_chunks=history,
            mem_pool=self.runtime.graph_mem_pool(),
        )
        try:
            self.pipeline.scheduler.set_timesteps(
                self.settings.num_inference_steps, device=self.device
            )
            runner.capture(
                timesteps=self.pipeline.scheduler.timesteps,
                sigmas=self.pipeline.scheduler.sigmas,
            )
        except Exception as exc:  # noqa: BLE001
            fails[key] = fails.get(key, 0) + 1
            print(
                f"#####[GRAPH] capture failed (attempt {fails[key]}/{_GRAPH_CAPTURE_MAX_FAILS}), "
                f"session stays eager: {exc!r}",
                flush=True,
            )
            del runner
            torch.cuda.empty_cache()
            return
        cache[key] = runner

    def _graph_cached_freqs(self, cached_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tf = self.pipeline.transformer
        cached_frame_ids = tf._get_token_frame_ids(
            (int(cached_ids.numel()), self.latent_h, self.latent_w),
            self.device,
            temporal_ids=cached_ids,
        )
        cos, sin = tf.get_rotary_pos_embed_from_ids(
            frame_ids=cached_frame_ids,
            spatial_shape=(self.latent_h, self.latent_w),
        )
        return cos.unsqueeze(0), sin.unsqueeze(0)

    @torch.no_grad()
    def _denoise_chunk_graph(
        self,
        runner,
        ref_chunk_latent: torch.Tensor,
        current_chunk_latents: torch.Tensor,
        *,
        profile: dict[str, Any],
        history_chunk_ids: list[int],
        active_chunk_id: int,
    ) -> torch.Tensor:
        profile["graph_path"] = 1
        self.graph_chunks += 1
        runner.in_ref_latent.copy_(ref_chunk_latent.to(self.target_dtype))
        started = self._timer_start(self.device)
        runner.in_noise.copy_(current_chunk_latents.to(self.target_dtype))
        runner.full_graph.replay()
        self._timer_record(profile, "dit_forward_cond_s", started)

        keep_before_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if self.ref_image_kv_prefilled:
            keep_before_store.add(self.pipeline._kv_cache_memory_id("ref_image"))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_before_store)

        started = self._timer_start(self.device)
        scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
        runner.publish_to_cache(
            scope_store, self.pipeline._kv_cache_memory_id("clean", active_chunk_id)
        )
        runner.last_graph_chunk = active_chunk_id
        self._timer_record(profile, "kv_store_forward_s", started)
        return runner.out_latents.clone()

    def _denoise_chunk(
        self,
        ref_chunk_latent: torch.Tensor,
        *,
        profile: dict[str, Any],
        chunk_idx: int,
        frozen_anchor_id: int | None = None,
    ) -> torch.Tensor:
        noise_shape = (1, self.latent_channels, self.chunk_size, self.latent_h, self.latent_w)
        current_chunk_latents = randn_tensor(
            noise_shape,
            generator=self.generator,
            device=self.device,
            dtype=self.target_dtype,
        )

        total_latent_frames = chunk_idx + 1
        chunk_window = self.pipeline._get_chunk_window(
            chunk_idx=chunk_idx,
            total_latent_frames=total_latent_frames,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        selected_chunk_ids = chunk_window["selected_chunk_ids"]
        history_chunk_ids = selected_chunk_ids[:-1]
        active_chunk_id = selected_chunk_ids[-1]

        gather_chunk_ids = selected_chunk_ids
        if (
            frozen_anchor_id is not None and
            history_chunk_ids and
            history_chunk_ids[-1] != frozen_anchor_id
        ):
            old_tail = history_chunk_ids[-1]

            fz_history = history_chunk_ids[:-1]
            anchor_is_dup = frozen_anchor_id in fz_history
            fz_history_with_anchor = fz_history if anchor_is_dup else fz_history + [frozen_anchor_id]
            fz_selected = fz_history_with_anchor + [active_chunk_id]

            disguised_tail = active_chunk_id - 1
            fz_gather_head = fz_history_with_anchor[:-1]
            degenerate = (
                anchor_is_dup or
                disguised_tail < 0 or
                disguised_tail in fz_gather_head
            )
            if not degenerate:
                history_chunk_ids = fz_history_with_anchor
                selected_chunk_ids = fz_selected
                gather_chunk_ids = fz_gather_head + [disguised_tail, active_chunk_id]
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} tail {old_tail}->anchor "
                    f"{frozen_anchor_id} (KV), pos-disguise->{disguised_tail}, "
                    f"kv_window={selected_chunk_ids} pos_window={gather_chunk_ids}",
                    flush=True,
                )
            else:
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} anchor={frozen_anchor_id} "
                    f"degenerate (dup_or_collision), falling back to normal path",
                    flush=True,
                )
        window_ids = self.pipeline._gather_window_temporal_ids(
            gather_chunk_ids,
            self.chunk_size,
            total_latent_frames,
            torch.device("cpu"),
            max_temporal_ids=self.settings.max_temporal_ids,
        )
        current_chunk_temporal_ids = window_ids[-self.chunk_size:]
        cached_temporal_ids = window_ids[: -self.chunk_size]
        if cached_temporal_ids.numel() == 0:
            cached_temporal_ids = None

        runner = self._graph_runner_for_chunk(
            chunk_idx=chunk_idx,
            selected_chunk_ids=selected_chunk_ids,
            gather_chunk_ids=gather_chunk_ids,
        )
        if runner is not None:
            return self._denoise_chunk_graph(
                runner,
                ref_chunk_latent,
                current_chunk_latents,
                profile=profile,
                history_chunk_ids=history_chunk_ids,
                active_chunk_id=active_chunk_id,
            )

        self.pipeline.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        timesteps_for_chunk = self.pipeline.scheduler.timesteps

        for timestep in timesteps_for_chunk:
            autocast_context = _autocast_ctx(
                self.device_type, self.target_dtype, self.autocast_enabled
            )
            with autocast_context:
                latent_model_input = current_chunk_latents.to(self.target_dtype)
                cache_memory_ids = [
                    self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids
                ]
                if self.ref_image_kv_prefilled:
                    cache_memory_ids.append(self.pipeline._kv_cache_memory_id("ref_image"))

                t_expand = timestep.repeat(latent_model_input.shape[0])
                with self.pipeline.transformer.cache_context("cond"):
                    started = self._timer_start(self.device)
                    noise_pred = self.pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=self.streaming_cond_embeds,
                        encoder_hidden_states_mask=self.streaming_cond_mask,
                        ref_video_latent=ref_chunk_latent,
                        current_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                            latent_model_input.shape[0], -1
                        ),
                        cached_temporal_ids=(
                            cached_temporal_ids.unsqueeze(0).expand(latent_model_input.shape[0], -1)
                            if cached_temporal_ids is not None
                            else None
                        ),
                        kv_cache_mode="reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=active_chunk_id,
                        kv_cache_selected_chunk_ids=cache_memory_ids,
                        kv_cache_pre_rope=True,
                        mask_text_padding=self.text_masked,
                    )[0]
                    self._timer_record(profile, "dit_forward_cond_s", started)

                sample_for_step = current_chunk_latents.clone()
                current_chunk_latents = self.pipeline.scheduler.step(
                    noise_pred,
                    timestep,
                    sample_for_step,
                    return_dict=False,
                )[0]

        keep_before_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if self.ref_image_kv_prefilled:
            keep_before_store.add(self.pipeline._kv_cache_memory_id("ref_image"))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_before_store)

        started = self._timer_start(self.device)
        store_self_only = self.settings.store_clean_self_only
        store_history_chunk_ids = [] if store_self_only else history_chunk_ids
        store_mode = "store" if store_self_only else "reuse_store"
        store_cached_temporal_ids = None if store_self_only else cached_temporal_ids
        self._timer_record(profile, "kv_store_setup_s", started)

        started = self._timer_start(self.device)
        self.pipeline._store_clean_chunk_kv_cache(
            self.pipeline.transformer,
            clean_chunk_latents=current_chunk_latents.to(self.target_dtype),
            chunk_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                current_chunk_latents.shape[0], -1
            ),
            prompt_embeds=self.streaming_cond_embeds,
            prompt_embeds_mask=self.streaming_cond_mask,
            active_chunk_id=active_chunk_id,
            history_chunk_ids=store_history_chunk_ids,
            pre_rope=True,
            cached_temporal_ids=store_cached_temporal_ids,
            store_mode=store_mode,
        )
        self._timer_record(profile, "kv_store_forward_s", started)
        return current_chunk_latents

    def _evict_after_store(
        self, chunk_idx: int, frozen_anchor_id: int | None = None
    ) -> None:
        next_selected = self._next_selected_chunk_ids(chunk_idx)
        keep_after_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in next_selected}
        if self.ref_image_kv_prefilled:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("ref_image"))

        if frozen_anchor_id is not None:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("clean", frozen_anchor_id))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_after_store)

    def _finish_chunk_profile(
        self,
        profile: dict[str, Any],
        total_started: float,
    ) -> None:
        if self.settings.profile_timings:
            self._timer_record(profile, "total_server_chunk_s", total_started)
        else:
            profile["total_server_chunk_s"] = time.perf_counter() - total_started
        if self.settings.profile_timings:
            profile["dit_denoise_s"] = float(profile.get("dit_forward_cond_s", 0.0))
            profile["kv_store_s"] = (
                float(profile.get("kv_store_setup_s", 0.0)) +
                float(profile.get("kv_store_forward_s", 0.0))
            )
        self.last_chunk_profile = profile

    def _log_chunk_profile(self, profile: dict[str, Any], input_frames: int, output_frames: int) -> None:
        elapsed = float(profile["total_server_chunk_s"])
        if self.settings.profile_timings:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s "
                f"vae_enc={float(profile.get('vae_encode_s', 0.0)):.3f}s "
                f"dit={float(profile.get('dit_denoise_s', 0.0)):.3f}s "
                f"kv_store={float(profile.get('kv_store_s', 0.0)):.3f}s "
                f"vae_dec={float(profile.get('vae_decode_s', 0.0)):.3f}s "
                f"jpeg={float(profile.get('jpeg_encode_s', 0.0)):.3f}s",
                flush=True,
            )
        else:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s profile_timings=off"
            )

    @staticmethod
    def _align_source_metas(
        source_metas: list[dict[str, Any]],
        output_count: int,
    ) -> list[dict[str, Any]]:
        if not source_metas:
            return [{} for _ in range(output_count)]
        if len(source_metas) >= output_count:
            return source_metas[-output_count:]
        return [source_metas[-1]] * output_count

    @torch.no_grad()
    def _run_chunk(
        self,
        source_frames: list[Image.Image],
        source_metas: list[dict[str, Any]],
        valid_count: int | None = None,
    ) -> StreamingChunkResult:
        """Run one chunk end to end on the calling thread and the current CUDA stream.

        encode -> DiT denoise (+ KV store) -> decode -> postprocess + copy to host ->
        pseudo-encode, in that stream order. The host waits only for the copy to host, so the
        pseudo-encode (the history the NEXT chunk's decode consumes, stream-ordered before it)
        can still be running on the GPU when this returns with chunk k's frames.

        The CPU-side static-input check runs after the encode is queued, so it overlaps the
        encode on the GPU instead of delaying it.
        """
        chunk_idx = self.chunk_idx
        total_started = time.perf_counter()
        profile = self._new_profile(chunk_idx, len(source_frames))

        started = self._timer_start()
        ref_chunk_latent = self._encode_reference_chunk(
            source_frames, profile=profile, chunk_idx=chunk_idx,
        )
        self._timer_record(profile, "reference_prepare_s", started)
        frozen_anchor_id = self._update_static_anchor(chunk_idx, source_frames)

        current_chunk_latents = self._denoise_chunk(
            ref_chunk_latent,
            profile=profile,
            chunk_idx=chunk_idx,
            frozen_anchor_id=frozen_anchor_id,
        )
        self._evict_after_store(chunk_idx, frozen_anchor_id=frozen_anchor_id)

        decoded_pixels = self._decode_chunk_pixels(
            current_chunk_latents, profile=profile, chunk_idx=chunk_idx,
        )
        packed = self._pack_decoded_pixels(
            decoded_pixels, profile=profile, chunk_idx=chunk_idx,
        )
        host_ready = None
        if torch.device(self.postprocess_device).type == "cuda":
            host_ready = torch.cuda.Event()
            host_ready.record(torch.cuda.current_stream(self.postprocess_device))
        self._encode_next_decode_pseudo_latent(
            decoded_pixels, profile=profile, chunk_idx=chunk_idx,
        )
        if host_ready is not None:
            host_ready.synchronize()

        frames = packed if isinstance(packed, np.ndarray) else None
        if frames is not None:
            packed = list(frames)
        n_out = len(packed)
        self._finish_chunk_profile(profile, total_started)
        self._log_chunk_profile(profile, len(source_frames), n_out)
        self.chunk_idx += 1
        return StreamingChunkResult(
            jpegs=packed,
            frames=frames,
            profile=profile,
            source_metas=self._align_source_metas(source_metas, n_out),
            elapsed=float(profile.get("total_server_chunk_s", 0.0)),
            valid_count=valid_count,
        )

    @torch.no_grad()
    def _encode_reference_chunk(
        self,
        source_frames: list[Image.Image],
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        if chunk_idx == 0:
            started = self._timer_start()
            source_window = self._frames_to_tensor(source_frames[:1])
            if profile is not None:
                self._timer_record(profile, "frames_to_tensor_s", started)
        else:
            if self.prev_source_frame is None:
                raise RuntimeError("Missing previous source frame for streaming VAE encode.")
            if len(source_frames) != self.ffactor_t:
                raise ValueError(
                    f"Expected {self.ffactor_t} frames after the first chunk, got {len(source_frames)}."
                )
            started = self._timer_start()
            new_frames = self._frames_to_tensor(source_frames)
            prev = self.prev_source_frame
            if prev.device != new_frames.device:
                prev = prev.to(new_frames.device)
            source_window = torch.cat([prev, new_frames], dim=2)
            if profile is not None:
                self._timer_record(profile, "frames_to_tensor_s", started)
        self.prev_source_frame = source_window[:, :, -1:].detach().clone()
        encode_device = _module_device(self.pipeline.vae)
        source_window = source_window.to(device=encode_device, dtype=self.target_dtype)

        _vc = _vae_compile_module()
        _vc.maybe_setup_encode(self.pipeline.vae)
        source_window = _vc.prep_input(source_window)
        started = self._timer_start(encode_device)

        _enc_dev_type = torch.device(encode_device).type
        _enc_ctx = _autocast_ctx(_enc_dev_type, self.vae_dtype, self.vae_autocast_enabled)
        with _enc_ctx:
            # Encode the startup frame or the complete overlapping window once.
            # The sequence wrapper also encodes its first frame separately,
            # but streaming only retains the final latent below.
            ref_latent = self.pipeline._encode_vae_single(
                source_window,
                enable_denormalization=self.enable_denormalization,
            )
        if profile is not None:
            self._timer_record(profile, "vae_encode_s", started)
        ref_latent = ref_latent[:, :, -self.chunk_size:].to(device=self.device, dtype=self.target_dtype)
        return ref_latent

    def _take_decode_pseudo_latent(
        self,
        chunk_idx: int,
        decode_device: torch.device,
    ) -> torch.Tensor:
        if self._pseudo_latent_chunk_idx != chunk_idx or self._pseudo_latent is None:
            raise RuntimeError(
                f"decode of chunk {chunk_idx} needs the pseudo latent re-encoded from chunk "
                f"{chunk_idx - 1}; held: {self._pseudo_latent_chunk_idx}"
            )
        pseudo_latent = self._pseudo_latent
        self._pseudo_latent = None
        self._pseudo_latent_chunk_idx = None
        return pseudo_latent.to(device=decode_device, dtype=self.target_dtype)

    def _store_decode_pseudo_latent(self, chunk_idx: int, pseudo_latent: torch.Tensor) -> None:
        self._pseudo_latent = pseudo_latent.detach()
        self._pseudo_latent_chunk_idx = chunk_idx

    @torch.no_grad()
    def _decode_chunk_pixels(
        self,
        current_chunk_latents: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        decode_vae = self.decode_vae
        vae_device = _module_device(decode_vae)
        chunk_lat_flat = current_chunk_latents
        if self.enable_denormalization:
            chunk_lat_flat = self.pipeline.denormalize_latents(chunk_lat_flat)

        vae_device_type = vae_device.type
        chunk_lat_flat = chunk_lat_flat.to(vae_device)

        if chunk_idx > 0:
            pseudo_latent = self._take_decode_pseudo_latent(chunk_idx, vae_device)
            decode_input = torch.cat([pseudo_latent, chunk_lat_flat], dim=2)
            del pseudo_latent
        else:
            decode_input = chunk_lat_flat

        _vc = _vae_compile_module()
        _vc.maybe_setup_decode(decode_vae)
        decode_input = _vc.prep_input(decode_input)

        vae_ctx = _autocast_ctx(vae_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with vae_ctx:
            started = self._timer_start(vae_device)
            chunk_decoded = decode_vae.decode(decode_input, return_dict=False)[0]
            if profile is not None:
                self._timer_record(profile, "vae_decode_s", started)

        if chunk_idx > 0:
            window_pixels = self.chunk_size * self.ffactor_t
            chunk_decoded = chunk_decoded[:, :, -window_pixels:]
        return chunk_decoded.detach()

    @torch.no_grad()
    def _encode_next_decode_pseudo_latent(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
    ) -> None:
        pseudo_vae = self.pseudo_encode_vae
        pseudo_device = _module_device(pseudo_vae)
        pseudo_device_type = pseudo_device.type
        prev_pixels = decoded_pixels[:, :, -1:].detach().to(device=pseudo_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        _vc.maybe_setup_encode(pseudo_vae)
        prev_pixels = _vc.prep_input(prev_pixels)
        vae_ctx = _autocast_ctx(pseudo_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with vae_ctx:
            pseudo_enc = pseudo_vae.encode(prev_pixels)
            if hasattr(pseudo_enc, "latent_dist"):
                pseudo_latent = pseudo_enc.latent_dist.sample()
            else:
                pseudo_latent = pseudo_enc
        self._store_decode_pseudo_latent(chunk_idx + 1, pseudo_latent)
        del prev_pixels, pseudo_enc, pseudo_latent

    @torch.no_grad()
    def _pack_decoded_pixels(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
    ) -> list[bytes] | np.ndarray:
        post_device = self.postprocess_device

        chunk_decoded = decoded_pixels.to(device=post_device, dtype=torch.float32)
        frames_u8 = (
            (chunk_decoded / 2 + 0.5)
            .clamp(0, 1)
            .mul_(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)[0]
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        if self.runtime.lossless_output or self.settings.output_codec == "h264":
            return _to_host(frames_u8).numpy()

        quality = max(1, min(100, int(self.runtime.output_quality)))
        global _NVJPEG_OK
        if _NVJPEG_OK and frames_u8.device.type == "cuda":
            started = time.perf_counter()
            try:
                chw = frames_u8.permute(0, 3, 1, 2).contiguous()
                torch.cuda.current_stream(chw.device).synchronize()
                encoded = _tv_encode_jpeg(list(chw.unbind(0)), quality=quality)
                torch.cuda.synchronize(chw.device)
                jpegs = [bytes(e.cpu().numpy()) for e in encoded]
                if profile is not None:
                    profile["jpeg_encode_s"] = float(profile.get("jpeg_encode_s", 0.0)) + (
                        time.perf_counter() - started
                    )
                return jpegs
            except Exception as exc:  # noqa: BLE001
                _NVJPEG_OK = False
                print(f"#####[STREAM] nvJPEG encode failed, falling back to cv2: {exc!r}", flush=True)

        arr = frames_u8.cpu().numpy()
        started = time.perf_counter()

        import cv2
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        jpegs = []
        for t in range(arr.shape[0]):
            bgr = cv2.cvtColor(arr[t], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, enc_params)
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for frame {t} of chunk {chunk_idx}")
            jpegs.append(buf.tobytes())
        if profile is not None:
            profile["jpeg_encode_s"] = float(profile.get("jpeg_encode_s", 0.0)) + (time.perf_counter() - started)
        return jpegs

    def _next_selected_chunk_ids(self, chunk_idx: int | None = None) -> list[int]:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        next_total = (chunk_idx + 2) * self.chunk_size
        window = self.pipeline._get_chunk_window(
            chunk_idx=chunk_idx + 1,
            total_latent_frames=next_total,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        return window["selected_chunk_ids"]

    def _at_session_size(self, frame: Image.Image | np.ndarray) -> bool:
        if isinstance(frame, np.ndarray):
            return frame.dtype == np.uint8 and frame.shape == (self.settings.height, self.settings.width, 3)
        return frame.mode == "RGB" and frame.size == (self.settings.width, self.settings.height)

    def _resize_frame(self, frame: Image.Image | np.ndarray) -> Image.Image | np.ndarray:
        # A frame is a PIL image or an (H, W, 3) uint8 RGB array. Either is used as it is when it
        # already has the session size (frames are never written in place); anything else goes
        # through PIL. An array skips the PIL round trip, which copies and repacks every pixel.
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.uint8 and frame.shape == (self.settings.height, self.settings.width, 3):
                return frame
            frame = Image.fromarray(frame)
        # convert() copies even when the mode already matches.
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        if frame.size == (self.settings.width, self.settings.height):
            return frame
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return frame.resize((self.settings.width, self.settings.height), resampling)

    def _frames_to_tensor(self, frames: list[Image.Image]) -> torch.Tensor:
        arrays = [np.asarray(self._resize_frame(frame), dtype=np.uint8) for frame in frames]
        encode_device = _module_device(self.pipeline.vae) if torch.cuda.is_available() else torch.device("cpu")
        if encode_device.type == "cuda":
            # Stack straight into pinned memory so the upload is a real async DMA; the
            # pinned block is recycled by the caching host allocator once the copy is done.
            staged = torch.empty((len(arrays),) + arrays[0].shape, dtype=torch.uint8, pin_memory=True)
            np.stack(arrays, axis=0, out=staged.numpy())
            u8 = staged.to(encode_device, non_blocking=True)
        else:
            u8 = torch.from_numpy(np.stack(arrays, axis=0))
        pixel = rearrange(u8, "t h w c -> 1 c t h w").to(torch.float32)
        return pixel / 127.5 - 1.0
