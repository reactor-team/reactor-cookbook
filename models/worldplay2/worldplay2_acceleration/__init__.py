"""Runtime kernels for WorldPlay2."""

from __future__ import annotations

import numpy as np
import torch

from worldplay2_inference import model as wp2_model

from . import vae_split


def _set(obj, name, value):
    setattr(obj, name, value)


def _install_fp32_tables(pipeline):
    _set(wp2_model, "TABLE_DTYPE", torch.float32)
    for model in _experts(pipeline):
        _set(model, "freqs", model.freqs.to(torch.complex64))


def _install_tf32():
    _set(torch.backends.cuda.matmul, "allow_tf32", True)
    _set(torch.backends.cudnn, "allow_tf32", True)


def _experts(pipeline):
    return (
        pipeline._find_wan_model(pipeline.high_noise_model),
        pipeline._find_wan_model(pipeline.low_noise_model),
    )


def _install_prefill_attention(pipeline):
    for model in _experts(pipeline):
        for block in model.blocks:
            _set(block.self_attn, "prefill_attention", "split")


def _install_switches(pipeline, pipeline_flags=(), model_flags=()):
    for name in pipeline_flags:
        _set(pipeline, name, True)
    for model in _experts(pipeline):
        for name in model_flags:
            _set(model, name, True)


def rgb8(rgb):
    return (
        rgb.float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .to(dtype=torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
    )


def _install_async_rgb(pipeline, engine):
    side = torch.cuda.Stream()
    decode = pipeline._decode_chunk_and_lr_encode
    original = engine._host_frames
    pending = {}

    def decode_and_start_copy(*args, **kwargs):
        out = decode(*args, **kwargs)
        rgb = out[1]
        pending.clear()
        if rgb is not None and rgb.is_cuda:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                u8 = rgb8(rgb)
                host = torch.empty(u8.shape, dtype=torch.uint8, pin_memory=True)
                host.copy_(u8, non_blocking=True)
                done = torch.cuda.Event()
                done.record(side)
            rgb.record_stream(side)
            pending.update(rgb=rgb, u8=u8, host=host, done=done)
        return out

    def host_frames(rgb) -> np.ndarray:
        slot = dict(pending)
        pending.clear()
        if slot.get("rgb") is rgb:
            slot["done"].synchronize()
            return slot["host"].numpy()
        return original(rgb)

    _set(pipeline, "_decode_chunk_and_lr_encode", decode_and_start_copy)
    _set(engine, "_host_frames", host_frames)


TECHNIQUES = {
    "fp32_tables": lambda p, e: _install_fp32_tables(p),
    "tf32": lambda p, e: _install_tf32(),
    "zero_tail_clip": lambda p, e: _install_switches(p, ("lazy_zero_clip",)),
    "kv_inplace": lambda p, e: _install_switches(p, ("kv_inplace",)),
    "rope_window": lambda p, e: _install_switches(p, model_flags=("rope_window",)),
    "e0_slice": lambda p, e: _install_switches(p, ("slim_memory",), ("e0_slice",)),
    "compress_window": lambda p, e: _install_switches(p, ("compress_window",)),
    "chunk_stream": lambda p, e: _install_switches(p, ("chunk_stream",)),
    "chunk_noise": lambda p, e: _set(p, "noise_mode", "chunk"),
    "prefill_split": lambda p, e: _install_prefill_attention(p),
    "vae_split": lambda p, e: vae_split.install(p, _set),
    "async_rgb": _install_async_rgb,
}

_COMMON = (
    "fp32_tables",
    "tf32",
    "zero_tail_clip",
    "kv_inplace",
    "rope_window",
    "e0_slice",
    "prefill_split",
    "compress_window",
    "chunk_stream",
    "chunk_noise",
)
_PROFILES = {
    1: _COMMON + ("async_rgb",),
    2: _COMMON + ("vae_split", "async_rgb"),
    4: _COMMON + ("vae_split", "async_rgb"),
}


def install(engine) -> None:
    """Install the fixed acceleration profile for this worker count.

    FA4, FP32 tables and TF32 can change rounding; no bitwise-equivalence
    guarantee is implied by these inference optimizations.
    """
    selected = _PROFILES[engine.world_size]
    installed = {name: TECHNIQUES[name](engine._pipeline, engine) for name in selected}
    if "vae_split" in installed:
        decoder = installed["vae_split"]
        reset = engine.reset

        def release_reset():
            reset()
            decoder.reset()

        _set(engine, "reset", release_reset)
