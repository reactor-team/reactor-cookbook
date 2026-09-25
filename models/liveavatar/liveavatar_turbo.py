"""Three-GPU compiled and five-GPU eager four-step serving profiles."""

from __future__ import annotations

import os


def turbo_enabled() -> bool:
    return os.environ.get("LIVEAVATAR_TURBO", "0") == "1"


def turbo_plan() -> dict:
    """Serving parameters for the active mode.

    ``world_size``    processes/GPUs to spawn (turbo three, released five).
    ``num_gpus_dit``  ranks that run DiT stages (turbo two, packed 2+2, with a
                      third dedicated VAE rank; released four, leaving rank four
                      for the dedicated VAE).
    ``output_rank``   rank that decodes and delivers clips to Runtime.
    ``sampling_steps`` denoising steps -- four in both modes (no reduction).
                      Overridable with ``LIVEAVATAR_STEPS`` for experiments.
    ``compile``       DiT-only ``torch.compile``.
    """
    if turbo_enabled():
        return {
            "world_size": 3,
            "num_gpus_dit": 2,
            "output_rank": 2,
            "sampling_steps": int(os.environ.get("LIVEAVATAR_STEPS", "4")),
            "compile": True,
        }
    return {
        "world_size": 5,
        "num_gpus_dit": 4,
        "output_rank": 4,
        "sampling_steps": int(os.environ.get("LIVEAVATAR_STEPS", "4")),
        "compile": False,
    }


def install_dit_compile() -> None:
    """Compile the DiT only; keep the streaming VAE eager.

    Must run before ``causal_s2v_pipeline_tpp`` is imported so the pinned
    ``@conditional_compile`` decorators pick up the selective wrapper. The
    upstream decorator would otherwise also compile ``stream_decode``, which
    regressed steady-state decode.
    """
    import importlib

    import torch

    os.environ["ENABLE_COMPILE"] = "true"
    inference_utils = importlib.import_module("liveavatar.models.wan.inference_utils")
    eager = {"stream_decode"}

    def selective(func):
        if func.__name__ in eager:
            return func
        return torch.compile(mode=None, backend="inductor", dynamic=None)(func)

    inference_utils.conditional_compile = selective
