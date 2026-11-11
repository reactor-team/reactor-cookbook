"""Opt-in exact attention kernels; no changes to model history or scheduling."""

import logging
import sys


def install_fa4():
    import torch
    from flash_attn.cute import flash_attn_func
    from liveavatar.models.wan.wan_2_2.modules import attention as attention_module

    originals = (attention_module.attention, attention_module.flash_attention)
    counts = {"fa4": 0, "fallback": 0}

    def wrap(original):
        def attention(
            q,
            k,
            v,
            q_lens=None,
            k_lens=None,
            dropout_p=0.0,
            softmax_scale=None,
            q_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=torch.bfloat16,
            **kwargs,
        ):
            # The native streaming workload is B=1; slicing preserves valid keys
            # without allocating cumulative-length metadata for every layer.
            if q.shape[0] != 1 or dropout_p or q.shape[-1] > 256:
                counts["fallback"] += 1
                return original(
                    q,
                    k,
                    v,
                    q_lens=q_lens,
                    k_lens=k_lens,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    q_scale=q_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                    dtype=dtype,
                    **kwargs,
                )
            if q_lens is not None and int(q_lens[0]) != q.shape[1]:
                raise ValueError("Padded queries are not supported by the B=1 FA4 path")
            if k_lens is not None:
                length = int(k_lens[0])
                k, v = k[:, :length], v[:, :length]
            original_dtype = q.dtype
            half = (torch.bfloat16, torch.float16)
            v = v if v.dtype in half else v.to(dtype)
            q, k = q.to(v.dtype), k.to(v.dtype)
            if q_scale is not None:
                q = q * q_scale
            window = tuple(None if item == -1 else item for item in window_size)
            result = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                softmax_scale=softmax_scale,
                window_size=window,
                deterministic=deterministic,
            )
            counts["fa4"] += 1
            # FA4 beta releases return either the output or (output, LSE).
            if isinstance(result, tuple):
                result = result[0]
            return result.to(original_dtype)

        return attention

    replacements = {id(original): wrap(original) for original in originals}
    for name, module in list(sys.modules.items()):
        if not name.startswith("liveavatar.") or module is None:
            continue
        for attr, value in list(vars(module).items()):
            if id(value) in replacements:
                setattr(module, attr, replacements[id(value)])
    logging.getLogger(__name__).info("Installed FlashAttention 4 dense B=1 kernels")
    return counts
