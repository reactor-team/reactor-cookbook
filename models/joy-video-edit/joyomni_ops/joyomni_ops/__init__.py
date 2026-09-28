"""joyomni_ops — minimal self-contained CUDA ops for the JoyOmni V2V DiT pipeline.

Extracted from sgl-kernel; no sglang / sgl_kernel runtime dependency.
Exposes the ops the pipeline uses under a thin wrapper matching the original
sgl_kernel signatures, so call sites only change the import.
"""
import glob
import os
from typing import Optional

import torch

# The extension is a pure TORCH_LIBRARY (no pybind module init), so load the .so
# with torch.ops.load_library to trigger op registration.
_here = os.path.dirname(__file__)
_so = glob.glob(os.path.join(_here, "_C*.so"))
if not _so:
    raise ImportError(f"joyomni_ops C extension not found in {_here}; build it first (pip install .)")
torch.ops.load_library(_so[0])

__all__ = [
    "fused_qk_norm_rope_3d_paired",
    "fused_norm_scale_shift",
    "rmsnorm",
    "fused_qk_norm_rope_3d_paired_to",
    "masked_attn_rescale",
    "sgl_per_token_quant_fp8",
    "per_token_quant_fp8_v2",
    "fused_norm_scale_shift_fp8",
    "gelu_tanh_quant_fp8",
    "fp8_scaled_mm",
    "has_fp8",
]

_ops = torch.ops.joyomni_ops

# fused_norm_scale_shift accepts a single [1, N] scale/shift row and broadcasts it over
# all M rows (no need to materialise an [M, N] copy of the modulation).
NORM_SCALE_SHIFT_BROADCAST_ROWS = True


def fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, q_weight, k_weight, cos, sin):
    """In-place fused RMSNorm(q,k) + 3D RoPE. q/k: [B, seq_len*num_heads, head_dim] bf16."""
    _ops.fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, q_weight, k_weight, cos, sin)


def fused_qk_norm_rope_3d_paired_to(q_src, k_src, q_dst, k_dst, eps, q_weight, k_weight, cos, sin):
    """RMSNorm(q,k) + 3D RoPE from q_src/k_src into q_dst/k_dst ([B, L, H, D] bf16, heads packed,
    any token stride): the same values as copying into the destination and running
    fused_qk_norm_rope_3d_paired there."""
    _ops.fused_qk_norm_rope_3d_paired_to(q_src, k_src, q_dst, k_dst, eps, q_weight, k_weight, cos, sin)


def masked_attn_rescale(o, lse, text_pad):
    """Padded-text attention output o [B, H, S, D] rescaled by 1 / (1 - P * exp(-lse)) per row,
    as [B, S, H, D] bf16 contiguous; P = text_pad [B] padding keys."""
    return _ops.masked_attn_rescale(o, lse, text_pad)


def fused_norm_scale_shift(x, gamma, beta, scale, shift, norm_type, eps=1e-5):
    """Norm(x; gamma, beta) * (1 + scale) + shift. norm_type: 'layer' or 'rms'."""
    nt = 0 if norm_type == "layer" else 1 if norm_type == "rms" else -1
    return _ops.fused_norm_scale_shift(x, gamma, beta, scale, shift, nt, eps)


def rmsnorm(x, weight, eps=1e-6):
    """(x / RMS(x)) * weight. x: [M, N], weight: [N]."""
    return _ops.rmsnorm(x, weight, eps)


def has_fp8() -> bool:
    return hasattr(_ops, "fp8_scaled_mm")


def sgl_per_token_quant_fp8(input, output_q, output_s):
    _ops.sgl_per_token_quant_fp8(input, output_q, output_s)


def per_token_quant_fp8_v2(input, output_q, output_s):
    """sgl_per_token_quant_fp8 with each row read once (held in registers): same outputs."""
    _ops.per_token_quant_fp8_v2(input, output_q, output_s)


def fused_norm_scale_shift_fp8(x, gamma, beta, scale, shift, norm_type, eps=1e-5):
    """sgl_per_token_quant_fp8(fused_norm_scale_shift(...)) in one pass: (q [M, N] fp8, s [M, 1] fp32)."""
    nt = 0 if norm_type == "layer" else 1 if norm_type == "rms" else -1
    return _ops.fused_norm_scale_shift_fp8(x, gamma, beta, scale, shift, nt, eps)


def gelu_tanh_quant_fp8(x):
    """sgl_per_token_quant_fp8(F.gelu(x, approximate="tanh")) in one pass: (q [M, N] fp8, s [M, 1] fp32)."""
    return _ops.gelu_tanh_quant_fp8(x)


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias: Optional[torch.Tensor] = None):
    return _ops.fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
