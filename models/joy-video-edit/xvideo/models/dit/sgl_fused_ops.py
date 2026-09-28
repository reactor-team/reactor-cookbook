from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch


_AVAILABLE: Optional[bool] = None
_FUSED_NORM_SCALE_SHIFT = None
_FUSED_QK_NORM_ROPE_3D = None
_RMSNORM = None


def available() -> bool:
    return _try_import()


def _try_import() -> bool:
    global _AVAILABLE, _FUSED_NORM_SCALE_SHIFT, _FUSED_QK_NORM_ROPE_3D, _RMSNORM
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
    except Exception:
        _AVAILABLE = False
        return _AVAILABLE
    _RMSNORM = rmsnorm
    _FUSED_NORM_SCALE_SHIFT = fused_norm_scale_shift
    _FUSED_QK_NORM_ROPE_3D = fused_qk_norm_rope_3d_paired
    _AVAILABLE = True
    return _AVAILABLE


# ---------------------------------------------------------------------------
# Triton fused LayerNorm + broadcast modulation kernel
# ---------------------------------------------------------------------------
# Replaces the hot path in fused_layernorm_modulate when scale/shift are
# broadcast tensors of shape (1, D).  Avoids the expand(B,L,D).contiguous()
# call that materialized 59 MB of scale/shift tensors per forward call,
# wasting ~50% of the original kernel's time.
#
# Formula: output = (1 + scale) * LayerNorm(x, gamma, beta, eps) + shift
# This matches joyomni_ops.fused_norm_scale_shift behaviour (verified).
# ---------------------------------------------------------------------------

_TRITON_LNMOD = None   # set on first import of triton


def _build_triton_lnmod():
    global _TRITON_LNMOD
    if _TRITON_LNMOD is not None:
        return _TRITON_LNMOD
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _lnmod_bcast_kernel(
            X_ptr, W_ptr, B_ptr, SCALE_ptr, SHIFT_ptr, OUT_ptr,
            D: tl.constexpr,
            eps: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            """
            One Triton program per row.
            X      : (N, D) bf16
            W, B   : (D,)   bf16  — LayerNorm weight / bias
            SCALE  : (D,)   bf16  — modulation scale (broadcast; original shape (1,D))
            SHIFT  : (D,)   bf16  — modulation shift (broadcast; original shape (1,D))
            OUT    : (N, D) bf16
            formula: (1 + scale) * LN(x, w, b) + shift
            """
            row  = tl.program_id(0)
            cols = tl.arange(0, BLOCK_D)
            mask = cols < D

            x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

            # Welford mean + variance in fp32
            mean = tl.sum(x, axis=0) / D
            x_c  = x - mean
            var  = tl.sum(x_c * x_c, axis=0) / D
            x_n  = x_c * (1.0 / tl.sqrt(var + eps))

            # LayerNorm affine
            gamma = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            beta  = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_ln  = x_n * gamma + beta

            # Modulation: (1 + scale) * LN + shift
            scale = tl.load(SCALE_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(SHIFT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out   = (1.0 + scale) * x_ln + shift

            tl.store(OUT_ptr + row * D + cols, out.to(tl.bfloat16), mask=mask)

        _TRITON_LNMOD = _lnmod_bcast_kernel
    except Exception:
        pass
    return _TRITON_LNMOD


def _triton_lnmod_broadcast(
    x_2d: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scale_1d: torch.Tensor,
    shift_1d: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Fused LayerNorm + broadcast modulation via Triton.
    All inputs bf16.  x_2d: (N, D); scale_1d/shift_1d: (D,) or (1, D).
    """
    kernel = _build_triton_lnmod()
    N, D = x_2d.shape
    out = torch.empty_like(x_2d)
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2
    num_warps = min(max(BLOCK_D // 32, 1), 32)
    kernel[(N,)](
        x_2d.contiguous(),
        gamma.contiguous(),
        beta.contiguous(),
        scale_1d.contiguous().view(-1),
        shift_1d.contiguous().view(-1),
        out,
        D=D,
        eps=eps,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out


@functools.lru_cache(maxsize=None)
def _norm_broadcast_rows() -> bool:
    try:
        import joyomni_ops
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(joyomni_ops, "NORM_SCALE_SHIFT_BROADCAST_ROWS", False))


def fused_layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not _try_import():
        raise RuntimeError(
            "joyomni_ops not available; build it with `pip install ./joyomni_ops`"
        )
    B, L, D = x.shape
    x_2d = x.reshape(-1, D)

    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

    # Fast path: broadcast scale/shift (token-dim == 1) — use Triton kernel
    # to avoid materialising (B, L, D) expand+contiguous tensors (~59 MB each).
    if (
        scale.shape[1] == 1 and L != 1
        and x.dtype == torch.bfloat16
        and weight is not None and bias is not None
        and _build_triton_lnmod() is not None
    ):
        scale_1d = scale.reshape(1, D)
        shift_1d = shift.reshape(1, D)
        out = _triton_lnmod_broadcast(x_2d, weight, bias, scale_1d, shift_1d, eps)
        return out.reshape(B, L, D)

    # Original path: per-token scale/shift or fallback
    if scale.shape[1] == 1 and L != 1 and B == 1 and _norm_broadcast_rows():
        # The kernel broadcasts one row itself: same values per row, no [L, D] copies.
        scale_2d = scale.reshape(1, D).contiguous()
        shift_2d = shift.reshape(1, D).contiguous()
    else:
        if scale.shape[1] == 1 and L != 1:
            scale = scale.expand(B, L, D)
            shift = shift.expand(B, L, D)
        scale_2d = scale.reshape(-1, D).contiguous()
        shift_2d = shift.reshape(-1, D).contiguous()

    out = _FUSED_NORM_SCALE_SHIFT(x_2d, weight, bias, scale_2d, shift_2d, "layer", eps)
    return out.reshape(B, L, D)


def fused_layernorm_modulate_fp8(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """fused_layernorm_modulate followed by per-token FP8 quantization, in one pass:
    (q [B*L, D] fp8, s [B*L, 1] fp32), the same values as quantizing the bf16 result.
    None when this input takes another path of fused_layernorm_modulate (Triton, or no
    broadcast-row support) or the installed joyomni_ops lacks the kernel."""
    if not has_joyomni_op("fused_norm_scale_shift_fp8") or x.dtype != torch.bfloat16:
        return None
    B, L, D = x.shape
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    # Only the joyomni_ops path of fused_layernorm_modulate: broadcast row, batch of one.
    if not (scale.shape[1] == 1 and L != 1 and B == 1 and _norm_broadcast_rows()):
        return None
    if weight is not None and bias is not None and _build_triton_lnmod() is not None:
        return None
    return torch.ops.joyomni_ops.fused_norm_scale_shift_fp8(
        x.reshape(-1, D), weight, bias, scale.reshape(1, D).contiguous(), shift.reshape(1, D).contiguous(), 0, eps
    )


def gelu_tanh_quant_fp8(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """tanh-GELU followed by per-token FP8 quantization, in one pass: (q [M, N] fp8, s [M, 1]),
    the same values as quantizing F.gelu(x, approximate="tanh"); None if unavailable."""
    if not has_joyomni_op("gelu_tanh_quant_fp8") or x.dtype != torch.bfloat16:
        return None
    return torch.ops.joyomni_ops.gelu_tanh_quant_fp8(x.reshape(-1, x.shape[-1]).contiguous())


# Every block of a forward passes the same (cos, sin) tensors, so the kernel's
# half-width bf16 tables are built once per tensor pair instead of once per
# block.  Keyed by object identity (the source tensors are kept alive by the
# entry itself); the tables are never written after they are built.
_ROPE_TABLES: tuple = ()


def _rope_tables_bf16(freqs_cis, D: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    global _ROPE_TABLES
    cos_src, sin_src = freqs_cis
    hit = _ROPE_TABLES
    if hit and hit[0] is cos_src and hit[1] is sin_src and hit[2] == D and hit[3] == device:
        return hit[4], hit[5]
    cos = cos_src.to(device)
    sin = sin_src.to(device)
    while cos.dim() > 2:
        if cos.shape[0] != 1:
            raise RuntimeError(f"freqs_cis cos has non-singleton leading dim: {cos.shape}")
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.shape[-1] == D:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()
    cos_bf16 = cos.to(torch.bfloat16)
    sin_bf16 = sin.to(torch.bfloat16)
    _ROPE_TABLES = (cos_src, sin_src, D, device, cos_bf16, sin_bf16)
    return cos_bf16, sin_bf16


@functools.lru_cache(maxsize=None)
def has_joyomni_op(name: str) -> bool:
    """Whether the installed joyomni_ops build registers `name` (older builds lack newer ops)."""
    if not _try_import():
        return False
    return hasattr(torch.ops.joyomni_ops, name)


def _rope_to_ok(src: torch.Tensor, dst: Optional[torch.Tensor] = None) -> bool:
    # The strided-source kernel needs packed heads, a contiguous head dim and 16-byte rows.
    for t in (src,) if dst is None else (src, dst):
        if (t.dim() != 4 or t.stride(3) != 1 or t.stride(2) != t.shape[3]
                or t.stride(1) % 8 or t.data_ptr() % 16):
            return False
    return True


def fused_qk_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    eps: float = 1e-6,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(q), RMSNorm(k) per head, then 3D RoPE; q/k [B, L, H, D] bf16.

    Returns contiguous results: written in place when q/k are contiguous, else into new tensors,
    or into `out` = (q_dst, k_dst) when given.  Reading a strided q/k (e.g. column slices of a
    fused QKV projection) straight into the destination gives the same values as copying there
    first and normalising in place, without the copy."""
    if not _try_import():
        raise RuntimeError(
            "joyomni_ops not available; build it with `pip install ./joyomni_ops`"
        )
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"fused_qk_norm_rope_3d requires bf16 q/k, got {q.dtype}"
        )

    B, L, H, D = q.shape
    cos_bf16, sin_bf16 = _rope_tables_bf16(freqs_cis, D, q.device)
    qw = q_norm_weight.to(torch.bfloat16)
    kw = k_norm_weight.to(torch.bfloat16)

    to_dst = out is not None or not (q.is_contiguous() and k.is_contiguous())
    if to_dst and has_joyomni_op("fused_qk_norm_rope_3d_paired_to"):
        q_dst, k_dst = out if out is not None else (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
        )
        if (_rope_to_ok(q, q_dst) and _rope_to_ok(k, k_dst) and q.stride(1) == k.stride(1)
                and q_dst.stride(1) == k_dst.stride(1)):
            torch.ops.joyomni_ops.fused_qk_norm_rope_3d_paired_to(
                q, k, q_dst, k_dst, eps, qw, kw, cos_bf16, sin_bf16
            )
            return q_dst, k_dst
    if out is not None:
        out[0].copy_(q)
        out[1].copy_(k)
        q, k = out
    q = q.contiguous()
    k = k.contiguous()
    q_r = q.view(B, L * H, D)
    k_r = k.view(B, L * H, D)

    _FUSED_QK_NORM_ROPE_3D(q_r, k_r, L, H, eps, qw, kw, cos_bf16, sin_bf16)
    return q, k


def masked_attention_rescale(out: torch.Tensor, lse: torch.Tensor, text_pad: torch.Tensor) -> Optional[torch.Tensor]:
    """out [B, H, S, D] bf16 * 1 / (1 - text_pad * exp(-lse)) per row, as [B, S, H, D] bf16, in
    one pass; None when the installed joyomni_ops lacks the kernel (the caller then evaluates
    the same expression with tensor ops)."""
    if (out.dtype != torch.bfloat16 or lse.dtype != torch.float32
            or not has_joyomni_op("masked_attn_rescale") or out.stride(3) != 1):
        return None
    return torch.ops.joyomni_ops.masked_attn_rescale(out, lse, text_pad)


def rmsnorm_qk_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not _try_import() or _RMSNORM is None:
        raise RuntimeError(
            "joyomni_ops rmsnorm not available; build it with `pip install ./joyomni_ops`"
        )
    assert x.dtype == torch.bfloat16, f"rmsnorm_qk_bf16 needs bf16, got {x.dtype}"
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.reshape(-1, D).contiguous()
    w = weight.to(dtype=torch.bfloat16)
    out = _RMSNORM(x_flat, w, eps)
    return out.reshape(orig_shape)


def fused_add_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.addcmul(residual, x, gate.unsqueeze(1))


# ---------------------------------------------------------------------------
# torch.compile compatibility
# ---------------------------------------------------------------------------
# Eagerly run _try_import() at module load so _AVAILABLE is set to True/False
# before torch.compile ever traces any function in this module.
_try_import()

# Prevent torch._dynamo from tracing into the three functions that call
# _try_import() inside their hot path.  On re-compilation (shapes change,
# guards fail), the _AVAILABLE global-state check produces a
# SpeculationLogDivergence error.  These functions all dispatch to custom
# joyomni_ops CUDA kernels that torch.compile cannot fuse further, so making
# them explicit graph-break points is both safe and correct.
fused_layernorm_modulate = torch.compiler.disable(fused_layernorm_modulate)
fused_qk_norm_rope_3d    = torch.compiler.disable(fused_qk_norm_rope_3d)
fused_layernorm_modulate_fp8 = torch.compiler.disable(fused_layernorm_modulate_fp8)
gelu_tanh_quant_fp8      = torch.compiler.disable(gelu_tanh_quant_fp8)
rmsnorm_qk_bf16          = torch.compiler.disable(rmsnorm_qk_bf16)
