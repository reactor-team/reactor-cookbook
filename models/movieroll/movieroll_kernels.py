"""Fused inference modulation with native intermediate rounding preserved.

The frame mapping follows upstream modulate_layers under the Tencent Hunyuan
Community License. LayerNorm, projections, RoPE and attention remain upstream.
"""

import torch
import triton
import triton.language as tl

_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def _pointwise(
    X,
    P,
    S,
    Y,
    N: tl.constexpr,
    L: tl.constexpr,
    C: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    FT: tl.constexpr,
    CO: tl.constexpr,
    CL: tl.constexpr,
    NO: tl.constexpr,
    CF: tl.constexpr,
    GATE: tl.constexpr,
    MUL_TYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel, sequence, batch = i % C, (i // C) % L, i // (L * C)
    if CF > 0:
        frame = tl.where(
            sequence < CL, (CO + sequence) // FT, CF + (NO + sequence - CL) // FT
        )
    else:
        frame = (NO + sequence) // FT
    x = tl.load(X + batch * X0 + sequence * X1 + channel * X2, i < N, 0).to(tl.float32)
    p = tl.load(P + batch * P0 + frame * P1 + channel * P2, i < N, 0).to(tl.float32)
    if GATE:
        y = x * p
    else:
        shift = tl.load(S + batch * S0 + frame * S1 + channel * S2, i < N, 0).to(
            tl.float32
        )
        # Native operations round 1+scale and x*(1+scale) separately.
        scale = (1.0 + p).to(P.dtype.element_ty).to(tl.float32)
        y = (x * scale).to(MUL_TYPE).to(tl.float32) + shift
    tl.store(Y + i, y, i < N)


def pointwise(
    x, parameter, shift=None, *, frame_tokens=None, cond=0, sp_size=1, sp_rank=0
):
    """Apply native modulation/gating to strided [B,L,C] and [B,T,C]."""
    gate = shift is None
    if parameter.ndim == 2:
        parameter = parameter[:, None, :]
    if shift is None:
        shift = parameter
    elif shift.ndim == 2:
        shift = shift[:, None, :]
    _, length, channels = x.shape
    frames = parameter.shape[1]
    if frame_tokens is None:
        assert length % frames == 0
        frame_tokens, cond_offset, cond_length, noisy_offset, cond = (
            length // frames,
            0,
            0,
            0,
            0,
        )
    elif cond > 0:
        cond_tokens = cond * frame_tokens
        cond_shard = triton.cdiv(cond_tokens, sp_size)
        noisy_shard = triton.cdiv((frames - cond) * frame_tokens, sp_size)
        cond_offset = sp_rank * cond_shard
        cond_length = max(0, min(cond_shard, cond_tokens - cond_offset))
        noisy_offset = sp_rank * noisy_shard
    else:
        cond_offset = cond_length = 0
        noisy_offset = sp_rank * triton.cdiv(frames * frame_tokens, sp_size)
    multiply_dtype = torch.promote_types(x.dtype, parameter.dtype)
    dtype = multiply_dtype if gate else torch.promote_types(multiply_dtype, shift.dtype)
    output = torch.empty(x.shape, dtype=dtype, device=x.device)
    if x.numel():
        _pointwise[(triton.cdiv(x.numel(), 1024),)](
            x,
            parameter,
            shift,
            output,
            x.numel(),
            length,
            channels,
            *x.stride(),
            *parameter.stride(),
            *shift.stride(),
            frame_tokens,
            cond_offset,
            cond_length,
            noisy_offset,
            cond,
            gate,
            _DTYPES[multiply_dtype],
            1024,
            enable_fp_fusion=False,
        )
    return output


def configure() -> None:
    from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info
    from movieroll.models.transformers import movieroll_transformer
    from movieroll.models.transformers.modules import mlp_layers, modulate_layers

    original_modulate = modulate_layers.modulate
    original_gate = modulate_layers.apply_gate

    def supported(x, parameter):
        return (
            not torch.is_grad_enabled()
            and x.is_cuda
            and x.ndim == 3
            and x.dtype in _DTYPES
            and parameter is not None
            and parameter.device == x.device
            and parameter.dtype in _DTYPES
            and parameter.ndim in (2, 3)
            and parameter.shape[0] == x.shape[0]
            and parameter.shape[-1] == x.shape[-1]
        )

    def invoke(x, parameter, shift, thw, cond):
        size = nccl_info.sp_size if get_sequence_parallel_state() else 1
        rank = nccl_info.rank_within_spgroup if size > 1 else 0
        frame_tokens = thw[1] * thw[2] if thw is not None else None
        return pointwise(
            x,
            parameter,
            shift,
            frame_tokens=frame_tokens,
            cond=cond or 0,
            sp_size=size,
            sp_rank=rank,
        )

    def layout_supported(x, parameter, thw, num_chunks):
        if thw is not None:
            return len(thw) == 3 and parameter.ndim == 3
        return (
            parameter.ndim == 2
            or parameter.shape[1] in (1, x.shape[1])
            or num_chunks == parameter.shape[1]
        )

    def modulate(
        x,
        shift=None,
        scale=None,
        thw=None,
        cond_latents_num_frames=None,
        num_chunks=None,
    ):
        if (
            not supported(x, scale)
            or not supported(x, shift)
            or scale.shape != shift.shape
            or not layout_supported(x, scale, thw, num_chunks)
        ):
            return original_modulate(
                x, shift, scale, thw, cond_latents_num_frames, num_chunks
            )
        return invoke(x, scale, shift, thw, cond_latents_num_frames)

    def apply_gate(
        x, gate=None, thw=None, cond_latents_num_frames=None, num_chunks=None
    ):
        if not supported(x, gate) or not layout_supported(x, gate, thw, num_chunks):
            return original_gate(x, gate, thw, cond_latents_num_frames, num_chunks)
        return invoke(x, gate, None, thw, cond_latents_num_frames)

    movieroll_transformer.modulate = mlp_layers.modulate = modulate
    movieroll_transformer.apply_gate = apply_gate
