from contextlib import contextmanager
import logging
import os
from typing import Callable, Dict, Iterable, List, Tuple, Optional
from einops import rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps



SOURCE_ID_TARGET = 0.0
SOURCE_ID_EDIT_CONDITION = 1.0
SOURCE_ID_EXTRA_REF_IMAGE = 2.0

TIME_FREQ_DIM = 256

NORM_EPS = 1e-6
NUM_MODULATION_CHUNKS = 6

SELF_ATTN_MODE_REF_IMAGE_CACHE = "ref_image_cache"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


_FP8_IMG_ENABLED = _env_on("JOYOMNI_FP8_IMG")
_FP8_TXT_ENABLED = _env_on("JOYOMNI_FP8_TXT")


def _fp8_stream_wanted(stream: str) -> bool:
    if stream == "img":
        return _FP8_IMG_ENABLED
    if stream == "txt":
        return _FP8_TXT_ENABLED
    return False


def _maybe_install_fp8_stream(block, stream: str) -> None:
    """Quantize the attn qkv/proj + mlp up/down Linears of one stream ("img" or "txt")
    to FP8. Idempotent per (block, stream); the original bf16 Linears are left in place."""
    if not _fp8_stream_wanted(stream):
        return
    installed_flag = f"_fp8_{stream}_installed"
    if getattr(block, installed_flag, False):
        return
    from xvideo.models.dit.fp8_linear import Fp8Linear
    setattr(block, f"_fp8_{stream}_attn_qkv",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_qkv")))
    setattr(block, f"_fp8_{stream}_attn_proj",
            Fp8Linear.from_linear(getattr(block, f"{stream}_attn_proj")))
    ff = getattr(block, f"{stream}_mlp")
    up_lin = down_lin = None
    gelu_mod = None
    for m in ff.net:
        if isinstance(m, nn.Linear) and down_lin is None and up_lin is not None:
            down_lin = m
            continue
        if isinstance(m, nn.Linear) and up_lin is None:
            up_lin = m
            continue
        if hasattr(m, "proj") and isinstance(m.proj, nn.Linear) and up_lin is None:
            up_lin = m.proj
            gelu_mod = m
    if up_lin is None or down_lin is None:
        raise RuntimeError(f"could not locate mlp up/down Linears in {ff}")
    setattr(block, f"_fp8_{stream}_mlp_up", Fp8Linear.from_linear(up_lin))
    setattr(block, f"_fp8_{stream}_mlp_down", Fp8Linear.from_linear(down_lin))
    _approx = getattr(gelu_mod, "approximate", "tanh") if gelu_mod is not None else "tanh"
    setattr(block, f"_{stream}_mlp_act",
            lambda x, _a=_approx: torch.nn.functional.gelu(x, approximate=_a))
    setattr(block, f"_{stream}_mlp_act_approx", _approx)
    setattr(block, installed_flag, True)


def _fp8_stream_enabled(block, stream: str) -> bool:
    return bool(getattr(block, f"_fp8_{stream}_installed", False))



from xvideo.models.dit import sgl_fused_ops as _sgl_fused

from xvideo.models.dit.rope import apply_rotary_emb, get_1d_rotary_pos_embed


class ModulateWan(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size,
                        dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


logger = logging.getLogger(__name__)


def _clone_kv_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone()


# Attention runs on cuDNN only, called through the aten op directly, so the
# choice binds this call alone.  The `sdpa_kernel(...)` context sets global
# backend flags, which would also bind any VAE attention run inside it
# (head_dim > 128, which cuDNN rejects).  An unsupported shape raises
# instead of silently switching to a slower or numerically different kernel.
_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention


# Every (q_len, kv_len, masked) the attention has run at.  cuDNN builds an execution plan the
# first time it meets a shape (~0.7 s on B200), so a serving process warms a known shape set up
# front; once `seal_attention_shapes()` is called, any shape outside that set is reported.
_attn_shapes: set[tuple[int, int, bool]] = set()
_attn_shapes_sealed = False


def attention_shapes() -> list[tuple[int, int, bool]]:
    return sorted(_attn_shapes)


def seal_attention_shapes() -> None:
    global _attn_shapes_sealed
    _attn_shapes_sealed = True


def _note_attention_shape(q_len: int, kv_len: int, masked: bool) -> None:
    shape = (q_len, kv_len, masked)
    if shape in _attn_shapes:
        return
    _attn_shapes.add(shape)
    if _attn_shapes_sealed:
        print(f"#####[ATTN] shape not covered by warm-up: q_len={q_len} kv_len={kv_len} "
              f"text_masked={masked} (first call builds a cuDNN plan)", flush=True)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    text_keep: Optional[torch.Tensor] = None,
    text_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """cuDNN attention over [B, S, H, D] tensors.

    With `text_keep` ([B, T, 1, 1], 1 = real token, 0 = padding) the last T keys are the text
    tokens, padded to a fixed length, and padding keys are masked out of the softmax.  The mask
    is applied without a bias tensor (which costs cuDNN ~2.5x at these shapes): padding keys and
    values are zeroed, so each contributes exp(0) = 1 to the softmax denominator and nothing to
    the numerator, and the output is rescaled by Z / (Z - P), Z = exp(logsumexp), P = the number
    of padding keys (`text_pad`, [B]), computed as 1 / (1 - P / Z) so that a large Z cannot
    overflow.  Both come from device tensors, so one captured graph
    serves every prompt length.  Real keys sum to far more than P at these sequence lengths, so
    the rescale is well conditioned.  Key/value rows are zeroed in place: they are this call's
    freshly computed text rows, never a cached entry.
    """
    original_dtype = query.dtype
    if query.dtype not in (torch.float16, torch.bfloat16):
        query, key, value = (t.to(torch.bfloat16) for t in (query, key, value))
    _note_attention_shape(query.shape[1], key.shape[1], text_keep is not None)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if text_keep is None:
        out = _cudnn_attention(q, k, v, None, False)[0]
        return out.transpose(1, 2).to(original_dtype)
    n_txt = text_keep.shape[1]
    keep = text_keep.to(key.dtype)
    key[:, -n_txt:].mul_(keep)
    value[:, -n_txt:].mul_(keep)
    out, lse = _cudnn_attention(q, k, v, None, True)[:2]
    if original_dtype == torch.bfloat16:
        # One fused pass over `out`, the same float operations as the expression below.
        fused = _sgl_fused.masked_attention_rescale(out, lse, text_pad)
        if fused is not None:
            return fused
    b, h, s_q = q.shape[0], q.shape[1], q.shape[2]
    # Z / (Z - P) written as 1 / (1 - P * exp(-lse)): exp(lse) itself overflows float32 once a
    # row's logits pass ~88, which this DiT's attention does, while exp(-lse) only underflows to 0
    # (factor 1, the right limit when the real keys dominate).
    p_exp = text_pad.to(torch.float32).view(b, 1, 1, 1) * torch.exp(-lse.reshape(b, h, s_q, 1).to(torch.float32))
    scale = (1.0 / (1.0 - p_exp)).transpose(1, 2)
    return (out.transpose(1, 2).to(torch.float32) * scale).to(original_dtype)


def attention_backend() -> str:
    return "cudnn"


def _concat_kv_entries(
    entries: Iterable[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    keys = []
    values = []
    pre_rope_offset = 0

    for entry in entries:
        if entry is None:
            continue
        key = entry.get("key")
        value = entry.get("value")
        if key is None or value is None:
            continue

        key = key.to(device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)

        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            seg_len = key.shape[1]
            cos_seg = cos_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            sin_seg = sin_all[..., pre_rope_offset: pre_rope_offset + seg_len, :]
            key = apply_rotary_emb(key, (cos_seg, sin_seg))
            pre_rope_offset += seg_len

        keys.append(key)
        values.append(value)

    if not keys:
        return None, None
    if len(keys) == 1:
        return keys[0], values[0]
    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = NORM_EPS,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def _norm_modulate(x, shift, scale, norm: nn.LayerNorm) -> torch.Tensor:
    return _sgl_fused.fused_layernorm_modulate(
        x, shift=shift, scale=scale,
        weight=norm.weight if norm.elementwise_affine else None,
        bias=norm.bias if norm.elementwise_affine else None,
        eps=norm.eps,
    )


def _norm_modulate_linear(x, shift, scale, norm: nn.LayerNorm, linear, fp8: bool) -> torch.Tensor:
    """linear(LayerNorm-modulate(x)).  With an FP8 linear the norm writes the per-token
    quantized input directly (same values as quantizing its bf16 output)."""
    if fp8:
        qs = _sgl_fused.fused_layernorm_modulate_fp8(
            x, shift=shift, scale=scale,
            weight=norm.weight if norm.elementwise_affine else None,
            bias=norm.bias if norm.elementwise_affine else None,
            eps=norm.eps,
        )
        if qs is not None:
            return linear.forward_quantized(qs[0], qs[1], x.shape[:-1])
    return linear(_norm_modulate(x, shift, scale, norm))


def _rms_norm_into(norm: "RMSNorm", x: torch.Tensor, out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """RMSNorm.forward(x).to(dtype), its last multiply written into `out`."""
    if not hasattr(norm, "weight") or out.dtype != dtype:
        out.copy_(norm(x).to(dtype))
        return out
    torch.mul(norm._norm(x.float()).type_as(x), norm.weight, out=out)
    return out


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu-approximate",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

        self.txt_mod = ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=NORM_EPS, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn=mlp_act_type)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        kv_cache_reader: Optional[Callable[[Optional[int]], Iterable[Dict[str, torch.Tensor]]]] = None,
        kv_cache_writer: Optional[Callable[[Optional[int], torch.Tensor, torch.Tensor], None]] = None,
        kv_cache_assembler: Optional[Callable[..., Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        layer_idx: Optional[int] = None,
        skip_text_stream: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: Optional[tuple] = None,
        text_keep: Optional[torch.Tensor] = None,
        text_pad: Optional[torch.Tensor] = None,
        kv_only: bool = False,
        keep_rows: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`kv_only`: only the KV cache write of this block is wanted (the last block of a
        forward whose output is discarded); the block returns right after the write and its
        (img, txt) return values are the unchanged inputs.

        `keep_rows`: only the first keep_rows image rows of the output are wanted (the last
        block of a denoise forward); the block returns those rows and the text states as given."""
        _maybe_install_fp8_stream(self, "img")
        _fp8_on = _fp8_stream_enabled(self, "img")
        if not skip_text_stream:
            _maybe_install_fp8_stream(self, "txt")
        _fp8_txt_on = (not skip_text_stream) and _fp8_stream_enabled(self, "txt")
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        if not skip_text_stream:
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec)

        img_qkv = _norm_modulate_linear(
            img, img_mod1_shift, img_mod1_scale, self.img_norm1,
            self._fp8_img_attn_qkv if _fp8_on else self.img_attn_qkv, _fp8_on,
        )
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # The pre-RoPE key is only consumed by the cache writer (store forwards).
        if kv_cache_pre_rope and kv_cache_writer is not None:
            img_k_for_cache = _sgl_fused.rmsnorm_qk_bf16(
                img_k, self.img_attn_k_norm.weight, eps=self.img_attn_k_norm.eps
            )
            if kv_only:
                kv_cache_writer(layer_idx, img_k_for_cache, img_v)
                return img, txt
        elif kv_only:
            raise ValueError("kv_only needs a pre-RoPE KV cache writer")
        # With a KV assembler (denoise forwards reading the window cache) the attention
        # operands are written straight into their final buffers: q into a joint
        # [image | text] buffer, k/v into the tail of the [cached | current] window
        # buffer whose cached prefix is filled once per window.  Same values, same
        # layout as concatenating and copying, without the intermediate tensors.
        in_place = kv_cache_assembler is not None and img.shape[0] == 1
        if in_place:
            L_img = img_q.shape[1]
            L_txt = 0 if skip_text_stream else txt.shape[1]
            full_k, full_v, n_cached = kv_cache_assembler(
                layer_idx,
                device=img_q.device,
                dtype=img_q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
                reserve_current=(L_img + L_txt, img_q.shape[2], img_q.shape[3]),
            )
            q_joint = torch.empty(
                (1, L_img + L_txt, img_q.shape[2], img_q.shape[3]), device=img_q.device, dtype=img_q.dtype
            )
            q_img = q_joint[:, :L_img]
            k_img = full_k[:, n_cached:n_cached + L_img]
            full_v[:, n_cached:n_cached + L_img].copy_(img_v)
        img_q, img_k = _sgl_fused.fused_qk_norm_rope_3d(
            img_q, img_k,
            q_norm_weight=self.img_attn_q_norm.weight,
            k_norm_weight=self.img_attn_k_norm.weight,
            freqs_cis=vis_freqs_cis,
            eps=self.img_attn_q_norm.eps,
            out=(q_img, k_img) if in_place else None,
        )
        if not kv_cache_pre_rope:
            img_k_for_cache = img_k

        if not skip_text_stream:
            txt_qkv = _norm_modulate_linear(
                txt, txt_mod1_shift, txt_mod1_scale, self.txt_norm1,
                self._fp8_txt_attn_qkv if _fp8_txt_on else self.txt_attn_qkv, _fp8_txt_on,
            )
            txt_q, txt_k, txt_v = rearrange(
                txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )
            if in_place:
                # The norms' last multiply writes straight into the attention buffers.
                txt_q = _rms_norm_into(self.txt_attn_q_norm, txt_q, q_joint[:, L_img:], txt_v.dtype)
                txt_k = _rms_norm_into(self.txt_attn_k_norm, txt_k, full_k[:, n_cached + L_img:], txt_v.dtype)
            else:
                txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
                txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        if in_place:
            if not skip_text_stream:
                full_v[:, n_cached + L_img:].copy_(txt_v)
            q, k, v = q_joint, full_k, full_v
        elif skip_text_stream:
            q = img_q
            k = img_k
            v = img_v
        else:
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)

        if kv_cache_writer is not None:
            kv_cache_writer(layer_idx, img_k_for_cache, img_v)

        if in_place:
            pass
        elif kv_cache_assembler is not None:
            # Pass current k/v so the assembler can build the full buffer (cached +
            # current) inside its existing torch.compiler.disable context.
            k, v = kv_cache_assembler(
                layer_idx,
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
                current_k=k,
                current_v=v,
            )
        elif kv_cache_reader is not None:
            cached_key, cached_value = _concat_kv_entries(
                kv_cache_reader(layer_idx),
                device=q.device,
                dtype=q.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
            )
            if cached_key is not None:
                k = torch.cat([cached_key, k], dim=1)
                v = torch.cat([cached_value, v], dim=1)

        if keep_rows is not None:
            # Only the first keep_rows image rows of this block's output are read: every op after
            # the K/V is row-wise, so the other query rows and the text stream are not computed.
            q = q[:, :keep_rows]
            img = img[:, :keep_rows]
        if skip_text_stream:
            attn = _attention(q, k, v)
        else:
            attn = _attention(q, k, v, text_keep=text_keep, text_pad=text_pad)
        attn = attn.flatten(2, 3)
        if skip_text_stream or keep_rows is not None:
            img_attn = attn[:, : img.shape[1]]
        else:
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        def _img_proj_call(x):
            return self._fp8_img_attn_proj(x) if _fp8_on else self.img_attn_proj(x)

        def _txt_proj_call(x):
            return self._fp8_txt_attn_proj(x) if _fp8_txt_on else self.txt_attn_proj(x)

        img = _sgl_fused.fused_add_gate(
            img, _img_proj_call(img_attn),
            img_mod1_gate,
        )
        img = _sgl_fused.fused_add_gate(
            img, self._mlp(img, img_mod2_shift, img_mod2_scale, "img", _fp8_on),
            img_mod2_gate,
        )

        if not skip_text_stream and keep_rows is None:
            txt = _sgl_fused.fused_add_gate(
                txt, _txt_proj_call(txt_attn),
                txt_mod1_gate,
            )
            txt = _sgl_fused.fused_add_gate(
                txt, self._mlp(txt, txt_mod2_shift, txt_mod2_scale, "txt", _fp8_txt_on),
                txt_mod2_gate,
            )

        return img, txt

    def _mlp(self, x, shift, scale, stream: str, fp8_on: bool) -> torch.Tensor:
        """MLP of one stream on its modulated norm2 input: up, GELU, down."""
        norm = getattr(self, f"{stream}_norm2")
        if not fp8_on:
            return getattr(self, f"{stream}_mlp")(_norm_modulate(x, shift, scale, norm))
        h = _norm_modulate_linear(x, shift, scale, norm, getattr(self, f"_fp8_{stream}_mlp_up"), True)
        down = getattr(self, f"_fp8_{stream}_mlp_down")
        if getattr(self, f"_{stream}_mlp_act_approx", None) == "tanh":
            # GELU and the down projection's per-token quantization in one pass.
            hq = _sgl_fused.gelu_tanh_quant_fp8(h)
            if hq is not None:
                return down.forward_quantized(hq[0], hq[1], h.shape[:-1])
        return down(getattr(self, f"_{stream}_mlp_act")(h))


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        _ = image_embed_dim, pos_embed_seq_len

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,   
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        project_text: bool = True,
    ):
        """`project_text=False`: the text projection is not computed and the text states are
        returned as given (for a forward that never reads the text stream)."""
        timestep_shape = timestep.shape
        timestep_is_sequence = timestep.ndim > 1
        if timestep_is_sequence:
            timestep = timestep.flatten()
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        if timestep_is_sequence:
            timestep_proj = timestep_proj.reshape(*timestep_shape, timestep_proj.shape[-1])

        if project_text:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return timestep_proj, encoder_hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        theta: int = 256,
        chunk_size: Optional[int] = None,
        local_window_size: int = 4,
        global_sink_chunk: bool = True,
        causal: bool = False,
        use_inference_kv_cache: bool = False,
        source_id_rope_dim: int = 128,
        source_id_rope_theta: float = 256.0,
    ):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive when provided, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        if causal and chunk_size is None:
            raise ValueError("`chunk_size` must be provided when `causal=True`.")
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.theta = theta
        self.chunk_size = chunk_size
        self.local_window_size = local_window_size
        self.global_sink_chunk = global_sink_chunk
        self.causal = causal
        self._initial_use_inference_kv_cache = use_inference_kv_cache

        self.source_id_rope_dim = int(source_id_rope_dim)
        self.source_id_rope_theta = float(source_id_rope_theta)

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=TIME_FREQ_DIM,
            time_proj_dim=hidden_size * NUM_MODULATION_CHUNKS,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=NORM_EPS
        )
        self.proj_out = nn.Linear(
            hidden_size, self.out_channels * math.prod(patch_size),
            **factory_kwargs)

        self._ensure_kv_cache_state()
        self._use_inference_kv_cache = use_inference_kv_cache

    def _ensure_kv_cache_state(self) -> None:
        if not hasattr(self, "_inference_kv_cache"):
            self._inference_kv_cache = {"cond": {}, "uncond": {}}
        if not hasattr(self, "_use_inference_kv_cache"):
            self._use_inference_kv_cache = bool(getattr(self, "_initial_use_inference_kv_cache", False))
        if not hasattr(self, "_kv_cache_mode"):
            self._kv_cache_mode = None
        if not hasattr(self, "_kv_cache_scope"):
            self._kv_cache_scope = None
        if not hasattr(self, "_kv_cache_chunk_id"):
            self._kv_cache_chunk_id = None
        if not hasattr(self, "_kv_cache_selected_chunk_ids"):
            self._kv_cache_selected_chunk_ids = None
        if not hasattr(self, "_kv_cache_pre_rope"):
            self._kv_cache_pre_rope = False
        if not hasattr(self, "_kv_cache_generation"):
            self._kv_cache_generation = 0
        if not hasattr(self, "_kv_assembly_version"):
            self._kv_assembly_version = None
        if not hasattr(self, "_kv_cumulative"):
            self._kv_cumulative = {}
        # Wrap KV cache callables with torch.compiler.disable so dynamo treats
        # them as opaque black boxes — prevents guards on dict sizes / object state
        if not hasattr(self, "_cached_kv_reader"):
            self._cached_kv_reader = torch.compiler.disable(self._read_layer_kv_cache)
            self._cached_kv_writer = torch.compiler.disable(self._write_layer_kv_cache)
            self._cached_kv_assembler = torch.compiler.disable(self._assemble_layer_kv_cache_cached)
        if not hasattr(self, "_kv_assembly_cache"):
            self._kv_assembly_cache = {}
            self._kv_assembly_epoch = self.__dict__.get("_kv_assembly_epoch", 0) + 1
        if not hasattr(self, "_kv_assembly_freqs"):
            self._kv_assembly_freqs = None
        if not hasattr(self, "_kv_prealloc_key"):
            self._kv_prealloc_key = {}
            self._kv_prealloc_val = {}
        # Full-KV buffer (assembled cache + current chunk) — eliminates torch.cat
        # at the attention call site without adding a new graph break.
        if not hasattr(self, "_kv_full_key"):
            self._kv_full_key = {}
            self._kv_full_val = {}

    def reset_inference_kv_cache(self) -> None:
        self._ensure_kv_cache_state()
        self._inference_kv_cache = {"cond": {}, "uncond": {}}
        self._use_inference_kv_cache = bool(getattr(self.config, "use_inference_kv_cache", False))
        self._kv_cache_mode = None
        self._kv_cache_scope = None
        self._kv_cache_chunk_id = None
        self._kv_cache_selected_chunk_ids = None
        self._kv_cache_pre_rope = False
        self._kv_cache_generation = 0
        self._kv_cumulative = {}
        self._kv_assembly_version = None
        self._kv_assembly_cache = {}
        self._kv_assembly_epoch = self.__dict__.get("_kv_assembly_epoch", 0) + 1
        self._kv_assembly_freqs = None
        self._kv_prealloc_key = {}
        self._kv_prealloc_val = {}
        self._kv_full_key = {}
        self._kv_full_val = {}

    def configure_inference_kv_cache(
        self,
        *,
        scope: Optional[str],
        mode: Optional[str],
        chunk_id: Optional[int] = None,
        selected_chunk_ids: Optional[list[int]] = None,
        pre_rope: bool = False,
    ) -> None:
        self._ensure_kv_cache_state()
        self._kv_cache_scope = scope
        self._kv_cache_mode = mode
        self._kv_cache_chunk_id = chunk_id
        self._kv_cache_selected_chunk_ids = list(selected_chunk_ids) if selected_chunk_ids is not None else None
        self._use_inference_kv_cache = mode in {"reuse", "store", "reuse_store"}
        self._kv_cache_pre_rope = bool(pre_rope)

    @contextmanager
    def cache_context(self, scope: Optional[str]):
        self._ensure_kv_cache_state()
        previous_scope = self._kv_cache_scope
        self._kv_cache_scope = scope
        try:
            yield self
        finally:
            self._kv_cache_scope = previous_scope

    def _get_cache_scope_store(self) -> Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]]:
        self._ensure_kv_cache_state()
        if self._kv_cache_scope is None:
            return None
        return self._inference_kv_cache.setdefault(self._kv_cache_scope, {})

    def _read_layer_kv_cache(self, layer_idx: Optional[int]) -> list[Dict[str, torch.Tensor]]:
        # Cumulative fast path removed: it bypassed _kv_cache_selected_chunk_ids
        # (local window=3+sink=4 chunks) → was feeding ALL 8 chunks to attention.
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None:
            return []
        selected_chunk_ids = self._kv_cache_selected_chunk_ids or []
        layer_entries = []
        for selected_chunk_id in selected_chunk_ids:
            chunk_store = scope_store.get(selected_chunk_id)
            if chunk_store is None:
                continue
            entry = chunk_store.get(layer_idx)
            if entry is not None:
                layer_entries.append(entry)
        return layer_entries

    def _assemble_layer_kv_cache_cached(
        self,
        layer_idx: Optional[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        cached_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
        current_k: Optional[torch.Tensor] = None,
        current_v: Optional[torch.Tensor] = None,
        reserve_current: Optional[Tuple[int, int, int]] = None,
    ):
        # Fast path: already assembled for this layer in this chunk's step 1.
        # The assembly stamp no longer includes kv_cache_generation so this
        # cache survives across multiple denoising steps within the same chunk.
        if layer_idx in self._kv_assembly_cache:
            cached_k, cached_v = self._kv_assembly_cache[layer_idx]
        else:
            entries = self._read_layer_kv_cache(layer_idx)

            if not entries:
                self._kv_assembly_cache[layer_idx] = (None, None)
                cached_k = cached_v = None
            elif len(entries) == 1:
                # Single-entry: no concat, just dtype/device convert + optional RoPE
                entry = entries[0]
                cached_k = entry["key"].to(device=device, dtype=dtype)
                cached_v = entry["value"].to(device=device, dtype=dtype)
                if entry.get("pre_rope", False) and cached_freqs_cis is not None:
                    cos_all, sin_all = cached_freqs_cis
                    cached_k = apply_rotary_emb(cached_k, (cos_all[..., :cached_k.shape[1], :], sin_all[..., :cached_k.shape[1], :]))
                self._kv_assembly_cache[layer_idx] = (cached_k, cached_v)
            else:
                # Multi-entry fast path: fill a pre-allocated buffer with copy_ instead
                # of torch.cat so we avoid per-chunk tensor allocation overhead (~2.85×
                # faster than torch.cat for the 4-chunk steady-state window).
                total_tokens = sum(e["key"].shape[1] for e in entries)
                sample_key = entries[0]["key"]
                B, _, H_kv, D_kv = sample_key.shape

                prealloc_key = getattr(self, "_kv_prealloc_key", None)
                prealloc_val = getattr(self, "_kv_prealloc_val", None)
                if prealloc_key is None:
                    self._kv_prealloc_key = {}
                    self._kv_prealloc_val = {}
                    prealloc_key = self._kv_prealloc_key
                    prealloc_val = self._kv_prealloc_val

                buf_shape = (B, total_tokens, H_kv, D_kv)
                if (
                    layer_idx not in prealloc_key
                    or prealloc_key[layer_idx].shape != buf_shape
                    or prealloc_key[layer_idx].dtype != dtype
                    or prealloc_key[layer_idx].device.type != device.type
                ):
                    prealloc_key[layer_idx] = torch.empty(buf_shape, dtype=dtype, device=device)
                    prealloc_val[layer_idx] = torch.empty(buf_shape, dtype=dtype, device=device)

                buf_k = prealloc_key[layer_idx]
                buf_v = prealloc_val[layer_idx]

                pre_rope_offset = 0
                offset = 0
                for entry in entries:
                    k_e = entry["key"]
                    v_e = entry["value"]
                    n = k_e.shape[1]

                    if k_e.device.type != device.type or k_e.dtype != dtype:
                        k_e = k_e.to(device=device, dtype=dtype)
                    if entry.get("pre_rope", False) and cached_freqs_cis is not None:
                        cos_all, sin_all = cached_freqs_cis
                        cos_seg = cos_all[..., pre_rope_offset:pre_rope_offset + n, :]
                        sin_seg = sin_all[..., pre_rope_offset:pre_rope_offset + n, :]
                        k_e = apply_rotary_emb(k_e, (cos_seg, sin_seg))
                        pre_rope_offset += n

                    buf_k[:, offset:offset + n].copy_(k_e)

                    if v_e.device.type != device.type or v_e.dtype != dtype:
                        v_e = v_e.to(device=device, dtype=dtype)
                    buf_v[:, offset:offset + n].copy_(v_e)

                    offset += n

                self._kv_assembly_cache[layer_idx] = (buf_k, buf_v)
                cached_k, cached_v = buf_k, buf_v

        if reserve_current is not None:
            return self._window_buffers(layer_idx, cached_k, cached_v, reserve_current, device, dtype)

        # Old API: no current_k → return just the assembled cache.
        if current_k is None:
            return cached_k, cached_v

        # New API: current_k provided → eliminate the torch.cat([cached, current])
        # at the attention call site by assembling the full buffer here (inside the
        # existing torch.compiler.disable context of _cached_kv_assembler, so no
        # new graph break is introduced).
        if cached_k is None:
            return current_k, current_v

        n_c = cached_k.shape[1]
        full_shape = (current_k.shape[0], n_c + current_k.shape[1], current_k.shape[2], current_k.shape[3])
        kv_full_key = getattr(self, "_kv_full_key", None)
        if kv_full_key is None:
            self._kv_full_key = {}
            self._kv_full_val = {}
            kv_full_key = self._kv_full_key
        kv_full_val = self._kv_full_val
        if (
            layer_idx not in kv_full_key
            or kv_full_key[layer_idx].shape != full_shape
            or kv_full_key[layer_idx].dtype != dtype
        ):
            kv_full_key[layer_idx] = torch.empty(full_shape, dtype=dtype, device=device)
            kv_full_val[layer_idx] = torch.empty(full_shape, dtype=dtype, device=device)
        full_buf_k = kv_full_key[layer_idx]
        full_buf_v = kv_full_val[layer_idx]
        full_buf_k[:, :n_c].copy_(cached_k)
        full_buf_k[:, n_c:].copy_(current_k)
        full_buf_v[:, :n_c].copy_(cached_v)
        full_buf_v[:, n_c:].copy_(current_v)
        return full_buf_k, full_buf_v

    def _window_buffers(self, layer_idx, cached_k, cached_v, current_shape, device, dtype):
        """[cached | current] key/value buffers for one layer, cached prefix filled.

        The prefix is copied only when the assembled window changed (a new assembly
        epoch); denoising steps within a chunk reuse it.  The current tail is left for
        the caller to write.  Returns (full_k, full_v, n_cached).
        """
        n_cur, heads, head_dim = current_shape
        n_c = 0 if cached_k is None else int(cached_k.shape[1])
        full_shape = (1, n_c + n_cur, heads, head_dim)
        kv_full_key, kv_full_val = self._kv_full_key, self._kv_full_val
        filled = self.__dict__.setdefault("_kv_full_epoch", {})
        buf = kv_full_key.get(layer_idx)
        if buf is None or tuple(buf.shape) != full_shape or buf.dtype != dtype or buf.device != torch.device(device):
            kv_full_key[layer_idx] = torch.empty(full_shape, dtype=dtype, device=device)
            kv_full_val[layer_idx] = torch.empty(full_shape, dtype=dtype, device=device)
            filled.pop(layer_idx, None)
        full_k, full_v = kv_full_key[layer_idx], kv_full_val[layer_idx]
        epoch = self.__dict__.get("_kv_assembly_epoch", 0)
        if n_c and filled.get(layer_idx) != epoch:
            full_k[:, :n_c].copy_(cached_k)
            full_v[:, :n_c].copy_(cached_v)
            filled[layer_idx] = epoch
        return full_k, full_v, n_c

    def _write_layer_kv_cache(
        self,
        layer_idx: Optional[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        scope_store = self._get_cache_scope_store()
        if scope_store is None or layer_idx is None or self._kv_cache_chunk_id is None:
            return
        chunk_store = scope_store.setdefault(self._kv_cache_chunk_id, {})
        chunk_store[layer_idx] = {
            "key": _clone_kv_tensor(key),
            "value": _clone_kv_tensor(value),
            "pre_rope": bool(self._kv_cache_pre_rope),
        }
        self._kv_cache_generation += 1
    def evict_kv_cache_chunks(self, chunk_ids_to_keep: set[int]) -> None:
        self._ensure_kv_cache_state()
        for scope in ("cond", "uncond"):
            scope_store = self._inference_kv_cache.get(scope)
            if scope_store is None:
                continue
            evict_ids = [cid for cid in scope_store if cid not in chunk_ids_to_keep]
            for cid in evict_ids:
                del scope_store[cid]
            if evict_ids:
                self._kv_cache_generation += 1

    def get_rotary_pos_embed_from_ids(
        self,
        *,
        frame_ids: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        post_patch_height, post_patch_width = spatial_shape
        device = frame_ids.device
        temporal_positions = frame_ids.to(dtype=torch.float32)
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_positions.numel() % spatial_tokens_per_frame != 0:
            raise ValueError(
                f"`frame_ids` length {temporal_positions.numel()} is not divisible by spatial token count {spatial_tokens_per_frame}."
            )

        h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
        w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
        h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
        h_positions = h_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)
        w_positions = w_grid.reshape(-1).repeat(temporal_positions.numel() // spatial_tokens_per_frame)

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal to head_dim of attention layer")

        cos_list = []
        sin_list = []
        for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions)):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                positions,
                theta=self.theta,
            )
            cos_list.append(cos)
            sin_list.append(sin)
        vis_freqs = (torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1))

        return vis_freqs

    def generate_source_id_rope(
        self,
        source_id: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        role_dim = max(0, min(int(self.source_id_rope_dim), int(head_dim)))

        half_head = head_dim // 2
        cos_half = torch.ones(*source_id.shape, half_head, device=device, dtype=torch.float32)
        sin_half = torch.zeros(*source_id.shape, half_head, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (
            self.source_id_rope_theta **
            (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
        )

        angles = source_id.unsqueeze(-1) * inv_freq
        cos_half[..., : role_dim // 2] = torch.cos(angles)
        sin_half[..., : role_dim // 2] = torch.sin(angles)
        return (
            cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
            sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def _get_patch_shape(latent: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        _, _, num_frames, height, width = latent.shape
        return (
            num_frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )

    @staticmethod
    def _get_token_frame_ids(
        post_patch_shape: Tuple[int, int, int],
        device: torch.device,
        temporal_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_frames, post_patch_height, post_patch_width = post_patch_shape
        spatial_tokens_per_frame = post_patch_height * post_patch_width
        if temporal_ids is None:
            frame_ids = torch.arange(num_frames, device=device, dtype=torch.long)
        else:
            frame_ids = torch.as_tensor(temporal_ids, device=device, dtype=torch.long)
            if frame_ids.ndim != 1 or frame_ids.numel() != num_frames:
                raise ValueError(
                    f"`temporal_ids` must be 1D with length {num_frames}, got {tuple(frame_ids.shape)}."
                )
        return frame_ids.repeat_interleave(spatial_tokens_per_frame)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
        ref_video_latent: Optional[torch.Tensor] = None,
        current_temporal_ids: Optional[torch.Tensor] = None,
        cached_temporal_ids: Optional[torch.Tensor] = None,
        kv_cache_mode: Optional[str] = None,
        kv_cache_scope: Optional[str] = None,
        kv_cache_chunk_id: Optional[int] = None,
        kv_cache_selected_chunk_ids: Optional[list[int]] = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: Optional[str] = None,
        skip_text_stream: bool = False,
        mask_text_padding: bool = False,
        return_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`mask_text_padding`: the text tokens are padded to a fixed length and
        `encoder_hidden_states_mask` marks the real ones; attention masks the rest.

        `return_output=False`: the caller wants only the KV cache this forward writes (a
        pre-RoPE store).  The last block then stops after its cache write, the output head
        is skipped, and (None, None) is returned."""

        self._ensure_kv_cache_state()
        self.configure_inference_kv_cache(
            scope=kv_cache_scope,
            mode=kv_cache_mode,
            chunk_id=kv_cache_chunk_id,
            selected_chunk_ids=kv_cache_selected_chunk_ids,
            pre_rope=kv_cache_pre_rope,
        )

        batch_size = hidden_states.shape[0]
        patch_size = tuple(self.patch_size)
        current_patch_shape = self._get_patch_shape(hidden_states, patch_size)
        current_seq_len = math.prod(current_patch_shape)
        device = hidden_states.device

        encoder_hidden_states_mask = encoder_hidden_states_mask.to(
            device=encoder_hidden_states.device,
            dtype=torch.bool,
        )

        hidden_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temporal_ids = None
        if current_temporal_ids is not None:
            current_temporal_ids = torch.as_tensor(current_temporal_ids, device=device, dtype=torch.long)
            assert current_temporal_ids.shape == (batch_size, current_patch_shape[0]), (
                f"`current_temporal_ids` must have shape {(batch_size, current_patch_shape[0])}, "
                f"got {tuple(current_temporal_ids.shape)}."
            )
            temporal_ids = current_temporal_ids[0]
        if self_attn_input_mode == SELF_ATTN_MODE_REF_IMAGE_CACHE:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_EXTRA_REF_IMAGE, device=device, dtype=torch.float32)
        else:
            current_source_id = torch.full((current_seq_len,), SOURCE_ID_TARGET, device=device, dtype=torch.float32)
        current_frame_ids = self._get_token_frame_ids(current_patch_shape, device, temporal_ids=temporal_ids)
        current_rotary = self.get_rotary_pos_embed_from_ids(
            frame_ids=current_frame_ids,
            spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
        )

        latent_segments = [hidden_tokens]
        rotary_segments = [current_rotary]
        source_id_segments = [current_source_id]

        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"Ref video latent batch size {ref_video_latent.shape[0]} does not match hidden states batch size {batch_size}."
                )
            ref_video_patch_shape = self._get_patch_shape(ref_video_latent, patch_size)
            if ref_video_patch_shape[1:] != current_patch_shape[1:]:
                raise ValueError(
                    "Ref video latent spatial patch shape must match noisy latent spatial patch shape: "
                    f"{ref_video_patch_shape[1:]} != {current_patch_shape[1:]}."
                )
            ref_video_tokens = self.img_in(ref_video_latent).flatten(2).transpose(1, 2)
            video_frame_ids = self._get_token_frame_ids(ref_video_patch_shape, device, temporal_ids=temporal_ids)
            latent_segments.append(ref_video_tokens)
            rotary_segments.append(self.get_rotary_pos_embed_from_ids(
                frame_ids=video_frame_ids,
                spatial_shape=(ref_video_patch_shape[1], ref_video_patch_shape[2]),
            ))
            source_id_segments.append(
                torch.full((ref_video_tokens.shape[1],), SOURCE_ID_EDIT_CONDITION, device=device, dtype=torch.float32)
            )

        img = torch.cat(latent_segments, dim=1)
        visual_source_id = torch.cat(source_id_segments, dim=0).unsqueeze(0)
        current_indices = torch.arange(current_seq_len, device=device).unsqueeze(0)
        vis_freqs_cis = (
            torch.cat([rotary[0] for rotary in rotary_segments], dim=0).unsqueeze(0),
            torch.cat([rotary[1] for rotary in rotary_segments], dim=0).unsqueeze(0),
        )

        head_dim = self.hidden_size // self.heads_num
        cos_3d, sin_3d = vis_freqs_cis
        cos_role, sin_role = self.generate_source_id_rope(
            source_id=visual_source_id,
            head_dim=head_dim,
            device=cos_3d.device,
            dtype=cos_3d.dtype,
        )
        new_cos = cos_3d * cos_role - sin_3d * sin_role
        new_sin = sin_3d * cos_role + cos_3d * sin_role
        vis_freqs_cis = (new_cos, new_sin)

        # A KV-only store forward without the text stream never reads the projected text.
        vec, txt = self.condition_embedder(
            timestep, encoder_hidden_states, project_text=not (skip_text_stream and not return_output)
        )
        vec = vec.unflatten(-1, (NUM_MODULATION_CHUNKS, -1))

        use_memo = bool(kv_cache_pre_rope and
                        cached_temporal_ids is not None and
                        kv_cache_mode in {"reuse", "reuse_store"})
        _assembly_stamp = None
        if use_memo:
            # Signature from the ids where the caller keeps them (host): no device round trip.
            _cached_ids_src = torch.as_tensor(cached_temporal_ids, dtype=torch.long)
            _temporal_sig = (tuple(_cached_ids_src.shape), tuple(_cached_ids_src.reshape(-1).tolist()))
            # Note: kv_cache_generation intentionally excluded from stamp.
            # Writes during the current step go to kv_cache_chunk_id, which is
            # never in selected_chunk_ids (causal constraint).  Including
            # kv_cache_generation caused the cache to be invalidated between
            # denoising steps of the same chunk, forcing torch.cat to re-run
            # for all 40 layers on every step.
            _assembly_stamp = (
                self._kv_cache_scope,
                tuple(self._kv_cache_selected_chunk_ids or ()),
                bool(self._kv_cache_pre_rope),
                _temporal_sig,
            )

        cached_freqs_cis = None
        if use_memo and _assembly_stamp == self._kv_assembly_version:
            cached_freqs_cis = self._kv_assembly_freqs
        elif kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_ids_tensor = torch.as_tensor(cached_temporal_ids, device=device, dtype=torch.long)
            cached_frame_ids = self._get_token_frame_ids(
                (cached_ids_tensor.shape[1], current_patch_shape[1], current_patch_shape[2]),
                device,
                temporal_ids=cached_ids_tensor[0],
            )
            cached_vis_freqs = self.get_rotary_pos_embed_from_ids(
                frame_ids=cached_frame_ids,
                spatial_shape=(current_patch_shape[1], current_patch_shape[2]),
            )
            cached_freqs_cis = (
                cached_vis_freqs[0].unsqueeze(0),
                cached_vis_freqs[1].unsqueeze(0),
            )
            if use_memo:
                self._kv_assembly_version = _assembly_stamp
                self._kv_assembly_freqs = cached_freqs_cis
                self._kv_assembly_cache = {}
                self._kv_assembly_epoch = self.__dict__.get("_kv_assembly_epoch", 0) + 1

        # A captured CUDA graph (xvideo/serving/graph_runner.py) swaps the python KV cache for
        # static buffers: its assembler returns [pool, current] and its writer copies into a
        # staging buffer, so every tensor the graph touches keeps a fixed address.
        _graph_assembler = getattr(self, "_graph_kv_assembler", None)
        _graph_writer = getattr(self, "_graph_kv_writer", None)
        text_keep = text_pad = None
        if mask_text_padding and not skip_text_stream:
            text_keep = encoder_hidden_states_mask[:, :, None, None]
            text_pad = encoder_hidden_states_mask.shape[1] - encoder_hidden_states_mask.sum(dim=1)
        kv_writer_on = _graph_writer is not None or kv_cache_mode in {"store", "reuse_store"}
        kv_only_last = (not return_output) and kv_cache_pre_rope and kv_writer_on
        # A denoise forward returns only the current chunk's rows (the gather below).
        keep_current_only = return_output and kv_cache_mode == "reuse"
        last_layer = len(self.double_blocks) - 1
        for layer_idx, block in enumerate(self.double_blocks):
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                layer_idx=layer_idx,
                kv_cache_reader=None if _graph_assembler is not None else (
                    self._cached_kv_reader if kv_cache_mode in {"reuse", "reuse_store"} else None),
                kv_cache_writer=_graph_writer if _graph_writer is not None else (
                    self._cached_kv_writer if kv_cache_mode in {"store", "reuse_store"} else None),
                kv_cache_assembler=_graph_assembler if _graph_assembler is not None else (
                    self._cached_kv_assembler if use_memo else None),
                skip_text_stream=skip_text_stream,
                kv_cache_pre_rope=kv_cache_pre_rope,
                cached_freqs_cis=cached_freqs_cis,
                text_keep=text_keep,
                text_pad=text_pad,
                kv_only=kv_only_last and layer_idx == last_layer,
                keep_rows=current_seq_len if (keep_current_only and layer_idx == last_layer) else None,
            )
        if not return_output:
            return None, None

        img = self.proj_out(self.norm_out(img))

        gather_index = current_indices.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img = torch.gather(img, dim=1, index=gather_index)
        img = self.unpatchify(img, current_patch_shape[0], current_patch_shape[1], current_patch_shape[2])

        return (img, txt)

    def unpatchify(self, x, t, h, w):
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs
