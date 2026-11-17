"""Inference-only data movement and fixed-timestep projection reuse.

The upstream attention, GEMMs, precision and sampling schedule stay unchanged.
All replacements live in a model worker; the installed upstream is untouched.
"""

from collections import OrderedDict
from functools import wraps
from inspect import signature
from itertools import accumulate


def configure_layout() -> "PrefixAttention":
    import torch
    import torch.distributed as dist
    from movieroll.models.transformers.modules import attention, streaming_attention

    if getattr(attention.all_gather, "_reactor_head_gather", False):
        return attention.chunk_inference_attention
    original_gather = attention.all_gather
    original_pack = streaming_attention.prefix_pack

    def gather_heads(x, dim=1, group=None):
        # This call site gathers heads of replicated text, not uneven image
        # shards. Each rank has exactly H / SP heads and the same B and S.
        if dim != 2 or x.ndim != 4 or torch.is_grad_enabled():
            return original_gather(x, dim=dim, group=group)
        shards = [
            torch.empty_like(x, memory_format=torch.contiguous_format)
            for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather(shards, x.contiguous(), group=group)
        return torch.cat(shards, dim=dim)

    def pack_prefix(x, lengths):
        # torch.cat([one_contiguous_slice]) needlessly copies the whole KV.
        # Keep precisely the original contiguous layout for FlashAttention.
        if x.shape[0] == 1 and len(lengths) == 1 and not torch.is_grad_enabled():
            return x[0, : lengths[0]].contiguous()
        return original_pack(x, lengths)

    gather_heads._reactor_head_gather = True
    attention.all_gather = gather_heads
    streaming_attention.prefix_pack = pack_prefix
    attention.chunk_inference_attention = PrefixAttention(streaming_attention)
    return attention.chunk_inference_attention


class PrefixAttention:
    """Native chunk attention with prefix geometry shared by all DiT layers.

    Derived from MovieRoll's chunk_inference_attention (Tencent Hunyuan
    Community License). Q/K/V, packing, FlashAttention calls and output padding
    follow that function. Only integer length/offset construction is reused.
    A retained mask identity plus its mutation version prevents stale captions.
    """

    def __init__(self, kernels):
        self.kernels = kernels
        self.clear()

    def clear(self):
        self.mask = None
        self.version = None
        self.lengths = None
        self.geometry = {}

    def __call__(
        self,
        query,
        key,
        value,
        encoder_query=None,
        encoder_key=None,
        encoder_value=None,
        text_mask=None,
    ):
        import torch

        ops = self.kernels
        if (
            torch.is_grad_enabled()
            or text_mask is None
            or encoder_query is None
            or encoder_key is None
            or encoder_value is None
        ):
            return ops.chunk_inference_attention(
                query, key, value, encoder_query, encoder_key, encoder_value, text_mask
            )
        try:
            version = text_mask._version
        except RuntimeError:
            version = None  # inference tensors cannot be checked for mutation
        if self.mask is not text_mask or version is None or self.version != version:
            self.mask, self.version = text_mask, version
            self.lengths = text_mask.sum(dim=-1, dtype=torch.int32).tolist()
            self.geometry.clear()

        batch, chunk, heads, depth = query.shape
        text = encoder_query.shape[1]
        cond = key.shape[1] - chunk
        geometry_key = (batch, chunk, cond, query.device)
        if geometry_key not in self.geometry:
            if len(self.geometry) >= 8:
                self.geometry.clear()
            lengths = self.lengths

            def cumulative(sizes):
                return torch.tensor(
                    [0, *accumulate(sizes)], dtype=torch.int32, device=query.device
                )

            img_kv = [cond + chunk + n for n in lengths]
            txt_kv = [chunk + n for n in lengths]
            self.geometry[geometry_key] = (
                img_kv,
                txt_kv,
                cumulative([chunk] * batch),
                cumulative(img_kv),
                cumulative(lengths),
                cumulative(txt_kv),
            )
        img_kv, txt_kv, cu_img, cu_img_kv, cu_txt, cu_txt_kv = self.geometry[
            geometry_key
        ]

        if cond == 0:
            q = ops.prefix_pack(torch.cat([query, encoder_query], dim=1), txt_kv)
            k = ops.prefix_pack(torch.cat([key, encoder_key], dim=1), txt_kv)
            v = ops.prefix_pack(torch.cat([value, encoder_value], dim=1), txt_kv)
            out = ops._attention_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_txt_kv,
                cu_seqlens_k=cu_txt_kv,
                max_seqlen_q=max(txt_kv),
                max_seqlen_k=max(txt_kv),
            )
            return ops.prefix_unpack(out, txt_kv, batch, chunk + text)

        k_img = torch.cat([key, encoder_key], dim=1)
        v_img = torch.cat([value, encoder_value], dim=1)
        k_txt, v_txt = k_img[:, cond:].contiguous(), v_img[:, cond:].contiguous()
        img = ops._attention_varlen(
            query.reshape(-1, heads, depth),
            ops.prefix_pack(k_img, img_kv),
            ops.prefix_pack(v_img, img_kv),
            cu_seqlens_q=cu_img,
            cu_seqlens_k=cu_img_kv,
            max_seqlen_q=chunk,
            max_seqlen_k=max(img_kv),
        ).reshape(batch, chunk, heads, depth)
        txt = ops._attention_varlen(
            ops.prefix_pack(encoder_query, self.lengths),
            ops.prefix_pack(k_txt, txt_kv),
            ops.prefix_pack(v_txt, txt_kv),
            cu_seqlens_q=cu_txt,
            cu_seqlens_k=cu_txt_kv,
            max_seqlen_q=max(self.lengths),
            max_seqlen_k=max(txt_kv),
        )
        txt = ops.prefix_unpack(txt, self.lengths, batch, text)
        return torch.cat([img, txt], dim=1)


class TimestepProjections:
    """Bounded reuse of the exact native AdaLN projection at the same shape.

    Image/text modulation depends only on the timestep embedding, not captions
    or video. A new timestep vector (including its shape) is a different key.
    We run the original projection on every miss; there is no batched GEMM,
    arithmetic fusion, activation cache, or text-conditioning cache.
    """

    def __init__(self, model, limit: int = 8):
        import torch

        self.context = None
        self.caches = []
        original = model.forward
        parameters = signature(original)

        def timestep_key(tensor):
            if tensor is None:
                return None
            return (tuple(tensor.shape), tensor.dtype, tuple(tensor.flatten().tolist()))

        @wraps(original)
        def forward(*args, **kwargs):
            if torch.is_grad_enabled():
                return original(*args, **kwargs)
            bound = parameters.bind(*args, **kwargs).arguments
            previous = self.context
            self.context = {
                "img": timestep_key(bound["timestep"]),
                "txt": timestep_key(bound.get("timestep_txt")),
            }
            try:
                return original(*args, **kwargs)
            finally:
                self.context = previous

        def install(module, role):
            cache = OrderedDict()
            self.caches.append(cache)
            project = module.forward

            @wraps(project)
            def cached(x):
                context = self.context
                if context is None or context[role] is None or torch.is_grad_enabled():
                    return project(x)
                key = (
                    context[role],
                    tuple(x.shape),
                    x.dtype,
                    x.device,
                    torch.is_autocast_enabled("cuda"),
                    torch.get_autocast_dtype("cuda"),
                )
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
                result = project(x)
                cache[key] = result
                if len(cache) > limit:
                    cache.popitem(last=False)
                return result

            module.forward = cached

        for block in model.double_blocks:
            install(block.img_mod, "img")
            install(block.txt_mod, "txt")
        install(model.final_layer.adaLN_modulation, "img")
        model.forward = forward

    def clear(self) -> None:
        for cache in self.caches:
            cache.clear()


def rgb_frames(pixels):
    """Native FP16 rounding and uint8 truncation, before transferring RGB.

    Separate Torch operators retain division/addition rounding boundaries.
    Moving float() earlier or rounding instead of truncating would differ.
    cpu() waits for completion before the Runtime can publish these bytes.
    """
    import torch

    pixels = (pixels / 2 + 0.5).clamp(0, 1).float()
    return (
        (pixels[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).contiguous().cpu().numpy()
    )
