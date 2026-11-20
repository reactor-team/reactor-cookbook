"""The denoiser's forward for one streamed block, without upstream's per-call overhead.

Upstream's ``_forward_inference`` serves every case LiveAvatar was trained and
evaluated in. A streamed block is one case of it: three latent frames attend to
their step's cache and the shared reference cache. This module computes the
same function for that case with the same weights, and drops what the general
path pays on every call:

- the 12k-token key/value window is no longer concatenated with the reference
  keys in every layer: each step keeps one buffer, the reference first and the
  rolling window after it, and attention reads a slice of it;
- the six modulation tensors stay one row per latent frame and broadcast over
  the frame's tokens, rather than repeating to the full sequence in float32;
- the rotary tables are built once per block (and once per reference offset),
  not once per step on the CPU;
- the text keys and values of the cross-attention are computed once per
  prompt, since the prompt does not change during a session;
- the audio encoder runs once per block, not once per step: its input is the
  block's audio, which no step changes;
- no value is read back to the host inside the forward, so a step compiles to
  two graphs per layer, and a whole step is captured as one CUDA graph.

The arithmetic dtypes follow upstream: the residual stream and modulation in
float32, the projections in bfloat16, rotary embedding in complex float32 on
the same half-precision tables upstream builds, written as its real cos/sin form
so it fuses with the projections around it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

#: Rotary offsets upstream draws for the reference tokens, per call: ``randint(4, 30)``.
_COND_OFFSETS = (4, 30)


@dataclass
class StepCache:
    """One denoising step's keys and values: the reference tokens first, then the rolling window.

    Attributes:
        k: Keys, ``(layers, 1, cond + window, heads, head_dim)``. The reference part is written every
            call with that call's rotary offset; the window part holds the roped keys of past blocks.
        v: Values, same layout. The reference part is written once per clip.
    """

    k: torch.Tensor
    v: torch.Tensor


class FastForward:
    """The streamed-block forward of a loaded ``CausalWanModel_S2V``.

    Args:
        model: The denoiser, with LiveAvatar's weights merged.
        frame_tokens: Tokens per latent frame.
        window_tokens: Tokens of the rolling window: one clip.
        steps: Denoising steps, one cache each.
    """

    def __init__(
        self, model: Any, *, frame_tokens: int, window_tokens: int, steps: int
    ) -> None:
        self.model = model
        self.frame_tokens = frame_tokens
        self.window_tokens = window_tokens
        self.heads = model.num_heads
        self.head_dim = model.dim // model.num_heads
        self.device = model.patch_embedding.weight.device
        self.dtype = torch.bfloat16
        self.cond_tokens = 0
        self._rope: dict[int, torch.Tensor] = {}
        self._cond_rope: dict[int, torch.Tensor] = {}
        self._cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._attention = _attend
        layers = model.num_layers
        self.caches = [
            StepCache(
                k=torch.zeros(
                    (layers, 1, 0, self.heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                v=torch.zeros(
                    (layers, 1, 0, self.heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )
            for _ in range(steps)
        ]
        self._inputs = self._inputs_eager
        self._outputs = self._outputs_eager
        self._graphs: dict[tuple[int, int, int], Any] | None = None
        self._static: dict[str, torch.Tensor] = {}

    def compile(self, *, cuda_graphs: bool = True) -> None:
        """Compile the two halves of a layer; every layer reuses the graphs with its own weights.

        Args:
            cuda_graphs: Also capture each step as a CUDA graph, keyed by the step, the block's place in
                the window and the window's filled length: at most 32 graphs. A key runs as it is the
                first time it comes up, which compiles it, and is captured the second time.
        """
        self._inputs = torch.compile(self._inputs_eager, fullgraph=True, dynamic=False)
        self._outputs = torch.compile(
            self._outputs_eager, fullgraph=True, dynamic=False
        )
        if cuda_graphs:
            self._graphs = {}
            self._pool = torch.cuda.graph_pool_handle()

    # -- per clip and per prompt ----------------------------------------------------

    def set_context(self, context: torch.Tensor) -> None:
        """Cache each layer's cross-attention keys and values for the encoded prompt ``(L, 4096)``."""
        model = self.model
        padded = torch.cat(
            [
                context,
                context.new_zeros(model.text_len - context.size(0), context.size(1)),
            ]
        )
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            text = model.text_embedding(padded[None])
            self._cross_kv = []
            for block in model.blocks:
                attn = block.cross_attn
                k = attn.norm_k(attn.k(text)).view(1, -1, self.heads, self.head_dim)
                v = attn.v(text).view(1, -1, self.heads, self.head_dim)
                self._cross_kv.append(
                    (k.to(self.dtype).contiguous(), v.to(self.dtype).contiguous())
                )

    def set_reference(self, cond_cache: list[dict[str, torch.Tensor]]) -> None:
        """Take the reference keys and values upstream's prefill wrote, and reset every window.

        Args:
            cond_cache: Upstream's per-layer ``cond_k``, ``cond_v`` and ``cond_end`` after its prefill.
        """
        cond = int(cond_cache[0]["cond_end"])
        self.cond_tokens = cond
        cond_k = torch.stack([layer["cond_k"][:, :cond] for layer in cond_cache])
        # Captured graphs read these buffers by address, so a new reference is copied into them.
        if (
            getattr(self, "cond_k", None) is not None
            and self.cond_k.shape == cond_k.shape
        ):
            self.cond_k.copy_(cond_k)
        else:
            if self._graphs:
                self._graphs.clear()
            self.cond_k = cond_k.contiguous()
        total = cond + self.window_tokens
        layers = len(cond_cache)
        for cache in self.caches:
            if cache.k.shape[2] != total:
                if self._graphs:
                    self._graphs.clear()
                shape = (layers, 1, total, self.heads, self.head_dim)
                cache.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
                cache.v = torch.zeros(shape, dtype=self.dtype, device=self.device)
            for i, layer in enumerate(cond_cache):
                cache.v[i, :, :cond] = layer["cond_v"][:, :cond]
        self._cond_rope.clear()

    # -- one block ------------------------------------------------------------------

    def encode_audio(
        self, audio_input: torch.Tensor, motion_frames: tuple[int, int]
    ) -> None:
        """Encode the block's audio features for the steps that follow.

        Args:
            audio_input: The block's audio features, ``(1, layers, 1024, 4 * F)``.
            motion_frames: ``(motion frames, motion latent frames)``.
        """
        model = self.model
        # Upstream pads the motion frames before the block with the block's first feature.
        audio = torch.cat(
            [audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input],
            dim=-1,
        )
        audio_global, audio_emb = (
            a.to(model.dtype) for a in model.casual_audio_encoder(audio)
        )
        frames = audio_emb.shape[1] - motion_frames[1]
        self._audio_global = (
            audio_global[:, motion_frames[1] :]
            .reshape(frames, -1, model.dim)[:, 0]
            .contiguous()
        )
        self._audio_tokens = (
            audio_emb[:, motion_frames[1] :].reshape(frames, -1, model.dim).contiguous()
        )

    # -- one step -------------------------------------------------------------------

    def __call__(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_states: torch.Tensor,
        step: int,
        current_start: int,
    ) -> torch.Tensor:
        """Predict the velocity of one block at one step, and write the block into the step's window.

        Args:
            latents: The block's noisy latents, ``(16, F, h, w)``.
            timestep: The step's timestep per latent frame, ``(1, F)``.
            cond_states: Pose conditioning, ``(1, 16, F, h, w)``.
            step: The denoising step, which selects the cache.
            current_start: The block's first token in the clip, as upstream counts it.

        The block's audio must have gone through :meth:`encode_audio` first.

        Returns:
            The velocity, ``(16, F, h, w)``.
        """
        frames = latents.shape[1]
        tokens = frames * self.frame_tokens
        frame = current_start // self.frame_tokens
        grid = (frames, latents.shape[2] // 2, latents.shape[3] // 2)
        window_at = current_start % self.window_tokens
        active = (
            self.window_tokens
            if current_start >= self.window_tokens
            else current_start + tokens
        )
        inputs = {
            "latents": latents,
            "freqs": self._block_rope(frame, grid),
            "cond_freqs": self._reference_rope(frame),
            "audio_global": self._audio_global,
            "audio_tokens": self._audio_tokens,
        }
        if self._graphs is None:
            return self._run(inputs, timestep, cond_states, step, window_at, active)
        key = (step, window_at, active)
        entry = self._graphs.get(key)
        if entry is None:
            self._graphs[key] = "seen"
            return self._run(inputs, timestep, cond_states, step, window_at, active)
        for name, value in inputs.items():
            static = self._static.get(name)
            if static is None:
                static = self._static[name] = torch.empty_like(value)
            static.copy_(value)
        if entry == "seen":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = self._run(
                    self._static, timestep, cond_states, step, window_at, active
                )
            entry = self._graphs[key] = (graph, out)
        graph, out = entry
        graph.replay()
        return out.clone()

    def _run(
        self,
        inputs: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        cond_states: torch.Tensor,
        step: int,
        window_at: int,
        active: int,
    ) -> torch.Tensor:
        """The step itself, on the tensors in ``inputs``; nothing here reads back to the host."""
        model = self.model
        cache = self.caches[step]
        latents = inputs["latents"]
        x = model.patch_embedding(latents.unsqueeze(0)) + model.cond_encoder(
            cond_states[0].unsqueeze(0)
        )
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + model.trainable_cond_mask.weight[0].to(x.dtype)
        freqs, cond_freqs = inputs["freqs"], inputs["cond_freqs"]

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t = torch.cat([timestep, torch.zeros_like(timestep)])
            e = model.time_embedding(_sinusoid(model.freq_dim, t.flatten()).float())
            e0 = (
                model.time_projection(e)
                .unflatten(1, (6, model.dim))
                .unflatten(dim=0, sizes=t.shape)
            )
        e = e[: timestep.shape[1]]
        modulation = e0[0]  # (F, 6, dim): the noisy tokens' rows

        injected = model.audio_injector.injected_block_id
        injector = model.audio_injector
        for idx, block in enumerate(model.blocks):
            slot = injected.get(idx)
            x = self._layer(
                block,
                injector.injector_adain_layers[slot] if slot is not None else None,
                injector.injector[slot] if slot is not None else None,
                inputs["audio_global"],
                inputs["audio_tokens"],
                x,
                modulation,
                freqs,
                cond_freqs,
                self.cond_k[idx],
                cache.k[idx],
                cache.v[idx],
                self._cross_kv[idx],
                window_at,
                active,
            )
        x = model.head(x, e)
        return model.unpatchify(x, torch.tensor([list(grid)]))[0]

    def _layer(
        self,
        block: Any,
        adain: Any,
        injector: Any,
        audio_global: torch.Tensor,
        audio_tokens: torch.Tensor,
        x: torch.Tensor,
        modulation: torch.Tensor,
        freqs: torch.Tensor,
        cond_freqs: torch.Tensor,
        cond_k: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        window_at: int,
        active: int,
    ) -> torch.Tensor:
        """One transformer layer of upstream's ``CausalWanS2VAttentionBlock`` for a streamed block.

        With ``adain`` and ``injector`` given, the layer also runs upstream's audio injection after it
        (``after_transformer_block``): each latent frame's tokens attend to that frame's audio tokens.

        The layer runs as two graphs with the cache writes between them, outside any graph: a graph that
        wrote into the cache would copy the whole buffer back on every call.
        """
        q, k, v, roped_cond_k = self._inputs(
            block, x, modulation, freqs, cond_freqs, cond_k
        )
        cond = self.cond_tokens
        start = cond + window_at
        k_cache[:, start : start + k.shape[1]] = k
        v_cache[:, start : start + v.shape[1]] = v
        k_cache[:, :cond] = roped_cond_k
        end = cond + active
        return self._outputs(
            block,
            adain,
            injector,
            audio_global,
            audio_tokens,
            x,
            modulation,
            q,
            k_cache[:, :end],
            v_cache[:, :end],
            cross_kv,
        )

    def _inputs_eager(
        self,
        block: Any,
        x: torch.Tensor,
        modulation: torch.Tensor,
        freqs: torch.Tensor,
        cond_freqs: torch.Tensor,
        cond_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The self-attention's roped queries and keys, its values, and this call's roped reference keys."""
        frames = modulation.shape[0]
        b, length, dim = x.shape
        e = (block.modulation.float().view(1, 6, dim) + modulation).unsqueeze(
            2
        )  # (F, 6, 1, dim)
        norm_x = (
            block.norm1(x).float().view(frames, length // frames, dim) * (1 + e[:, 1])
            + e[:, 0]
        )
        attn = block.self_attn
        hidden = norm_x.view(b, length, dim).to(self.dtype)
        q = attn.norm_q(attn.q(hidden)).view(b, length, self.heads, self.head_dim)
        k = attn.norm_k(attn.k(hidden)).view(b, length, self.heads, self.head_dim)
        v = attn.v(hidden).view(b, length, self.heads, self.head_dim)
        return (
            _rope(q, freqs).to(self.dtype),
            _rope(k, freqs).to(self.dtype),
            v.to(self.dtype),
            _rope(cond_k, cond_freqs).to(self.dtype),
        )

    def _outputs_eager(
        self,
        block: Any,
        adain: Any,
        injector: Any,
        audio_global: torch.Tensor,
        audio_tokens: torch.Tensor,
        x: torch.Tensor,
        modulation: torch.Tensor,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """The rest of the layer: self-attention over ``keys``/``values``, cross-attention, the FFN, the audio."""
        frames = modulation.shape[0]
        b, length, dim = x.shape
        e = (block.modulation.float().view(1, 6, dim) + modulation).unsqueeze(
            2
        )  # (F, 6, 1, dim)
        e = [
            e[:, i] for i in range(6)
        ]  # each (F, 1, dim), broadcast over a frame's tokens

        def per_frame(t: torch.Tensor) -> torch.Tensor:
            return t.view(frames, length // frames, dim)

        attn = block.self_attn
        y = attn.o(self._attention(q, keys, values).flatten(2))
        x = (per_frame(x.float()) + per_frame(y.float()) * e[2]).view(b, length, dim)

        cross = block.cross_attn
        hidden = block.norm3(x.to(self.dtype)).to(self.dtype)
        cq = cross.norm_q(cross.q(hidden)).view(b, length, self.heads, self.head_dim)
        y = self._attention(cq.to(self.dtype), cross_kv[0], cross_kv[1])
        x = x + cross.o(y.flatten(2)).float()

        norm2_x = per_frame(block.norm2(x).float()) * (1 + e[4]) + e[3]
        y = block.ffn(norm2_x.view(b, length, dim).to(self.dtype))
        x = (per_frame(x) + per_frame(y.float()) * e[5]).view(b, length, dim)
        x = x.to(self.dtype)
        if injector is not None:
            hidden = adain(x.view(frames, length // frames, dim), temb=audio_global)
            lens = torch.full(
                (frames,), audio_tokens.shape[1], dtype=torch.long, device=x.device
            )
            residual = injector(x=hidden, context=audio_tokens, context_lens=lens)
            x = x + residual.reshape(b, length, dim)
        return x

    # -- rotary tables --------------------------------------------------------------

    def _block_rope(self, frame: int, grid: tuple[int, int, int]) -> torch.Tensor:
        """The block's rotary table: upstream's ``rope_precompute`` at the block's frame, cached."""
        table = self._rope.get(frame)
        if table is None:
            from liveavatar.models.wan.causal_s2v_utils import rollout_grid_sizes
            from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils import (
                rope_precompute,
            )

            if len(self._rope) > 64:
                self._rope.clear()
            sizes_t = torch.tensor([list(grid)])
            sizes = [[torch.zeros_like(sizes_t), sizes_t, sizes_t]]
            tokens = grid[0] * grid[1] * grid[2]
            probe = torch.empty(
                (1, tokens, self.heads, self.head_dim), device=self.device
            )
            table = rope_precompute(
                probe, rollout_grid_sizes(sizes, frame), self.model.freqs, start=None
            )
            table = _cos_sin(table)
            self._rope[frame] = table
        return table

    def _reference_rope(self, frame: int) -> torch.Tensor:
        """The reference tokens' rotary table for this call: upstream draws a fresh offset per call."""
        from liveavatar.models.wan.causal_s2v_utils import rollout_grid_sizes
        from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils import rope_precompute

        relative = random.randint(*_COND_OFFSETS)
        shift = max(0, frame - (30 - relative))
        table = self._cond_rope.get(shift)
        if table is None:
            if len(self._cond_rope) > 256:
                self._cond_rope.clear()
            rope_cache = self.model.rope_cache
            probe = torch.empty(rope_cache["cond_shape"], device=self.device)
            table = rope_precompute(
                probe,
                rollout_grid_sizes(rope_cache["grid_sizes"], shift),
                self.model.freqs,
                start=None,
            )
            table = _cos_sin(table[:, : self.cond_tokens])
            self._cond_rope[shift] = table
        return table


class Fp8Linear(torch.nn.Module):
    """A linear layer on FP8 tensor cores: upstream's ``--fp8`` scheme, in a form ``torch.compile`` traces.

    The weight is quantized once to ``float8_e4m3fn`` with one scale per tensor; each input is quantized
    per tensor at call time, and the product accumulates in FP8's fast mode into bfloat16.
    """

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach()
        scale = _FP8_MAX / weight.abs().max().float().clamp(min=1e-12)
        self.register_buffer(
            "weight_fp8",
            (weight.float() * scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn),
        )
        self.register_buffer("weight_scale", scale.reciprocal())
        self.bias = (
            None
            if linear.bias is None
            else torch.nn.Parameter(linear.bias.detach().to(torch.bfloat16))
        )
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        amax = x2.abs().max().float()
        scale = (_FP8_MAX / amax.clamp(min=1e-12)).clamp(max=_FP8_MAX)
        x8 = (x2.float() * scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x8,
            self.weight_fp8.t(),
            scale_a=scale.reciprocal(),
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return out.reshape(*shape[:-1], self.out_features)


_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def quantize_blocks_fp8(model: Any) -> int:
    """Swap the transformer blocks' projections for :class:`Fp8Linear`. Returns how many were swapped."""
    swapped = 0
    for block in model.blocks:
        for parent, names in (
            (block.self_attn, ("q", "k", "v", "o")),
            (block.cross_attn, ("q", "o")),
            (block.ffn, ("0", "2")),
        ):
            for name in names:
                linear = parent._modules[name]
                if isinstance(linear, torch.nn.Linear):
                    parent._modules[name] = Fp8Linear(linear)
                    swapped += 1
    torch.cuda.empty_cache()
    return swapped


def _cos_sin(table: torch.Tensor) -> torch.Tensor:
    """Upstream's complex rotary table ``(1, L, heads, head_dim / 2)`` as float32 ``(2, 1, L, 1, head_dim / 2)``.

    Every head shares the table, so one head's row is kept.
    """
    real = torch.view_as_real(table[:, :, :1]).float()
    return torch.stack([real[..., 0], real[..., 1]]).contiguous()


def _rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``(1, L, heads, head_dim)`` by a :func:`_cos_sin` table: the complex product, in float32."""
    b, length, heads, dim = x.shape
    pairs = x.float().reshape(b, length, heads, dim // 2, 2)
    re, im = pairs[..., 0], pairs[..., 1]
    cos, sin = freqs[0, :, :length], freqs[1, :, :length]
    return torch.stack([re * cos - im * sin, re * sin + im * cos], dim=-1).flatten(3)


def _sinusoid(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000, -torch.arange(half, device=position.device).to(position).div(half)
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def install_dense_attention() -> None:
    """Route upstream's ``flash_attention`` through cuDNN's dense attention for full-length keys.

    The audio injector calls it with ``k_lens`` equal to the key length, so the keys carry no
    padding and dense attention over all of them is the same computation.
    """
    from liveavatar.models.wan.wan_2_2.modules import model as wan_model

    original = wan_model.flash_attention

    def flash_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        dense = (
            q_lens is None
            and not kwargs.get("causal", False)
            and kwargs.get("window_size", (-1, -1)) == (-1, -1)
        )
        if dense and (k_lens is None or isinstance(k_lens, torch.Tensor)):
            return _attend(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            ).to(q.dtype)
        return original(q, k, v, q_lens=q_lens, k_lens=k_lens, **kwargs)

    wan_model.flash_attention = flash_attention


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dense attention over all of ``k`` with cuDNN, the kernel upstream's ``attention`` uses; ``(B, L, heads, dim)``."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
    return out.transpose(1, 2)
