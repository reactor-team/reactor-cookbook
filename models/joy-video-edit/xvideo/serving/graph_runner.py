from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import torch

from xvideo.models.dit.rope import apply_rotary_emb

GRAPH_ENV = "JOYOMNI_CUDA_GRAPH"

# A captured graph's KV pool is laid out as [sink, prev1][:history_chunks] (+ ref): the steady
# runner (history_chunks=2) serves the window [0, k-1, k] -- and the frozen window [0, anchor, k],
# whose anchor sits at position k-1 -- while history_chunks=1 serves chunk 1 ([0, 1]) and
# history_chunks=0 serves chunk 0.
GRAPH_WINDOW_CHUNKS = 3


def graph_env_enabled() -> bool:
    return os.environ.get(GRAPH_ENV, "1").lower() in {"1", "true", "yes", "on"}


class StreamingGraphRunner:
    def __init__(
        self,
        transformer,
        *,
        chunk_tokens: int,
        ref_tokens: int,
        latent_shape: tuple[int, ...],
        txt_len: int,
        max_temporal_ids: int,
        pos_freqs: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        autocast_ctx: Callable[[], Any],
        mask_text_padding: bool = False,
        history_chunks: int = 2,
        mem_pool: Any = None,
    ):
        self.transformer = transformer
        # The text is padded to txt_len and in_prompt_mask marks the real tokens (bound per
        # session): attention masks the padding from the mask buffer, so the graph serves any
        # prompt that fits.
        self.mask_text_padding = bool(mask_text_padding)
        self.device = device
        self.dtype = dtype
        self.autocast_ctx = autocast_ctx
        self.num_layers = len(transformer.double_blocks)
        heads = int(transformer.config.heads_num)
        head_dim = int(transformer.config.hidden_size) // heads

        self.chunk_tokens = chunk_tokens
        self.ref_tokens = int(ref_tokens)
        self.txt_len = int(txt_len)
        self.max_temporal_ids = int(max_temporal_ids)
        self.history_chunks = int(history_chunks)
        if self.history_chunks not in (0, 1, 2):
            raise ValueError(f"history_chunks must be 0, 1 or 2, got {history_chunks}")
        # Graphs are replayed one at a time on one stream and their output is copied out right
        # after replay, so every runner can capture into one shared memory pool.
        self.mem_pool = mem_pool
        self.sink_off = 0
        self.prev1_off = chunk_tokens
        self.ref_off = self.history_chunks * chunk_tokens
        self.pool_len = self.history_chunks * chunk_tokens + self.ref_tokens

        L = self.num_layers
        # The steady runner replays chunk after chunk, so its pool is kept as the prefix of a
        # [pool | current] attention window per layer: a denoise block writes the current tokens
        # (image, reference video, text) into the tail and attends over the whole buffer, with no
        # per-call copy of the pool.  The chunk-0/1 runners serve one chunk per session and copy.
        self.win_cur = 2 * chunk_tokens + self.txt_len if self.history_chunks == 2 else 0
        self.win_k = torch.zeros(L, 1, self.pool_len + self.win_cur, heads, head_dim, device=device, dtype=dtype)
        self.win_v = torch.zeros_like(self.win_k)
        self.pool_k = self.win_k[:, :, :self.pool_len]
        self.pool_v = self.win_v[:, :, :self.pool_len]
        self.stage_k = torch.zeros(L, 1, chunk_tokens, heads, head_dim, device=device, dtype=dtype)
        self.stage_v = torch.zeros_like(self.stage_k)
        # Double-buffered published KV for the steady runner, which serves chunk after chunk; the
        # chunk-0/1 runners serve one chunk per session and publish a copy instead.
        n_pub = 2 if self.history_chunks == 2 else 0
        self.pub_k = torch.zeros(n_pub, L, 1, chunk_tokens, heads, head_dim, device=device, dtype=dtype)
        self.pub_v = torch.zeros_like(self.pub_k)
        self._pub_parity = 0

        cos, sin = pos_freqs
        ct = chunk_tokens
        # Rope table for cached-KV positions 0..mti-1: row p = freqs of one chunk at temporal position p.
        self.pos_cos = cos.squeeze(0).view(self.max_temporal_ids, ct, -1).contiguous()
        self.pos_sin = sin.squeeze(0).view(self.max_temporal_ids, ct, -1).contiguous()
        # Writable buffer read by run_commit inside the graph; row is swapped per chunk via set_positions.
        self.commit_cos = self.pos_cos[self.max_temporal_ids - 1].clone().unsqueeze(0)
        self.commit_sin = self.pos_sin[self.max_temporal_ids - 1].clone().unsqueeze(0)

        self.in_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_timestep = torch.zeros((latent_shape[0],), device=device, dtype=torch.float32)
        self.in_ref_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_store_latent = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.in_store_timestep = torch.zeros((latent_shape[0],), device=device, dtype=dtype)
        self.in_prompt = torch.zeros((1, self.txt_len, int(transformer.config.text_states_dim)),
                                     device=device, dtype=dtype)
        self.in_prompt_mask = torch.ones((1, self.txt_len), device=device, dtype=torch.bool)
        self.in_current_ids = torch.full((1, latent_shape[2]), self.max_temporal_ids,
                                         device=device, dtype=torch.long)

        self.full_graph: Optional[torch.cuda.CUDAGraph] = None
        self.in_noise = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.out_latents: Optional[torch.Tensor] = None
        self.pool_dirty = True
        self.ready = False
        self.bound_token: Optional[int] = None
        self.last_graph_chunk = -(10 ** 9)

    def _pool_assembler(self, layer_idx, *, device, dtype, cached_freqs_cis=None,
                        current_k=None, current_v=None, reserve_current=None):
        # Same contract as the transformer's own assembler: [cached, current] along tokens.
        pk, pv = self.pool_k[layer_idx], self.pool_v[layer_idx]
        if reserve_current is not None:
            # [pool | current] buffers with the pool prefix filled; the block writes the
            # current tokens into the tail itself (no intermediate current tensor, no cat).
            n_cur, heads, head_dim = reserve_current
            n = pk.shape[1]
            if n_cur == self.win_cur:
                return self.win_k[layer_idx], self.win_v[layer_idx], n
            full_k = torch.empty((1, n + n_cur, heads, head_dim), device=pk.device, dtype=pk.dtype)
            full_v = torch.empty_like(full_k)
            full_k[:, :n].copy_(pk)
            full_v[:, :n].copy_(pv)
            return full_k, full_v, n
        return (torch.cat([pk, current_k], dim=1),
                torch.cat([pv, current_v], dim=1))

    def _stage_writer(self, layer_idx, key, value):
        self.stage_k[layer_idx].copy_(key)
        self.stage_v[layer_idx].copy_(value)

    def capture(self, *, timesteps: torch.Tensor, sigmas: torch.Tensor) -> None:
        tf = self.transformer
        t0 = time.time()
        mem_pool = self.mem_pool if self.mem_pool is not None else torch.cuda.graph_pool_handle()
        torch.cuda.synchronize(self.device)

        def run_denoise():
            with self.autocast_ctx():
                with tf.cache_context("cond"):
                    return tf(
                        hidden_states=self.in_latent,
                        timestep=self.in_timestep,
                        encoder_hidden_states=self.in_prompt,
                        encoder_hidden_states_mask=self.in_prompt_mask,
                        ref_video_latent=self.in_ref_latent,
                        current_temporal_ids=self.in_current_ids,
                        cached_temporal_ids=None,
                        kv_cache_mode="reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=None,
                        kv_cache_selected_chunk_ids=[],
                        kv_cache_pre_rope=True,
                        mask_text_padding=self.mask_text_padding,
                    )[0]

        def run_store():
            with self.autocast_ctx():
                with tf.cache_context("cond"):
                    tf(
                        hidden_states=self.in_store_latent,
                        timestep=self.in_store_timestep,
                        encoder_hidden_states=self.in_prompt,
                        encoder_hidden_states_mask=self.in_prompt_mask,
                        current_temporal_ids=self.in_current_ids,
                        cached_temporal_ids=None,
                        kv_cache_mode="store",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=None,
                        kv_cache_selected_chunk_ids=[],
                        kv_cache_pre_rope=True,
                        skip_text_stream=True,
                        return_output=False,
                    )

        def run_commit():
            if self.history_chunks != 2:
                return  # the next chunk runs on another runner, seeded from the python cache
            seg = slice(self.prev1_off, self.prev1_off + self.chunk_tokens)
            for li in range(self.num_layers):
                roped = apply_rotary_emb(self.stage_k[li], (self.commit_cos, self.commit_sin))
                self.pool_k[li, :, seg].copy_(roped)
                self.pool_v[li, :, seg].copy_(self.stage_v[li])

        ts_vals = [float(t) for t in timesteps]
        dt_vals = [float(sigmas[i + 1] - sigmas[i]) for i in range(len(ts_vals))]

        def run_full():
            lat32 = self.in_noise.to(torch.float32)
            tf._graph_kv_assembler = self._pool_assembler
            tf._graph_kv_writer = None
            for t_val, dt in zip(ts_vals, dt_vals):
                self.in_latent.copy_(lat32.to(self.dtype))
                self.in_timestep.fill_(t_val)
                pred = run_denoise()
                lat32 = lat32 + pred.to(torch.float32) * dt
            self.in_store_latent.copy_(lat32.to(self.dtype))
            tf._graph_kv_assembler = None
            tf._graph_kv_writer = self._stage_writer
            try:
                run_store()
            finally:
                tf._graph_kv_writer = None
            run_commit()
            return lat32

        try:
            run_full()
            torch.cuda.synchronize(self.device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=mem_pool, capture_error_mode="thread_local"):
                self.out_latents = run_full()
            self.full_graph = g
        finally:
            tf._graph_kv_assembler = None
            tf._graph_kv_writer = None

        torch.cuda.synchronize(self.device)
        self.ready = True
        static_gib = sum(t.numel() * t.element_size() for t in (
            self.win_k, self.win_v, self.stage_k, self.stage_v, self.pub_k, self.pub_v)) / 2 ** 30
        print(f"#####[GRAPH] captured whole-chunk graph in {time.time() - t0:.1f}s "
              f"(static bufs {static_gib:.2f} GiB, txt_len={self.txt_len}, "
              f"ref_tokens={self.ref_tokens}, history_chunks={self.history_chunks}, "
              f"mti={self.max_temporal_ids})", flush=True)

    def bind_session(self, token: int, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor) -> None:
        self.in_prompt.copy_(prompt_embeds.to(dtype=self.dtype))
        self.in_prompt_mask.copy_(prompt_mask.to(dtype=torch.bool).reshape(self.in_prompt_mask.shape))
        self.bound_token = token
        self.pool_dirty = True
        self.last_graph_chunk = -(10 ** 9)

    def _pos_freqs(self, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos_cos[pos].unsqueeze(0), self.pos_sin[pos].unsqueeze(0)

    def set_positions(self, chunk_idx: int) -> None:
        """Point the graph's position-dependent buffers at chunk_idx's temporal layout.

        Mirrors _gather_window_temporal_ids for the window [0, k-1, k]:
        current sits at min(k, mti), the committed KV (prev1 of chunk k+1) at min(k, mti-1).
        """
        mti = self.max_temporal_ids
        self.in_current_ids.fill_(min(chunk_idx, mti))
        commit_pos = min(chunk_idx, mti - 1)
        self.commit_cos.copy_(self.pos_cos[commit_pos].unsqueeze(0))
        self.commit_sin.copy_(self.pos_sin[commit_pos].unsqueeze(0))

    def _seed_segment(self, chunk_store: dict, off: int, length: int,
                      freqs: Optional[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for li in range(self.num_layers):
            entry = chunk_store[li]
            k = entry["key"].to(device=self.device, dtype=self.dtype)
            v = entry["value"].to(device=self.device, dtype=self.dtype)
            if entry.get("pre_rope", False):
                if freqs is None:
                    raise RuntimeError("graph seed: pre-rope entry in a no-rope segment")
                k = apply_rotary_emb(k, freqs)
            self.pool_k[li, :, off:off + length].copy_(k)
            self.pool_v[li, :, off:off + length].copy_(v)

    def seed_from_cache(self, scope_store: dict, *, sink_mem_id: Optional[int],
                        prev1_mem_id: Optional[int], prev1_pos: int, ref_mem_id: Optional[int]) -> None:
        segments = [
            (sink_mem_id, self.sink_off, self._pos_freqs(0)),
            (prev1_mem_id, self.prev1_off, self._pos_freqs(prev1_pos)),
        ][: self.history_chunks]
        for mem_id, off, freqs in segments:
            store = scope_store.get(mem_id)
            if store is None:
                raise RuntimeError(f"graph seed: python cache missing mem_id={mem_id}")
            self._seed_segment(store, off, self.chunk_tokens, freqs)
        if self.ref_tokens > 0:
            store = scope_store.get(ref_mem_id)
            if store is None:
                raise RuntimeError(f"graph seed: python cache missing ref mem_id={ref_mem_id}")
            self._seed_segment(store, self.ref_off, self.ref_tokens, None)
        self.pool_dirty = False

    def publish_to_cache(self, scope_store: dict, mem_id: int) -> None:
        if self.history_chunks != 2:
            scope_store[mem_id] = {
                li: {"key": self.stage_k[li].clone(), "value": self.stage_v[li].clone(), "pre_rope": True}
                for li in range(self.num_layers)
            }
            return
        p = self._pub_parity
        self._pub_parity ^= 1
        self.pub_k[p].copy_(self.stage_k)
        self.pub_v[p].copy_(self.stage_v)
        chunk_store = {}
        for li in range(self.num_layers):
            chunk_store[li] = {
                "key": self.pub_k[p, li],
                "value": self.pub_v[p, li],
                "pre_rope": True,
            }
        scope_store[mem_id] = chunk_store
