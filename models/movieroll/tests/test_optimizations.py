"""Numerical boundaries of inference optimizations, without model weights."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from movieroll_optimizations import PrefixAttention, TimestepProjections, rgb_frames


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.silu(x) * 3


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = [
            SimpleNamespace(img_mod=Projection(), txt_mod=Projection())
        ]
        self.final_layer = SimpleNamespace(adaLN_modulation=Projection())

    def forward(self, hidden_states, timestep, text_states, timestep_txt=None):
        img = timestep[..., None].float()
        txt = timestep_txt[..., None].float()
        block = self.double_blocks[0]
        return (
            block.img_mod(img).sum()
            + block.txt_mod(txt).sum()
            + self.final_layer.adaLN_modulation(img).sum()
            + text_states.sum()
            + hidden_states.sum()
        )


def test_timestep_reuse_keeps_prompt_and_video_live_and_bounds_storage():
    model = Transformer()
    cache = TimestepProjections(model, limit=2)
    block = model.double_blocks[0]
    x, prompt = torch.ones(2), torch.ones(3)
    timestep, text_t = torch.tensor([[1000, 800]]), torch.tensor([[1000]])
    with torch.no_grad():
        first = model(x, timestep, prompt, timestep_txt=text_t)
        second = model(x + 1, timestep.clone(), prompt + 1, timestep_txt=text_t)
        assert second - first == 5  # conditioning and video are never cached
        assert block.img_mod.calls == block.txt_mod.calls == 1
        for value in (600, 400, 200):
            model(x, torch.tensor([[value, value]]), prompt, timestep_txt=text_t)
        assert all(len(entries) <= 2 for entries in cache.caches)
        # Same scalar values with a different shape are a different GEMM.
        model(x, torch.tensor([[200], [200]]), prompt, timestep_txt=text_t)
        assert block.img_mod.calls == 5
        cache.clear()
        assert all(not entries for entries in cache.caches)
        model(x, timestep, prompt, timestep_txt=text_t)
        assert block.img_mod.calls == 6
    model(x, timestep, prompt, timestep_txt=text_t)
    assert block.img_mod.calls == 7  # autograd always runs the original


def test_rgb_preserves_all_finite_fp16_rounding_and_uint8_truncation():
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
    values = values[torch.isfinite(values)].repeat(3).reshape(1, 3, 1, 1, -1)
    native = (values / 2 + 0.5).clamp(0, 1).float()
    expected = (native[0].permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
    actual = rgb_frames(values)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("batch,cond", [(1, 0), (1, 4), (2, 0), (2, 4)])
def test_prefix_metadata_matches_attention_and_invalidates_for_mask_changes(
    batch, cond
):
    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)

    def varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, **_):
        q_bounds, k_bounds = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        return torch.cat(
            [
                sdpa(q[q0:q1], k[k0:k1], v[k0:k1])
                for q0, q1, k0, k1 in zip(
                    q_bounds, q_bounds[1:], k_bounds, k_bounds[1:]
                )
            ]
        )

    def unpack(x, lengths, batch, seqlen):
        out = x.new_zeros(batch, seqlen, *x.shape[1:])
        offset = 0
        for i, n in enumerate(lengths):
            out[i, :n] = x[offset : offset + n]
            offset += n
        return out

    ops = SimpleNamespace(
        prefix_pack=lambda x, lens: torch.cat([x[b, :n] for b, n in enumerate(lens)]),
        prefix_unpack=unpack,
        _attention_varlen=varlen,
    )
    torch.manual_seed(9021)
    q = torch.randn(batch, 3, 2, 8)
    k, v = [torch.randn(batch, cond + 3, 2, 8) for _ in range(2)]
    tq, tk, tv = [torch.randn(batch, 5, 2, 8) for _ in range(3)]
    mask = torch.tensor([[True, True, True, False, False]] * batch)
    attention = PrefixAttention(ops)
    with torch.no_grad():
        for mutate in (False, False, True):
            if mutate:
                mask[:, 3] = True
            actual = attention(q, k, v, tq, tk, tv, mask)
            expected = torch.zeros_like(actual)
            for b in range(batch):
                n = int(mask[b].sum())
                image_k, image_v = (
                    torch.cat([k[b], tk[b, :n]]),
                    torch.cat([v[b], tv[b, :n]]),
                )
                if cond:
                    expected[b, :3] = sdpa(q[b], image_k, image_v)
                    expected[b, 3 : 3 + n] = sdpa(
                        tq[b, :n], image_k[cond:], image_v[cond:]
                    )
                else:
                    merged = sdpa(torch.cat([q[b], tq[b, :n]]), image_k, image_v)
                    expected[b, : 3 + n] = merged
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Replacement mask identity must invalidate even if its version matches.
        changed = mask.clone()
        changed[:, 1:] = False
        attention(q, k, v, tq, tk, tv, changed)
        assert attention.lengths == [1] * batch
        attention.clear()
        assert attention.mask is None and not attention.geometry
