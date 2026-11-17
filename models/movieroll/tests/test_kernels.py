"""Optional GPU checks; no upstream checkout or model weights needed."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_modulation_rounding_with_mixed_dtypes_and_ragged_sequence_shards():
    from movieroll_kernels import pointwise

    for x_dtype, parameter_dtype in (
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
    ):
        for cond in (0, 5):
            for rank in range(4):
                torch.manual_seed(932)
                parameter = torch.randn(
                    2, 9, 6 * 61, device="cuda", dtype=parameter_dtype
                )[..., 61:122]
                shift = torch.randn_like(parameter)

                def expand(values, cond=cond, rank=rank):
                    def shard(part):
                        return part.repeat_interleave(17, dim=1).chunk(4, dim=1)[rank]

                    return (
                        torch.cat(
                            [shard(values[:, :cond]), shard(values[:, cond:])], dim=1
                        )
                        if cond
                        else shard(values)
                    )

                scale_expanded, shift_expanded = expand(parameter), expand(shift)
                x = torch.randn(
                    2, scale_expanded.shape[1], 122, device="cuda", dtype=x_dtype
                )[..., ::2]
                for gate in (False, True):
                    expected = (
                        x * scale_expanded
                        if gate
                        else x * (1 + scale_expanded) + shift_expanded
                    )
                    actual = pointwise(
                        x,
                        parameter,
                        None if gate else shift,
                        frame_tokens=17,
                        cond=cond,
                        sp_size=4,
                        sp_rank=rank,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
