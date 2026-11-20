import torch
from typing import Union, Tuple


def reshape_for_broadcast(
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
):
    ndim = x.ndim
    shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, f"RoPE requires an even head dimension, got {x.shape[-1]}"
    cos, sin = reshape_for_broadcast(freqs_cis, x)
    cos, sin = cos.to(x.device), sin.to(x.device)
    return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    pos_device = pos.device if torch.is_tensor(pos) else torch.device("cpu")
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=pos_device)[: (dim // 2)].float() / dim)
    )
    freqs = torch.outer(pos * interpolation_factor, freqs)
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
    return freqs_cos, freqs_sin
