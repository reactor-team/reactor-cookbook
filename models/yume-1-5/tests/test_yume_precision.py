"""The released sampler retains float32 VAE weights, latent arithmetic and sample_5b numerics."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_backend_keeps_native_vae_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    import yume_backend
    from yume_backend import YumeBackend

    transformer = Mock()
    transformer.to.return_value = transformer
    transformer.eval.return_value = transformer
    transformer.requires_grad_.return_value = transformer
    vae_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(vae_model, "to", lambda *_a, **_kw: vae_model)
    wan = SimpleNamespace(model=transformer, vae=SimpleNamespace(model=vae_model))
    monkeypatch.setitem(sys.modules, "wan23", SimpleNamespace(Yume=lambda **_kw: wan))
    monkeypatch.setitem(
        sys.modules, "wan23.configs", SimpleNamespace(WAN_CONFIGS={"ti2v-5B": {}})
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "Generator", lambda **_kw: Mock())
    wrapped = []
    monkeypatch.setattr(
        yume_backend, "_wrap_like_sample_5b", lambda module: wrapped.append(module) or module
    )
    tf32 = torch.backends.cuda.matmul.allow_tf32  # not monkeypatchable
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        backend = YumeBackend(
            SimpleNamespace(source_path=Path("."), checkpoint_path=Path("."))
        )
        # sample_5b main(): TF32 on, DiT cast to bf16 before its FSDP wrapper.
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert next(vae_model.parameters()).dtype == torch.float32
    transformer.to.assert_called_with(torch.bfloat16)
    assert wrapped == [transformer] and backend.transformer is transformer


@pytest.mark.parametrize("video", [False, True])
def test_conditioning_matches_native_float_tensor_resize(video):
    import numpy as np

    torch = pytest.importorskip("torch")
    from yume_backend import _resize_conditioning

    pixels = np.random.default_rng(5).integers(0, 256, (3, 7, 11, 3), dtype=np.uint8)
    if not video:
        pixels = pixels[0]
    source = (
        torch.from_numpy(pixels if video else pixels[None]).permute(0, 3, 1, 2).float()
        / 255.0
    )
    if video:
        expected = torch.nn.functional.interpolate(
            (source - 0.5) / 0.5, size=(13, 17), mode="bicubic"
        ).clamp(-1, 1)
    else:
        expected = (
            torch.nn.functional.interpolate(
                source, size=(13, 17), mode="bilinear", align_corners=False
            )
            - 0.5
        ) * 2
    actual = _resize_conditioning(pixels, video=video, height=13, width=17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_frames_round_like_diffusers_postprocess():
    import numpy as np

    torch = pytest.importorskip("torch")
    from yume_backend import _to_uint8

    video = torch.linspace(-1.2, 1.2, 3 * 2 * 5 * 7).reshape(3, 2, 5, 7)
    expected = (
        ((video * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 3, 0).numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    actual = _to_uint8(video)
    assert actual.shape == (2, 5, 7, 3)
    np.testing.assert_array_equal(actual, expected)
    # 0.0 maps to 127.5 -> 128 (rounded), where truncation gave 127.
    assert _to_uint8(torch.zeros(3, 1, 1, 1))[0, 0, 0, 0] == 128
