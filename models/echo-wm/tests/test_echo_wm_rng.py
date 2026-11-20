"""The native causal rollout RNG is independent of conditioning setup draws."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_backend_restarts_native_rollout_generator(monkeypatch):
    torch = pytest.importorskip("torch")
    from echo_wm_assets import activate_source, read_config
    from reactor_runtime import get_weights_path

    config = read_config(Path(__file__).parents[1] / "echo_wm.yaml", get_weights_path())
    if not config.source_path.exists():
        pytest.skip("Pinned upstream source is required for the native RNG oracle")
    activate_source(config)
    import echo_wm_backend as module

    backend = module.EchoWMBackend.__new__(module.EchoWMBackend)
    backend._config = SimpleNamespace(
        height=32,
        width=32,
        fps=24,
        max_chunks=512,
        video_chunk_size=3,
        frames_per_chunk=24,
    )
    backend._device = torch.device("cpu")
    backend._dtype = torch.float32
    backend._total_pixel_frames = 25
    backend._audio_latents = 5
    backend._video_encoder = object()
    backend._components = SimpleNamespace(audio_patchifier=object())
    backend._patches_per_frame = 1
    backend._wrapper = object()
    backend._video_blocks = [(0, 1)]
    backend._audio_blocks = [(0, 1)]
    backend._sigmas = [1.0]
    backend._encode_prompt = lambda _: SimpleNamespace(
        video_encoding=None, audio_encoding=None, attention_mask=None
    )
    backend._empty_action_condition = lambda _: {}
    video = torch.zeros(1, 4, 2)
    audio = torch.zeros(1, 5, 2)
    monkeypatch.setattr(module, "combined_image_conditionings", lambda *a: [])
    monkeypatch.setattr(module, "GaussianNoiser", lambda generator: generator)
    monkeypatch.setattr(module, "AudioLatentTools", lambda *a: None)

    def video_setup(shape, generator, *args):
        torch.randn(77, generator=generator)
        return SimpleNamespace(clean_latent=video, positions=None), None

    def audio_setup(tools, conditioning, generator, *args):
        torch.randn(31, generator=generator)
        return SimpleNamespace(clean_latent=audio, positions=None)

    monkeypatch.setattr(module, "noise_video_state", video_setup)
    monkeypatch.setattr(module, "create_noised_state", audio_setup)
    monkeypatch.setattr(module._BlockForward, "create", lambda **kwargs: object())
    monkeypatch.setattr(module, "_generate_audio_prefix", lambda *args: None)
    sentinel = object()
    backend._forward = sentinel
    for invalid in (True, False, 1.5, 0, 513):
        with pytest.raises(ValueError, match="integer"):
            backend.reset(
                image=Path("image.png"),
                prompt="scene",
                seed=42,
                fov_degrees=70,
                chunks=invalid,
            )
        assert backend._forward is sentinel
    backend.reset(
        image=Path("image.png"), prompt="scene", seed=42, fov_degrees=70, chunks=1
    )
    assert backend._total_pixel_frames == 25
    assert backend._total_video_latents == 4
    expected = module._RolloutBuffers.create(
        video, audio, 1, torch.Generator(device="cpu").manual_seed(42)
    )
    torch.testing.assert_close(
        backend._buffers.initial_video, expected.initial_video, rtol=0, atol=0
    )
    torch.testing.assert_close(
        backend._buffers.initial_audio, expected.initial_audio, rtol=0, atol=0
    )
