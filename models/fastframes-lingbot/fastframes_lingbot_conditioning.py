"""Encode an image and the constant zero tail used by Apple's LingBot recipe."""

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


@torch.inference_mode()
def anchor_condition(vae, image: Image.Image, latents: int, device: torch.device) -> torch.Tensor:
    """Run the pinned causal VAE on one image followed by zero-valued video."""
    image_tensor = to_tensor(image).sub_(0.5).div_(0.5).to(device)
    first = image_tensor[None, :, None]
    black = image_tensor.new_zeros((1, 3, 4, image.height, image.width))
    model = vae.model
    model.clear_cache()
    cache = model._enc_feat_map
    try:
        with torch.autocast("cuda", dtype=vae.dtype, enabled=vae.dtype != torch.float32):
            features = [model.encoder(first, feat_cache=cache, feat_idx=[0])]
            for index in range(1, latents):
                previous = tuple(cache) if index % 8 == 0 else ()
                feature = model.encoder(black, feat_cache=cache, feat_idx=[0])
                # With identical cache state and a constant input, every later
                # encoder step has the same input and returns the same feature.
                if previous and all(
                    old is new or (old is not None and new is not None and torch.equal(old, new))
                    for old, new in zip(previous, cache, strict=True)
                ):
                    features.append(feature.expand(-1, -1, latents - index, -1, -1))
                    break
                features.append(feature)
            mu = model.conv1(torch.cat(features, dim=2)).chunk(2, dim=1)[0]
            mean, inverse_std = vae.scale
            encoded = ((mu - mean.view(1, -1, 1, 1, 1)) * inverse_std.view(1, -1, 1, 1, 1)).float()[0]
    finally:
        model.clear_cache()
    mask = encoded.new_zeros((4, latents, image.height // 8, image.width // 8))
    mask[:, 0] = 1
    return torch.cat((mask, encoded))
