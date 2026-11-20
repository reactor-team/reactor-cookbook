"""Shard independent native VAE spatial tiles across ranks and stitch in native order.

The tested profile batches equally shaped tiles and leaves compilation disabled.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def install(vae):
    """Shard spatial tiles over the Runtime worker group, batching each rank's tiles."""
    world = dist.get_world_size()
    rank = dist.get_rank()

    original = vae._decode_clip

    def sharded_decode_clip(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return original(z)

        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        ratio = vae.spatial_compression_ratio

        coords = [
            (y_indices[i], y_lengths[i], x_indices[j], x_lengths[j])
            for i in range(len(y_indices))
            for j in range(len(x_indices))
        ]
        # Contiguous blocks, not round-robin: `all_gather` concatenates rank 0's block, then rank
        # 1's, and so on, so contiguous assignment is what puts the tiles back in tile order for
        # free. The list is padded up to an equal count per rank so the gather stays a plain
        # fixed-size one; the padding entries are recomputed duplicates of the last tile, at most
        # world-1 of them, and are dropped after the gather.
        per_rank = (len(coords) + world - 1) // world
        padded = coords + [coords[-1]] * (per_rank * world - len(coords))
        mine = padded[rank * per_rank : (rank + 1) * per_rank]

        slices = [
            z[
                ...,
                y // ratio : y // ratio + y_len // ratio,
                x // ratio : x // ratio + x_len // ratio,
            ]
            for y, y_len, x, x_len in mine
        ]
        local_stack = vae.decoder(vae.post_quant_conv(torch.cat(slices, dim=0)))

        gathered = [torch.empty_like(local_stack) for _ in range(world)]
        dist.all_gather(gathered, local_stack)
        flat = torch.cat(gathered, dim=0)[: len(coords)]

        # `_stitch_tiles` only touches dims -2/-1, so a missing batch axis would not raise — it
        # would return `(3, T, H, W)` and `_decode` would then slice the height axis where it means
        # to slice frames. Restore the axis explicitly.
        rows, index = [], 0
        for _ in range(len(y_indices)):
            row = []
            for _ in range(len(x_indices)):
                row.append(flat[index].unsqueeze(0))
                index += 1
            rows.append(row)
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    vae._decode_clip = sharded_decode_clip
