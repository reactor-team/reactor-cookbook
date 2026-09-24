# Attributions

`cosmos_framework/` in this directory is a pruned subset of:

- **Source repository:** https://github.com/NVIDIA/cosmos-framework
- **Commit:** `cf5d68c` (`Release 2026-09-23`)
- **License:** OpenMDW-1.1 (OpenMDW License Agreement, version 1.1). See
  `LICENSE.cosmos-framework`, copied verbatim from upstream's `LICENSE`.
- **Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
  rights reserved. Every copied source file carries its original upstream
  SPDX header, unmodified.
- **Third-party notices:** `NOTICE.cosmos-framework` is upstream's `NOTICE`,
  verbatim. Upstream's full third-party attribution list is
  https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c/ATTRIBUTIONS.md.

See `VENDOR.md` for exactly which files were kept and how the keep-set was
derived.

## Model weights

The checkpoints this model loads are published by NVIDIA under the OpenMDW-1.1
license on Hugging Face and are downloaded at first load, never checked in:

- https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID
- https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B (the `Wan2.2_VAE.pth` video
  tokenizer, Apache-2.0)
