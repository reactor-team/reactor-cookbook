# Vendored: jd-opensource/JoyAI-Video-Edit (`deploy/joyomni_ops`)

- Source:  https://github.com/jd-opensource/JoyAI-Video-Edit
- Commit:  ca17e1d1030f454cb98b0ed549b4d31a60139ceb (`deploy/joyomni_ops`)
- License: Apache-2.0 (`LICENSE` in this directory). The kernels derived from
  sgl-kernel, TensorRT-LLM and CUTLASS keep their original copyright headers.

`joyomni_ops` is upstream's small CUDA operator library for the DiT: fused
QK-norm with 3D RoPE, fused norm with adaLN scale/shift, RMSNorm, per-token
FP8 quantisation, and the FP8 scaled GEMM. The Dockerfile builds it from this
directory against CUTLASS v4.7.1.

Added:

- `csrc/fused_qknorm_rope_3d_to_kernel.cu`: QK-norm and 3D RoPE reading q and
  k from strided rows of the fused QKV projection and writing them straight
  into the attention buffers.
- `csrc/masked_attn_rescale.cu`: the output rescale of the padded-text
  attention.
- `csrc/quant_fp8_fused.cu`: per-token FP8 quantisation with its two
  producers fused into it.

Changed:

- `csrc/fused_norm_scale_shift.cu` reads one broadcast scale/shift row
  instead of an expanded per-token copy.
- `csrc/pybind.cpp`, `joyomni_ops/__init__.py`, and `setup.py` register the
  new ops; `setup.py` takes the architecture list from
  `JOYOMNI_OPS_CUDA_ARCHS` (the Dockerfile builds `90;100f`, one binary for
  sm_100 and sm_103).
- `README.md` keeps the build notes (architecture list, reference CUTLASS
  revision) of an earlier upstream revision.
