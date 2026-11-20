/*
 * Fused QK-Norm + 3D RoPE reading q/k from strided rows (e.g. the column slices of a fused QKV
 * projection output) and writing the result to separate destination rows (e.g. the attention
 * operand buffers). Per head, the arithmetic is the in-place kernel's
 * (fused_qknorm_rope_3d_kernel.cu, same warp layout, same reduction), so the result is the same
 * bits as copying q/k into the destination and running that kernel there.
 *
 * Derived from the JoyOmni sgl-kernel fork (Apache-2.0).
 * Copyright (c) 2025, NVIDIA CORPORATION. Licensed under the Apache License 2.0.
 *
 * q_src/k_src: [batch, seq_len, num_heads, head_dim] bf16, head_dim contiguous, heads packed
 *              (stride head_dim), any batch/token stride.
 * q_dst/k_dst: same shape, same constraints (typically contiguous).
 * cos/sin: [seq_len, head_dim/2] bf16.  weight: [head_dim] bf16.
 */
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

constexpr unsigned kFullMaskTo = 0xffffffffu;

template <typename T, int num>
struct packed_as_to;
template <>
struct packed_as_to<unsigned, 1> { using type = unsigned; };
template <>
struct packed_as_to<unsigned, 2> { using type = uint2; };
template <>
struct packed_as_to<unsigned, 4> { using type = uint4; };

__inline__ __device__ float warpReduceSumTo(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(kFullMaskTo, val, mask, 32);
  return val;
}

// One warp normalizes+rotates one head. Q and K are packed into a single grid:
// warps [0, N) handle Q, warps [N, 2N) handle K, where N = batch*seq*heads.
template <int head_dim>
__global__ void fusedQKNormRope3DPairedTo(
    __nv_bfloat16 const* __restrict__ q_src,
    __nv_bfloat16 const* __restrict__ k_src,
    __nv_bfloat16* __restrict__ q_dst,
    __nv_bfloat16* __restrict__ k_dst,
    int64_t const src_bstride, int64_t const src_tstride,
    int64_t const dst_bstride, int64_t const dst_tstride,
    int const batch_size,
    int const seq_len,
    int const num_heads,
    float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,
    __nv_bfloat16 const* __restrict__ k_weight,
    __nv_bfloat16 const* __restrict__ cos_ptr,
    __nv_bfloat16 const* __restrict__ sin_ptr) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

  int const total_qk_heads = batch_size * seq_len * num_heads * 2;
  if (globalWarpIdx >= total_qk_heads) return;

  int const is_k = globalWarpIdx / (batch_size * seq_len * num_heads);
  int const local_idx = globalWarpIdx % (batch_size * seq_len * num_heads);
  int const batch_idx = local_idx / (seq_len * num_heads);
  int const remaining = local_idx % (seq_len * num_heads);
  int const token_idx = remaining / num_heads;
  int const head_idx = remaining % num_heads;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as_to<unsigned, vecSize>::type;

  __nv_bfloat16 const* src = is_k ? k_src : q_src;
  __nv_bfloat16* dst = is_k ? k_dst : q_dst;
  __nv_bfloat16 const* weight = is_k ? k_weight : q_weight;

  int64_t const srcThread = batch_idx * src_bstride + token_idx * src_tstride +
                            head_idx * head_dim + laneId * numElemsPerThread;
  int64_t const dstThread = batch_idx * dst_bstride + token_idx * dst_tstride +
                            head_idx * head_dim + laneId * numElemsPerThread;

  float sumOfSquares = 0.0f;
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&src[srcThread]);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<unsigned*>(&vec) + i));
      sumOfSquares += vals.x * vals.x;
      sumOfSquares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  sumOfSquares = warpReduceSumTo(sumOfSquares);
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    elements[i] *= rms_rcp * __bfloat162float(weight[dim]);
  }

  int const cos_sin_row_offset = token_idx * (head_dim / 2);
  float rotated[numElemsPerThread];
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i += 2) {
    int dim_idx = laneId * numElemsPerThread + i;
    int cos_sin_idx = dim_idx / 2;
    float c = __bfloat162float(cos_ptr[cos_sin_row_offset + cos_sin_idx]);
    float s = __bfloat162float(sin_ptr[cos_sin_row_offset + cos_sin_idx]);
    float x1 = elements[i];
    float x2 = elements[i + 1];
    rotated[i] = x1 * c - x2 * s;
    rotated[i + 1] = x1 * s + x2 * c;
  }

  {
    vec_T vec;
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(rotated[2 * i], rotated[2 * i + 1]));
      reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<unsigned*>(&vec) + i)) = vals;
    }
    *reinterpret_cast<vec_T*>(&dst[dstThread]) = vec;
  }
}

}  // namespace

void fused_qk_norm_rope_3d_paired_to(
    const torch::Tensor& q_src, const torch::Tensor& k_src, torch::Tensor& q_dst, torch::Tensor& k_dst,
    double eps, const torch::Tensor& q_weight, const torch::Tensor& k_weight,
    const torch::Tensor& cos, const torch::Tensor& sin) {
  const torch::Tensor* all[4] = {&q_src, &k_src, &q_dst, &k_dst};
  for (const torch::Tensor* t : all) {
    TORCH_CHECK(t->dim() == 4, "q/k tensors must be 4D [batch, seq_len, num_heads, head_dim]");
    TORCH_CHECK(t->scalar_type() == torch::kBFloat16, "q/k tensors must be bf16");
    TORCH_CHECK(t->is_cuda(), "q/k tensors must be CUDA");
    TORCH_CHECK(t->sizes() == q_src.sizes(), "q/k tensors must have the same shape");
    TORCH_CHECK(t->stride(3) == 1 && t->stride(2) == t->size(3), "heads must be packed, head_dim contiguous");
    TORCH_CHECK(t->stride(1) % 8 == 0 && t->stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "q/k rows must be 16-byte aligned");
  }
  // One (batch, token) stride pair per side: q and k must agree on it (a batch of one has no batch stride).
  const bool one = q_src.size(0) == 1;
  TORCH_CHECK(q_src.stride(1) == k_src.stride(1) && (one || q_src.stride(0) == k_src.stride(0)), "q_src/k_src strides must match");
  TORCH_CHECK(q_dst.stride(1) == k_dst.stride(1) && (one || q_dst.stride(0) == k_dst.stride(0)), "q_dst/k_dst strides must match");
  JO_CHECK_INPUT(q_weight, torch::kBFloat16);
  JO_CHECK_INPUT(k_weight, torch::kBFloat16);
  JO_CHECK_INPUT(cos, torch::kBFloat16);
  JO_CHECK_INPUT(sin, torch::kBFloat16);
  const int64_t batch_size = q_src.size(0), seq_len = q_src.size(1), num_heads = q_src.size(2), head_dim = q_src.size(3);
  TORCH_CHECK(cos.dim() == 2 && cos.size(0) == seq_len && cos.size(1) == head_dim / 2, "cos shape must be [seq_len, head_dim/2]");
  TORCH_CHECK(sin.dim() == 2 && sin.size(0) == seq_len && sin.size(1) == head_dim / 2, "sin shape must be [seq_len, head_dim/2]");
  TORCH_CHECK(q_weight.numel() == head_dim && k_weight.numel() == head_dim, "weight size must match head_dim");

  const c10::cuda::CUDAGuard guard(q_src.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_src.get_device());
  constexpr int blockSize = 256;
  const int64_t totalQKHeads = batch_size * seq_len * num_heads * 2;
  dim3 grid((unsigned)((totalQKHeads + blockSize / 32 - 1) / (blockSize / 32)));
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16 const*>(t.data_ptr()); };
#define LAUNCH_TO(HD)                                                                                  \
  fusedQKNormRope3DPairedTo<HD><<<grid, blockSize, 0, stream>>>(                                        \
      bf(q_src), bf(k_src), reinterpret_cast<__nv_bfloat16*>(q_dst.data_ptr()),                         \
      reinterpret_cast<__nv_bfloat16*>(k_dst.data_ptr()), q_src.stride(0), q_src.stride(1),             \
      q_dst.stride(0), q_dst.stride(1), (int)batch_size, (int)seq_len, (int)num_heads, (float)eps,       \
      bf(q_weight), bf(k_weight), bf(cos), bf(sin));                                                    \
  break
  switch (head_dim) {
    case 64: LAUNCH_TO(64);
    case 128: LAUNCH_TO(128);
    case 256: LAUNCH_TO(256);
    default: TORCH_CHECK(false, "Unsupported head_dim for fused_qk_norm_rope_3d_paired_to: ", head_dim);
  }
#undef LAUNCH_TO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace joyomni_ops
