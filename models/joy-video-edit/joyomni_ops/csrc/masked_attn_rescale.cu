/*
 * Output rescale of the padded-text attention (xvideo/models/dit/dit.py `_attention`).
 *
 * Padding keys and values are zeroed before the attention call, so each padding key adds exp(0)
 * to a row's softmax denominator and nothing to its numerator. The exact output is recovered as
 *   y[b, s, h, :] = bf16( float(o[b, h, s, :]) * 1 / (1 - P_b * exp(-lse[b, h, s])) )
 * with P_b the number of padding keys. One pass: reads o and lse, writes y [B, S, H, D]
 * contiguous. Each float operation is rounded on its own, in the order the unfused tensor
 * expression evaluates it (exp, then P * e, then 1 - that, then the reciprocal, then the product),
 * so the result equals that expression bit for bit.
 *
 *   o:        [B, H, S, D] bf16, any strides with stride(D) == 1
 *   lse:      [B, H, S] (or [B, H, S, 1]) fp32, any strides
 *   text_pad: [B] integer (padding keys per batch row)
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

constexpr int kVec = 8;  // bf16 per 16-byte access

// One block per (batch, token) row of y; thread t handles the 8 values [h, d0:d0+8) of that row.
__global__ void maskedAttnRescaleKernel(
    const __nv_bfloat16* __restrict__ o, const float* __restrict__ lse, const float* __restrict__ pad,
    __nv_bfloat16* __restrict__ y, int H, int S, int D,
    int64_t ob, int64_t oh, int64_t os, int64_t lb, int64_t lh, int64_t ls) {
  const int row = blockIdx.x;  // b * S + s
  const int b = row / S, s = row - b * S;
  const int vec_per_head = D / kVec;
  for (int t = threadIdx.x; t < H * vec_per_head; t += blockDim.x) {
    const int h = t / vec_per_head;
    const int d = (t - h * vec_per_head) * kVec;
    const float l = lse[b * lb + h * lh + s * ls];
    const float ex = expf(-l);
    const float p_exp = __fmul_rn(pad[b], ex);
    const float den = __fsub_rn(1.0f, p_exp);
    const float scale = __fdiv_rn(1.0f, den);
    const uint4 raw = *reinterpret_cast<const uint4*>(o + b * ob + h * oh + s * os + d);
    const __nv_bfloat16* xin = reinterpret_cast<const __nv_bfloat16*>(&raw);
    uint4 packed;
    __nv_bfloat16* xout = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int j = 0; j < kVec; ++j) xout[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(xin[j]), scale));
    *reinterpret_cast<uint4*>(y + ((int64_t)row * H + h) * D + d) = packed;
  }
}

}  // namespace

torch::Tensor masked_attn_rescale(const torch::Tensor& o, const torch::Tensor& lse, const torch::Tensor& text_pad) {
  JO_CHECK_CUDA(o);
  JO_CHECK_CUDA(lse);
  JO_CHECK_CUDA(text_pad);
  TORCH_CHECK(o.dim() == 4, "o must be [B, H, S, D]");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bf16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be fp32");
  const int64_t B = o.size(0), H = o.size(1), S = o.size(2), D = o.size(3);
  TORCH_CHECK(o.stride(3) == 1 && D % kVec == 0, "o needs a contiguous last dim divisible by 8");
  TORCH_CHECK(o.stride(0) % kVec == 0 && o.stride(1) % kVec == 0 && o.stride(2) % kVec == 0 &&
                  reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0,
              "o must be 16-byte aligned per row");
  torch::Tensor l = lse.dim() == 4 ? lse.squeeze(-1) : lse;
  TORCH_CHECK(l.dim() == 3 && l.size(0) == B && l.size(1) == H && l.size(2) == S, "lse must be [B, H, S]");
  TORCH_CHECK(text_pad.numel() == B, "text_pad must have B elements");
  const c10::cuda::CUDAGuard guard(o.device());
  // The count of padding keys as float, as the unfused expression converts it.
  auto pad = text_pad.reshape({B}).to(torch::kFloat32).contiguous();
  auto y = torch::empty({B, S, H, D}, o.options());
  if (B * S == 0) return y;
  const int threads = (int)std::min<int64_t>(1024, (H * D / kVec + 31) / 32 * 32);
  maskedAttnRescaleKernel<<<(unsigned)(B * S), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()), l.data_ptr<float>(), pad.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), (int)H, (int)S, (int)D,
      o.stride(0), o.stride(1), o.stride(2), l.stride(0), l.stride(1), l.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace joyomni_ops
