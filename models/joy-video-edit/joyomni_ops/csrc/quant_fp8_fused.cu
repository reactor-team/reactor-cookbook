/*
 * Row-in-registers FP8 (e4m3) per-token quantization, plus two producers fused into it.
 * Each op is bit-identical to the unfused sequence it replaces:
 *
 *   per_token_quant_fp8_v2(x, q, s)            == sgl_per_token_quant_fp8(x, q, s)
 *   fused_norm_scale_shift_fp8(x, g, b, sc, sh, norm_type, eps) -> (q, s)
 *                                              == sgl_per_token_quant_fp8(fused_norm_scale_shift(...))
 *   gelu_tanh_quant_fp8(x) -> (q, s)           == sgl_per_token_quant_fp8(F.gelu(x, approximate="tanh"))
 *
 * Each row is loaded once into registers with 16-byte loads, its absmax is reduced, and it is
 * quantized from registers with 16-byte stores. Quant / GELU: one block per row. LayerNorm at
 * N == 4096: one warp per row (see layerNormFp8WarpRow4096); other norm shapes: the original
 * block-per-row layout.
 * The quantization arithmetic is the same expression sequence as per_token_quant_fp8.cu:
 *   amax = max|float(x)|; scale = amax / 448; inv = scale == 0 ? 0 : 1 / scale;
 *   q = fp8(clamp(float(x) * inv, -448, 448)).
 * fmaxf is order-independent, so the reduction layout does not affect the scale.
 *
 * The fused producers compute each element exactly as the op they replace, round it to bf16
 * (the tensor the unfused path materialises), and quantize those bf16 values:
 *   - LayerNorm/RMSNorm + scale/shift: the arithmetic of normScaleShiftPerRow
 *     (fused_norm_scale_shift.cu) with its exact float summation order for mean / variance.
 *   - GELU(tanh): ATen's CUDA tanh-GELU expression in float (ActivationGeluKernel.cu).
 *
 * Unspecialised shapes / dtypes fall back to the unfused ops, so every input is handled.
 * Build without --use_fast_math: bit identity depends on IEEE division and accurate tanhf.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <tuple>

#include "joyomni_ops.h"

namespace joyomni_ops {

// Defined in per_token_quant_fp8.cu / fused_norm_scale_shift.cu; used as fallbacks.
void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s);
torch::Tensor fused_norm_scale_shift(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps);

namespace qfused {

constexpr float kFp8Max = 448.0f;

// ---------------------------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ __nv_fp8_e4m3 quant_one(float x, float scale_inv) {
  float val = x * scale_inv;
  val = fmaxf(fminf(val, kFp8Max), -kFp8Max);
  return static_cast<__nv_fp8_e4m3>(val);
}

// Block-wide max, result broadcast to every thread. Values are >= 0.
__device__ __forceinline__ float block_max_all(float v) {
  __shared__ float sh_max[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
  const int nwarps = (blockDim.x + 31) >> 5;
  if (nwarps == 1) return v;
  if ((threadIdx.x & 31) == 0) sh_max[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = sh_max[0];
  for (int w = 1; w < nwarps; ++w) r = fmaxf(r, sh_max[w]);
  return r;
}

// ATen tanh-GELU (GeluType::Tanh) for opmath_t = float: identical constants and expression.
__device__ __forceinline__ float gelu_tanh_f32(float x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * float(0.5);
  constexpr float kKappa = 0.044715;
  const float x_cube = x * x * x;
  const float inner = kBeta * (x + kKappa * x_cube);
  return float(0.5) * x * (float(1) + tanhf(inner));
}

// ---------------------------------------------------------------------------------------------
// (1) + (3): one block per row, row held in registers.
//   Each thread owns VPT groups of 16 consecutive elements (two 16 B loads, one 16 B store).
//   K == blockDim.x * VPT * 16.
// ---------------------------------------------------------------------------------------------

template <int VPT, bool kGelu>
__global__ void rowQuantRegKernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_fp8_e4m3* __restrict__ out_q, float* __restrict__ out_s,
                                  const int64_t hidden_dim) {
  const int64_t row = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(input + row * hidden_dim);
  uint4* dst = reinterpret_cast<uint4*>(out_q + row * hidden_dim);
  const int nthr = blockDim.x;

  uint4 v[VPT][2];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int g = i * nthr + threadIdx.x;
    v[i][0] = __ldg(src + 2 * g);
    v[i][1] = __ldg(src + 2 * g + 1);
  }

  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v[i][h]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (kGelu) e[j] = __nv_bfloat16(gelu_tanh_f32(static_cast<float>(e[j])));
        amax = fmaxf(amax, fabsf(static_cast<float>(e[j])));
      }
    }
  }

  amax = block_max_all(amax);
  const float scale = amax / kFp8Max;
  if (threadIdx.x == 0) out_s[row] = scale;
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    union { __nv_fp8_e4m3 b[16]; uint4 u; } o;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v[i][h]);
#pragma unroll
      for (int j = 0; j < 8; ++j) o.b[h * 8 + j] = quant_one(static_cast<float>(e[j]), scale_inv);
    }
    dst[i * nthr + threadIdx.x] = o.u;
  }
}

// Launch geometry: groups of 16 elements per thread. Measured on B200 (K = 4096 / 16384,
// M = 640..3120): plain quant is fastest at 4 groups per thread (memory-bound, wants bytes in
// flight per thread); GELU at 2 (bound by the accurate tanhf, wants more threads).
// Returns false (caller falls back) when hidden_dim has no warp-multiple factorisation.
inline bool pick_rowreg_config(int64_t hidden_dim, int* vpt, int* threads, bool gelu = false) {
  if (hidden_dim % 16 != 0) return false;
  const int64_t groups = hidden_dim / 16;
  constexpr int kQuantOrder[8] = {4, 2, 8, 1, 6, 3, 5, 7};
  constexpr int kGeluOrder[8] = {2, 4, 1, 8, 3, 6, 5, 7};
  for (int c : (gelu ? kGeluOrder : kQuantOrder)) {
    if (groups % c) continue;
    const int64_t t = groups / c;
    if (t % 32 == 0 && t >= 32 && t <= 1024) { *vpt = c; *threads = (int)t; return true; }
  }
  return false;
}

template <bool kGelu>
bool launch_rowreg(const __nv_bfloat16* in, __nv_fp8_e4m3* q, float* s, int64_t M, int64_t K,
                   int vpt, int threads, cudaStream_t stream) {
  if (M == 0) return true;
  dim3 grid((unsigned)M), block((unsigned)threads);
  switch (vpt) {
#define JO_CASE(V) \
  case V: rowQuantRegKernel<V, kGelu><<<grid, block, 0, stream>>>(in, q, s, K); break;
    JO_CASE(1) JO_CASE(2) JO_CASE(3) JO_CASE(4) JO_CASE(5) JO_CASE(6) JO_CASE(7) JO_CASE(8)
#undef JO_CASE
    default: return false;
  }
  return true;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// ---------------------------------------------------------------------------------------------
// (2): normScaleShiftPerRow (fused_norm_scale_shift.cu), bf16 only, storing FP8 + row scale.
//   Everything up to the rounded bf16 output is the original kernel verbatim.
// ---------------------------------------------------------------------------------------------

enum NormType : int { kLayerNorm = 0, kRMSNorm = 1 };

template <typename T, int NumVals>
__device__ __forceinline__ void warpReduceSum(T (&vals)[NumVals]) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
#pragma unroll
    for (int i = 0; i < NumVals; ++i) vals[i] += __shfl_down_sync(mask, vals[i], offset);
}

template <typename T, int NumVals>
__device__ __forceinline__ void blockReduceSum(T (&vals)[NumVals]) {
  __shared__ T shared[32][NumVals];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  warpReduceSum<T, NumVals>(vals);
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < NumVals; ++i) shared[wid][i] = vals[i];
  __syncthreads();
  if (wid == 0) {
    T acc[NumVals];
#pragma unroll
    for (int i = 0; i < NumVals; ++i) acc[i] = T(0);
    int num_warps = (blockDim.x + 31) / 32;
#pragma unroll
    for (int w = 0; w < 32; ++w)
      if (w < num_warps)
#pragma unroll
        for (int i = 0; i < NumVals; ++i) acc[i] += shared[w][i];
#pragma unroll
    for (int i = 0; i < NumVals; ++i) vals[i] = acc[i];
  }
  __syncthreads();
}

struct alignas(8) bf16_4 { __nv_bfloat16 x, y, z, w; };

template <typename T4, typename T, int ITEM_PER_THREAD, int norm_type>
__global__ void normScaleShiftFp8PerRow(
    __nv_fp8_e4m3* __restrict__ out_q, float* __restrict__ out_s, const T4* input, const T4* gamma,
    const T4* beta, const T4* scale, const T4* shift, const int n, const int mod_row_stride4,
    bool affine, float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;
  input += offset;
  scale += (int64_t)m_idx * mod_row_stride4;
  shift += (int64_t)m_idx * mod_row_stride4;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    if constexpr (norm_type == kLayerNorm) {
      local_sums[0] += float(local_val[i].x) + float(local_val[i].y) + float(local_val[i].z) + float(local_val[i].w);
    } else {
      local_sums[0] += float(local_val[i].x) * float(local_val[i].x) + float(local_val[i].y) * float(local_val[i].y) +
                       float(local_val[i].z) * float(local_val[i].z) + float(local_val[i].w) * float(local_val[i].w);
    }
  }
  if (blockDim.x <= 32) warpReduceSum<float, 1>(local_sums);
  else blockReduceSum<float, 1>(local_sums);
  if (tid == 0) s_mean = local_sums[0] / n;
  __syncthreads();

  if constexpr (norm_type == kLayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        float4 t = {float(local_val[i].x) - s_mean, float(local_val[i].y) - s_mean,
                    float(local_val[i].z) - s_mean, float(local_val[i].w) - s_mean};
        local_sums[0] += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
      }
    }
    if (blockDim.x <= 32) warpReduceSum<float, 1>(local_sums);
    else blockReduceSum<float, 1>(local_sums);
  }
  if (tid == 0) s_variance = rsqrtf(local_sums[0] / n + eps);
  __syncthreads();

  T4 outv[ITEM_PER_THREAD];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index >= n_4) continue;
    const T4 g = affine ? gamma[index] : T4{T(1.f), T(1.f), T(1.f), T(1.f)};
    const T4 sc = scale[index];
    const T4 sh = shift[index];
    T4 out;
    if constexpr (norm_type == kLayerNorm) {
      const T4 b = affine ? beta[index] : T4{T(0.f), T(0.f), T(0.f), T(0.f)};
      out.x = T(((float(local_val[i].x) - s_mean) * s_variance * float(g.x) + float(b.x)) * (1.f + float(sc.x)) + float(sh.x));
      out.y = T(((float(local_val[i].y) - s_mean) * s_variance * float(g.y) + float(b.y)) * (1.f + float(sc.y)) + float(sh.y));
      out.z = T(((float(local_val[i].z) - s_mean) * s_variance * float(g.z) + float(b.z)) * (1.f + float(sc.z)) + float(sh.z));
      out.w = T(((float(local_val[i].w) - s_mean) * s_variance * float(g.w) + float(b.w)) * (1.f + float(sc.w)) + float(sh.w));
    } else {
      out.x = T((float(local_val[i].x) * s_variance * float(g.x)) * (1.f + float(sc.x)) + float(sh.x));
      out.y = T((float(local_val[i].y) * s_variance * float(g.y)) * (1.f + float(sc.y)) + float(sh.y));
      out.z = T((float(local_val[i].z) * s_variance * float(g.z)) * (1.f + float(sc.z)) + float(sh.z));
      out.w = T((float(local_val[i].w) * s_variance * float(g.w)) * (1.f + float(sc.w)) + float(sh.w));
    }
    outv[i] = out;
    amax = fmaxf(amax, fabsf(static_cast<float>(out.x)));
    amax = fmaxf(amax, fabsf(static_cast<float>(out.y)));
    amax = fmaxf(amax, fabsf(static_cast<float>(out.z)));
    amax = fmaxf(amax, fabsf(static_cast<float>(out.w)));
  }

  amax = block_max_all(amax);
  const float qscale = amax / kFp8Max;
  if (tid == 0) out_s[m_idx] = qscale;
  const float scale_inv = (qscale == 0.f) ? 0.f : 1.0f / qscale;

  uint32_t* qrow = reinterpret_cast<uint32_t*>(out_q + (int64_t)m_idx * n);
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index >= n_4) continue;
    union { __nv_fp8_e4m3 b[4]; uint32_t u; } o;
    o.b[0] = quant_one(static_cast<float>(outv[i].x), scale_inv);
    o.b[1] = quant_one(static_cast<float>(outv[i].y), scale_inv);
    o.b[2] = quant_one(static_cast<float>(outv[i].z), scale_inv);
    o.b[3] = quant_one(static_cast<float>(outv[i].w), scale_inv);
    qrow[index] = o.u;
  }
}

// LayerNorm at N == 4096, one WARP per row. fused_norm_scale_shift runs this shape as a
// 1024-thread block (one bf16x4 per thread). Lane L here reproduces warp L of that block:
// it owns elements [128 L, 128 L + 128), forms the same 32 per-thread partial sums, applies the
// same __shfl_down_sync tree (offsets 16..1) serially, and the 32 warp sums are then added in
// warp order starting from 0.0f exactly as blockReduceSum's warp 0 does. Every float operation
// and its operand order is therefore the original's; only the thread mapping differs.
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) layerNormFp8WarpRow4096(
    __nv_fp8_e4m3* __restrict__ out_q, float* __restrict__ out_s, const bf16_4* __restrict__ input,
    const bf16_4* __restrict__ gamma, const bf16_4* __restrict__ beta, const bf16_4* __restrict__ scale,
    const bf16_4* __restrict__ shift, const int n, const int mod_row_stride4, bool affine, float eps,
    const int M) {
  using T = __nv_bfloat16;
  using T4 = bf16_4;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (row >= M) return;  // warp-uniform
  const int base = lane * 32;  // first bf16x4 index (== original thread id) owned by this lane

  union { uint4 u[16]; T4 v[32]; } xv;
  const uint4* src = reinterpret_cast<const uint4*>(input + (int64_t)row * (n / 4) + base);
#pragma unroll
  for (int j = 0; j < 16; ++j) xv.u[j] = __ldg(src + j);

  float p[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    float local_sum = 0.0f;
    local_sum += float(xv.v[k].x) + float(xv.v[k].y) + float(xv.v[k].z) + float(xv.v[k].w);
    p[k] = local_sum;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int i = 0; i < off; ++i) p[i] += p[i + off];
  float acc = 0.0f;
#pragma unroll
  for (int w = 0; w < 32; ++w) acc += __shfl_sync(0xffffffffu, p[0], w);
  const float s_mean = acc / n;

#pragma unroll
  for (int k = 0; k < 32; ++k) {
    float local_sum = 0.0f;
    float4 t = {float(xv.v[k].x) - s_mean, float(xv.v[k].y) - s_mean,
                float(xv.v[k].z) - s_mean, float(xv.v[k].w) - s_mean};
    local_sum += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    p[k] = local_sum;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int i = 0; i < off; ++i) p[i] += p[i + off];
  acc = 0.0f;
#pragma unroll
  for (int w = 0; w < 32; ++w) acc += __shfl_sync(0xffffffffu, p[0], w);
  const float s_variance = rsqrtf(acc / n + eps);

  const uint4* scp = reinterpret_cast<const uint4*>(scale + (int64_t)row * mod_row_stride4 + base);
  const uint4* shp = reinterpret_cast<const uint4*>(shift + (int64_t)row * mod_row_stride4 + base);
  const uint4* gp = reinterpret_cast<const uint4*>(gamma + base);
  const uint4* bp = reinterpret_cast<const uint4*>(beta + base);
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    union { uint4 u; T4 v[2]; } sc4, sh4, g4, b4;
    sc4.u = __ldg(scp + j);
    sh4.u = __ldg(shp + j);
    if (affine) { g4.u = __ldg(gp + j); b4.u = __ldg(bp + j); }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k = 2 * j + h;
      const T4 g = affine ? g4.v[h] : T4{T(1.f), T(1.f), T(1.f), T(1.f)};
      const T4 b = affine ? b4.v[h] : T4{T(0.f), T(0.f), T(0.f), T(0.f)};
      const T4 sc = sc4.v[h];
      const T4 sh = sh4.v[h];
      const T4 lv = xv.v[k];
      T4 out;
      out.x = T(((float(lv.x) - s_mean) * s_variance * float(g.x) + float(b.x)) * (1.f + float(sc.x)) + float(sh.x));
      out.y = T(((float(lv.y) - s_mean) * s_variance * float(g.y) + float(b.y)) * (1.f + float(sc.y)) + float(sh.y));
      out.z = T(((float(lv.z) - s_mean) * s_variance * float(g.z) + float(b.z)) * (1.f + float(sc.z)) + float(sh.z));
      out.w = T(((float(lv.w) - s_mean) * s_variance * float(g.w) + float(b.w)) * (1.f + float(sc.w)) + float(sh.w));
      xv.v[k] = out;
      amax = fmaxf(amax, fabsf(static_cast<float>(out.x)));
      amax = fmaxf(amax, fabsf(static_cast<float>(out.y)));
      amax = fmaxf(amax, fabsf(static_cast<float>(out.z)));
      amax = fmaxf(amax, fabsf(static_cast<float>(out.w)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
  const float qscale = amax / kFp8Max;
  if (lane == 0) out_s[row] = qscale;
  const float scale_inv = (qscale == 0.f) ? 0.f : 1.0f / qscale;

  uint4* dst = reinterpret_cast<uint4*>(out_q + (int64_t)row * n + (int64_t)base * 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    union { __nv_fp8_e4m3 b[16]; uint4 u; } o;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const T4 v4 = xv.v[4 * j + e];
      o.b[4 * e + 0] = quant_one(static_cast<float>(v4.x), scale_inv);
      o.b[4 * e + 1] = quant_one(static_cast<float>(v4.y), scale_inv);
      o.b[4 * e + 2] = quant_one(static_cast<float>(v4.z), scale_inv);
      o.b[4 * e + 3] = quant_one(static_cast<float>(v4.w), scale_inv);
    }
    dst[j] = o.u;
  }
}

inline torch::TensorOptions fp8_opts(const torch::Tensor& x) {
  return x.options().dtype(at::kFloat8_e4m3fn);
}
inline torch::TensorOptions f32_opts(const torch::Tensor& x) {
  return x.options().dtype(at::kFloat);
}

}  // namespace qfused

// =============================================================================================
// Host entry points
// =============================================================================================

void per_token_quant_fp8_v2(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  JO_CHECK_CUDA(input);
  JO_CHECK_CONTIGUOUS(input);
  JO_CHECK_CUDA(output_q);
  JO_CHECK_CUDA(output_s);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_tokens, hidden_dim]");
  const int64_t M = input.size(0), K = input.size(1);
  int vpt = 0, threads = 0;
  if (input.scalar_type() == torch::kBFloat16 && output_q.is_contiguous() && output_s.is_contiguous() &&
      qfused::pick_rowreg_config(K, &vpt, &threads) && qfused::aligned16(input.data_ptr()) &&
      qfused::aligned16(output_q.data_ptr())) {
    const c10::cuda::CUDAGuard guard(input.device());
    qfused::launch_rowreg<false>(static_cast<const __nv_bfloat16*>(input.data_ptr()),
                                 static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                 static_cast<float*>(output_s.data_ptr()), M, K, vpt, threads,
                                 at::cuda::getCurrentCUDAStream());
    return;
  }
  sgl_per_token_quant_fp8(input, output_q, output_s);
}

std::tuple<torch::Tensor, torch::Tensor> gelu_tanh_quant_fp8(const torch::Tensor& x) {
  JO_CHECK_CUDA(x);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  const int64_t M = x.size(0), N = x.size(1);
  auto q = torch::empty({M, N}, qfused::fp8_opts(x));
  auto s = torch::empty({M, 1}, qfused::f32_opts(x));
  int vpt = 0, threads = 0;
  if (x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && qfused::aligned16(x.data_ptr()) &&
      qfused::pick_rowreg_config(N, &vpt, &threads, /*gelu=*/true)) {
    const c10::cuda::CUDAGuard guard(x.device());
    qfused::launch_rowreg<true>(static_cast<const __nv_bfloat16*>(x.data_ptr()),
                                static_cast<__nv_fp8_e4m3*>(q.data_ptr()),
                                static_cast<float*>(s.data_ptr()), M, N, vpt, threads,
                                at::cuda::getCurrentCUDAStream());
    return {q, s};
  }
  sgl_per_token_quant_fp8(at::gelu(x.contiguous(), "tanh"), q, s);
  return {q, s};
}

// warp_rows: rows (warps) per block for the N == 4096 LayerNorm warp-per-row kernel;
// 0 forces the verbatim block-per-row kernel (used for every other shape).
std::tuple<torch::Tensor, torch::Tensor> fused_norm_scale_shift_fp8_impl(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps, int warp_rows) {
  JO_CHECK_CUDA(x);
  JO_CHECK_CUDA(scale);
  JO_CHECK_CUDA(shift);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  const int64_t M = x.size(0), N = x.size(1);
  const bool fast = x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && (N % 4) == 0;
  if (!fast) {
    auto y = fused_norm_scale_shift(x, gamma_opt, beta_opt, scale, shift, norm_type, eps);
    auto q = torch::empty({M, N}, qfused::fp8_opts(x));
    auto s = torch::empty({M, 1}, qfused::f32_opts(x));
    sgl_per_token_quant_fp8(y.contiguous(), q, s);
    return {q, s};
  }
  // Argument checks mirror fused_norm_scale_shift.
  TORCH_CHECK(scale.dim() == 2 && shift.dim() == 2, "scale/shift must be 2D [M, N]");
  TORCH_CHECK(scale.size(0) == shift.size(0), "scale/shift must have the same number of rows");
  TORCH_CHECK(scale.size(0) == M || scale.size(0) == 1,
              "scale/shift rows must equal M (per-row modulation) or 1 (broadcast)");
  const bool broadcast_rows = scale.size(0) == 1 && M != 1;
  if (!broadcast_rows) {
    TORCH_CHECK(scale.stride(0) == N && shift.stride(0) == N, "per-row scale/shift must be contiguous");
  }
  TORCH_CHECK(scale.size(1) == N && shift.size(1) == N, "scale/shift last dim must be N");
  TORCH_CHECK(scale.stride(-1) == 1 && shift.stride(-1) == 1, "scale/shift last dim must be contiguous");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x/scale/shift dtype must match");
  TORCH_CHECK(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms)");
  const bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  const bool has_beta = beta_opt.has_value() && beta_opt->defined();
  if (has_gamma) TORCH_CHECK(gamma_opt->numel() == N, "gamma must be length N");
  if (has_beta) TORCH_CHECK(beta_opt->numel() == N, "beta must be length N");
  TORCH_CHECK(!(has_gamma && norm_type == 0 && !has_beta), "LayerNorm with gamma requires beta");
  const bool affine = has_gamma;

  const c10::cuda::CUDAGuard guard(x.device());
  auto q = torch::empty({M, N}, qfused::fp8_opts(x));
  auto s = torch::empty({M, 1}, qfused::f32_opts(x));
  if (M == 0) return {q, s};
  const int row4 = broadcast_rows ? 0 : (int)(N / 4);
  using T4 = qfused::bf16_4;
  using T = __nv_bfloat16;
  const T4* gp = has_gamma ? static_cast<const T4*>(gamma_opt->data_ptr()) : nullptr;
  const T4* bp = has_beta ? static_cast<const T4*>(beta_opt->data_ptr()) : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool warp_row_ok = warp_rows > 0 && N == 4096 && norm_type == qfused::kLayerNorm &&
                           qfused::aligned16(x.data_ptr()) && qfused::aligned16(scale.data_ptr()) &&
                           qfused::aligned16(shift.data_ptr()) &&
                           (!has_gamma || qfused::aligned16(gp)) && (!has_beta || qfused::aligned16(bp));
  if (warp_row_ok) {
    dim3 wgrid((unsigned)((M + warp_rows - 1) / warp_rows)), wblock((unsigned)(warp_rows * 32));
#define JO_WLAUNCH(W)                                                                               \
  qfused::layerNormFp8WarpRow4096<W><<<wgrid, wblock, 0, stream>>>(                                \
      static_cast<__nv_fp8_e4m3*>(q.data_ptr()), static_cast<float*>(s.data_ptr()),               \
      static_cast<const T4*>(x.data_ptr()), gp, bp, static_cast<const T4*>(scale.data_ptr()),     \
      static_cast<const T4*>(shift.data_ptr()), (int)N, row4, affine, (float)eps, (int)M)
    switch (warp_rows) {
      case 1: JO_WLAUNCH(1); break;
      case 2: JO_WLAUNCH(2); break;
      case 4: JO_WLAUNCH(4); break;
      case 8: JO_WLAUNCH(8); break;
      default: TORCH_CHECK(false, "warp_rows must be 1, 2, 4 or 8");
    }
#undef JO_WLAUNCH
    return {q, s};
  }
  dim3 grid((unsigned)M), block;
#define JO_LAUNCH(IPT, NT)                                                                          \
  qfused::normScaleShiftFp8PerRow<T4, T, IPT, NT><<<grid, block, 0, stream>>>(                    \
      static_cast<__nv_fp8_e4m3*>(q.data_ptr()), static_cast<float*>(s.data_ptr()),               \
      static_cast<const T4*>(x.data_ptr()), gp, bp, static_cast<const T4*>(scale.data_ptr()),     \
      static_cast<const T4*>(shift.data_ptr()), (int)N, row4, affine, (float)eps)
  // Launch geometry identical to fused_norm_scale_shift (it fixes the float reduction order).
  if (N <= 4096) {
    block.x = (unsigned)((N / 4 + 31) / 32 * 32);
    if (block.x > 1024) block.x = 1024;
    if (norm_type == qfused::kLayerNorm) { JO_LAUNCH(1, qfused::kLayerNorm); } else { JO_LAUNCH(1, qfused::kRMSNorm); }
  } else {
    block.x = (unsigned)(((N + 7) / 8 + 31) / 32 * 32);
    if (block.x > 1024) block.x = 1024;
    if (norm_type == qfused::kLayerNorm) { JO_LAUNCH(8, qfused::kLayerNorm); } else { JO_LAUNCH(8, qfused::kRMSNorm); }
  }
#undef JO_LAUNCH
  return {q, s};
}

std::tuple<torch::Tensor, torch::Tensor> fused_norm_scale_shift_fp8(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps) {
  return fused_norm_scale_shift_fp8_impl(x, gamma_opt, beta_opt, scale, shift, norm_type, eps,
                                         /*warp_rows=*/2);
}

}  // namespace joyomni_ops
