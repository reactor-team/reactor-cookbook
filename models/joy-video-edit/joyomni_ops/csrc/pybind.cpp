// Single Torch library registration for joyomni_ops.
// Op names are exposed as torch.ops.joyomni_ops.<name> and mirror the sgl_kernel
// signatures the JoyOmni pipeline uses, so the Python shim is a thin rename.
//
// Define JOYOMNI_OPS_NO_FP8 to build without the cutlass FP8 GEMM (lets the 3
// light kernels compile on toolchains without cutlass / CUDA<12.8).
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

namespace joyomni_ops {

void fused_qk_norm_rope_3d_paired(
    torch::Tensor& q, torch::Tensor& k, int64_t seq_len, int64_t num_heads, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight, torch::Tensor& cos, torch::Tensor& sin);

torch::Tensor fused_norm_scale_shift(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps);

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps);

void fused_qk_norm_rope_3d_paired_to(
    const torch::Tensor& q_src, const torch::Tensor& k_src, torch::Tensor& q_dst, torch::Tensor& k_dst,
    double eps, const torch::Tensor& q_weight, const torch::Tensor& k_weight,
    const torch::Tensor& cos, const torch::Tensor& sin);

torch::Tensor masked_attn_rescale(const torch::Tensor& o, const torch::Tensor& lse, const torch::Tensor& text_pad);

#ifndef JOYOMNI_OPS_NO_FP8
void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s);

void per_token_quant_fp8_v2(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s);

std::tuple<torch::Tensor, torch::Tensor> fused_norm_scale_shift_fp8(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps);

std::tuple<torch::Tensor, torch::Tensor> gelu_tanh_quant_fp8(const torch::Tensor& x);

torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
    const torch::Tensor& scales_b, const torch::ScalarType& out_dtype,
    const c10::optional<torch::Tensor>& bias);
#endif

}  // namespace joyomni_ops

TORCH_LIBRARY(joyomni_ops, m) {
  m.def(
      "fused_qk_norm_rope_3d_paired(Tensor! q, Tensor! k, int seq_len, int num_heads, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin) -> ()");
  m.def(
      "fused_norm_scale_shift(Tensor x, Tensor? gamma_opt, Tensor? beta_opt, "
      "Tensor scale, Tensor shift, int norm_type, float eps) -> Tensor");
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_qk_norm_rope_3d_paired_to(Tensor q_src, Tensor k_src, Tensor! q_dst, Tensor! k_dst, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin) -> ()");
  m.def("masked_attn_rescale(Tensor o, Tensor lse, Tensor text_pad) -> Tensor");
#ifndef JOYOMNI_OPS_NO_FP8
  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor! output_q, Tensor! output_s) -> ()");
  m.def("per_token_quant_fp8_v2(Tensor input, Tensor! output_q, Tensor! output_s) -> ()");
  m.def(
      "fused_norm_scale_shift_fp8(Tensor x, Tensor? gamma_opt, Tensor? beta_opt, Tensor scale, "
      "Tensor shift, int norm_type, float eps) -> (Tensor, Tensor)");
  m.def("gelu_tanh_quant_fp8(Tensor x) -> (Tensor, Tensor)");
  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, "
      "ScalarType out_dtype, Tensor? bias) -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL(joyomni_ops, CUDA, m) {
  m.impl("fused_qk_norm_rope_3d_paired", &joyomni_ops::fused_qk_norm_rope_3d_paired);
  m.impl("fused_norm_scale_shift", &joyomni_ops::fused_norm_scale_shift);
  m.impl("rmsnorm", &joyomni_ops::rmsnorm);
  m.impl("fused_qk_norm_rope_3d_paired_to", &joyomni_ops::fused_qk_norm_rope_3d_paired_to);
  m.impl("masked_attn_rescale", &joyomni_ops::masked_attn_rescale);
#ifndef JOYOMNI_OPS_NO_FP8
  m.impl("sgl_per_token_quant_fp8", &joyomni_ops::sgl_per_token_quant_fp8);
  m.impl("per_token_quant_fp8_v2", &joyomni_ops::per_token_quant_fp8_v2);
  m.impl("fused_norm_scale_shift_fp8", &joyomni_ops::fused_norm_scale_shift_fp8);
  m.impl("gelu_tanh_quant_fp8", &joyomni_ops::gelu_tanh_quant_fp8);
  m.impl("fp8_scaled_mm", &joyomni_ops::fp8_scaled_mm);
#endif
}
