// Low-latency grouped FP8 masked GEMM — fragment of ``deep_gemm`` (see 000_deep_gemm_torch_library_root.cu).
#include "low_latency_fp8_masked.h"

torch::Tensor group_gemm_fp8_masked(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matri_a_scale,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool block_wise,
    bool b_overlap,
    const c10::optional<torch::Tensor> &signal)
{
  DEEP_GEMM::FP8_GROUP_GEMM::MASKED::masked_fp8_gemm(
      matrix_a, matrix_b, matri_a_scale, matrixb_scale, actual_tokens, matrix_c,
      max_tokens, experts, cu_s, block_wise, b_overlap, signal);
  return matrix_c;
}

TORCH_LIBRARY_FRAGMENT(deep_gemm, m)
{

  m.def(
      "low_latency_grouped_gemm("
      "Tensor matrix_a, "
      "Tensor matrix_b, "
      "Tensor matrix_a_scale, "
      "Tensor matrix_b_scale, "
      "Tensor actual_tokens, "
      "Tensor! matrix_c, "
      "int max_tokens, "
      "int experts, "
      "int cu_s, "
      "bool block_wise, "
      "bool b_overlap, "
      "Tensor? signal"
      ") -> (Tensor)");
  m.impl("low_latency_grouped_gemm", torch::kCUDA, &group_gemm_fp8_masked);
}

