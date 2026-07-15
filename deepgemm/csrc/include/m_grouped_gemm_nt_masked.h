#pragma once

#include <torch/extension.h>

namespace deepgemm {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
torch::Tensor m_grouped_marlin_w8a8_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
torch::Tensor m_grouped_marlin_fp8_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    bool enable_overlap,
    std::optional<torch::Tensor> signal,
    int mode);
torch::Tensor m_grouped_bf16_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
}