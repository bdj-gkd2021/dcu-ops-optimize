// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/extension.h>

namespace moe_marlin {

/// MoE Marlin W8A8 GEMM2 prefill-down (HIP). Tensor contract matches MoE ``moe_gemm_marlin_w8a8``
/// GEMM2 path; tile is fixed: ``launch_moe_w8a8_second_stage_prefill<32,128,64,32,32,64,2>``
/// (see ``kernel_maps_gemm2_prefill`` mode 86 in ``moe_w8a8_config.h``).
torch::Tensor moe_marlin_w8a8_gemm2_prefill_down(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t real_topk);

} // namespace moe_marlin
