// SPDX-License-Identifier: Apache-2.0
// MoE Marlin W8A8 GEMM2 prefill-down — use TORCH_LIBRARY_FRAGMENT so ``deep_gemm`` shares one
// namespace with low_latency_grouped_gemm.cu (PyTorch allows only one TORCH_LIBRARY per ns).

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "moe_marlin/moe_w8a8_gemm_params.h"
#include "moe_marlin/moe_w8a8_utils.h"
#include "moe_marlin/moe_marlin_prefill_device.inc.h"
#include "moe_marlin/moe_marlin_prefill_launch.inc.h"

namespace moe_marlin {

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
    int64_t real_topk) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));

  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kChar, "input int8");
  TORCH_CHECK(b_qweight.scalar_type() == at::kChar, "weight int8");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output bf16");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
      "scales float32");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights float32");

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n =
      static_cast<int>(b_qweight.size(2) * b_qweight.size(1) / size_k);

  TORCH_CHECK(
      size_k == 128 || size_k == 256 || size_k == 320 || size_k == 384 ||
          size_k == 640 || size_k == 768 || size_k == 2048,
      "moe_w8a8_gemm2_prefill_down: K=",
      size_k,
      " not supported by embedded PREFILL_DOWN kernel (expected one of 128,256,320,384,640,768,2048)");

  constexpr int BLOCK_SIZE_N = 128;
  constexpr int N_LOOP = 4;
  TORCH_CHECK(
      size_n % (BLOCK_SIZE_N * N_LOOP) == 0,
      "moe_w8a8_gemm2_prefill_down: N=",
      size_n,
      " must be divisible by ",
      BLOCK_SIZE_N * N_LOOP,
      " for tile <32,128,64,32,32,64,2>");

  const int stride_asm = static_cast<int>(a_scale.stride(0));
  const int stride_ask = static_cast<int>(a_scale.stride(1));
  const int stride_bse = static_cast<int>(b_scale.stride(0));
  const int stride_bsn = static_cast<int>(b_scale.stride(1));
  const int stride_bsk = static_cast<int>(b_scale.stride(2));

  const int64_t EM = sorted_token_ids.size(0);

  torch::Tensor output_alias = output.alias();

  GemmParams<char, bhalf_t> params_in(
      reinterpret_cast<const char*>(input.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<int8_t>()),
      reinterpret_cast<bhalf_t*>(output_alias.data_ptr()),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      0,
      num_tokens_post_pad.data_ptr<int32_t>(),
      static_cast<uint32_t>(size_m),
      static_cast<uint32_t>(size_n),
      static_cast<uint32_t>(size_k),
      static_cast<uint32_t>(stride_asm),
      static_cast<uint32_t>(stride_ask),
      static_cast<uint32_t>(stride_bse),
      static_cast<uint32_t>(stride_bsn),
      static_cast<uint32_t>(stride_bsk),
      static_cast<uint32_t>(EM),
      static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(real_topk),
      true);

  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  launch_moe_w8a8_second_stage_prefill<32, 128, 64, 32, 32, 64, 2>(params_in,
                                                                     stream);
  return output;
}

} // namespace moe_marlin

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def(
      "moe_w8a8_i8_marlin_prefill_down("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor topk_weights, "
      "Tensor sorted_token_ids, "
      "Tensor expert_ids, "
      "Tensor num_tokens_post_pad, "
      "int top_k, "
      "int real_topk"
      ") -> Tensor");
  m.impl(
      "moe_w8a8_i8_marlin_prefill_down",
      torch::kCUDA,
      &moe_marlin::moe_marlin_w8a8_gemm2_prefill_down);
}
