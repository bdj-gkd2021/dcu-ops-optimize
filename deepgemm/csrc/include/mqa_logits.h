#pragma once

#include <torch/extension.h>

namespace deepgemm {
torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor> &KV_scale =  torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor> &D_out = torch::nullopt);
}