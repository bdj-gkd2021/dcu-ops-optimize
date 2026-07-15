from __future__ import annotations

import torch

__all__ = [
    "moe_w8a8_i8_marlin_prefill_down",
    "moe_prefill_routing_contiguous_expert_blocks",
    "moe_prefill_prepare_scales_2d_3d",
    "assert_moe_prefill_shapes",
]

# Supported K and N divisibility are enforced in C++; keep in sync with README / cu checks.
_MOE_PREFILL_K_ALLOWLIST = frozenset({128, 256, 320, 384, 640, 768, 2048})
_MOE_N_ALIGN = 512  # BLOCK_SIZE_N * n_loop_num == 128 * 4


def moe_w8a8_i8_marlin_prefill_down(
    input: torch.Tensor,
    b_qweight: torch.Tensor,
    output: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    real_topk: int,
) -> torch.Tensor:
    """MoE Marlin GEMM2 prefill-down (int8 activations × int8 Marlin weights → bf16)."""
    return torch.ops.deep_gemm.moe_w8a8_i8_marlin_prefill_down(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        top_k,
        real_topk,
    )


def moe_prefill_prepare_scales_2d_3d(
    a_scale_1d: torch.Tensor,
    b_scale_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match device expectations: ``a_scale`` [M,1] float32, ``b_scale`` [E,N,1] float32."""
    if a_scale_1d.dim() != 1:
        raise ValueError(f"a_scale_1d must be [M], got shape {tuple(a_scale_1d.shape)}")
    if b_scale_2d.dim() != 2:
        raise ValueError(f"b_scale_2d must be [E,N], got shape {tuple(b_scale_2d.shape)}")
    a2 = a_scale_1d.unsqueeze(-1).contiguous()
    b3 = b_scale_2d.unsqueeze(-1).contiguous()
    return a2, b3


def moe_prefill_routing_contiguous_expert_blocks(
    num_experts: int,
    tokens_per_expert: int,
    *,
    device: torch.device,
    dtype_ids: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens_per_expert <= 0 or num_experts <= 0:
        raise ValueError("num_experts and tokens_per_expert must be positive")
    if tokens_per_expert % 32 != 0:
        raise ValueError(
            "tokens_per_expert must be divisible by 32 (kernel BLOCK_SIZE_M) "
            f"for this helper, got {tokens_per_expert}"
        )
    m = num_experts * tokens_per_expert
    sorted_token_ids = torch.arange(m, device=device, dtype=dtype_ids)
    num_blocks = m // 32
    block_starts = torch.arange(num_blocks, device=device, dtype=dtype_ids) * 32
    expert_ids = torch.clamp(block_starts // tokens_per_expert, max=num_experts - 1)
    num_tokens_post_pad = torch.tensor([m], device=device, dtype=dtype_ids)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def assert_moe_prefill_shapes(*, k: int, n: int) -> None:
    if k not in _MOE_PREFILL_K_ALLOWLIST:
        raise ValueError(
            f"K={k} not in MoE prefill-down allowlist {_MOE_PREFILL_K_ALLOWLIST}"
        )
    if n % _MOE_N_ALIGN != 0:
        raise ValueError(f"N={n} must be divisible by {_MOE_N_ALIGN} for this kernel tile")
