import torch
from typing import Optional, Dict, Any, List, NamedTuple
import functools
import os
import math
from . import op

# BLOCK_M of the embedded ``moe_w8a8_marlin_decode_down_fp8`` op (mode 38 tile
# <16,128,64,16,32,64,2>). Must match the kernel's BLOCK_SIZE_M template arg.
_MARLIN_DECODE_BLOCK_M = 16


def _build_contiguous_routing_from_m_indices(m_indices: torch.Tensor, block_m: int):
    """Convert contiguous ``m_indices`` (shape [M], int32, ``-1`` marks padding rows)
    into the MoE-routing tensors expected by the ``moe_w8a8_marlin_decode_down_fp8``
    op (`sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights`).

    Routing encoding (matches the kernel at ``moe_marlin_decode_device.inc.h``):
      * sorted_token_ids[j] = j                  if m_indices[j] >= 0  (topk_id=0, valid)
      * sorted_token_ids[j] = (1 << 24) | j      if m_indices[j] == -1 (topk_id=1 ≥ real_topk=1
                                                  → per-row store is skipped)
      * expert_ids[b] = m_indices[b*block_m]     (first row of the b-th block; -1 triggers
                                                  the kernel's expert==-1 zero-fill path)
      * num_tokens_post_pad[0] = M               (kernel's early-exit boundary)
      * topk_weights[j] = 1.0                    (kernel multiplies output by it; we want 1×)
    """
    if m_indices.dim() != 1:
        raise ValueError(f"m_indices must be 1-D [M], got shape {tuple(m_indices.shape)}")
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    M = int(m_indices.shape[0])
    if M % block_m != 0:
        raise ValueError(
            f"M ({M}) must be a multiple of BLOCK_M ({block_m}); the contiguous "
            f"caller is responsible for padding (test aligns to 256, divisible by 16)."
        )
    device = m_indices.device
    rows = torch.arange(M, device=device, dtype=torch.int32)
    pad_marker = (m_indices < 0).to(torch.int32) << 24
    sorted_token_ids = (pad_marker | rows).contiguous()
    expert_ids = m_indices[::block_m].contiguous()
    num_tokens_post_pad = torch.tensor([M], device=device, dtype=torch.int32)
    topk_weights = torch.ones(M, device=device, dtype=torch.float32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights


def m_grouped_i8_gemm_nt_contiguous( a: tuple[torch.Tensor, torch.Tensor],
                                       b: tuple[torch.Tensor, torch.Tensor], 
                                       output:torch.Tensor, m_indices:torch.Tensor, 
                                       config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:
    mode = config['MODE']
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_w8a8_gemm_nt_contiguous(input, b_qweight, output, a_scale, b_scale, m_indices, mode) 

def m_grouped_i8_gemm_nt_contiguous_nopad( a: tuple[torch.Tensor, torch.Tensor],
                                             b: tuple[torch.Tensor, torch.Tensor],
                                             output:torch.Tensor, m_indices:torch.Tensor,
                                             token_per_expert:torch.Tensor) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_w8a8_gemm_nt_nopad_contiguous(input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert)     

def m_grouped_fp8_gemm_nt_contiguous( a: tuple[torch.Tensor, torch.Tensor],
                                             b: tuple[torch.Tensor, torch.Tensor],
                                             output:torch.Tensor, m_indices:torch.Tensor,
                                             config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:
    
    input, a_scale = a
    b_qweight, b_scale = b
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
    ) = _build_contiguous_routing_from_m_indices(m_indices, _MARLIN_DECODE_BLOCK_M)
    torch.ops.deep_gemm.moe_w8a8_marlin_decode_down_fp8(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        1,  # top_k
        1,  # real_topk
    )
    return output

def m_grouped_bf16_gemm_nt_contiguous(a:torch.Tensor, b:torch.Tensor, 
                                             output:torch.Tensor, m_indices:torch.Tensor,
                                             config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:
    mode = config['MODE']
    M = a.shape[0]
    E, N = b.shape[0],b.shape[1]
    a_scale = torch.ones(M, dtype=torch.float32, device=a.device)
    b_scale = torch.ones((E,N),dtype=torch.float32, device=b.device)
    return op.m_grouped_w16a16_gemm_nt_contiguous(a, b, output, a_scale, b_scale, m_indices, mode)   

