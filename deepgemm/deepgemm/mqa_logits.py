import torch
from typing import Optional, Dict, Any, List, NamedTuple
import functools
import os
import math
from . import op

def mqa_logits(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    Weights: torch.Tensor, 
    ks: torch.Tensor, 
    ke: torch.Tensor, 
    kv_scale: Optional[torch.Tensor] = None,
    clean_logit: bool = True,
    D_out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    q_seq_len, num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[0]
    
    assert num_heads in [1, 2, 4, 8, 16, 32, 64, 128], "num_heads only support [1, 2, 4, 8, 16, 32, 64, 128]"
    assert head_dim == 128, "head_dim only support 128"
    
    outs = op.mqa_logits(
        Q, K, Weights, ks, ke, 
        q_seq_len, kv_seq_len, num_heads, head_dim, 
        kv_scale, clean_logit, D_out
    )
    return outs

def get_paged_mqa_logits_metadata(
        context_lens: torch.Tensor,
        block_kv: int,
        num_sms: int) -> torch.Tensor:
    schedule_metadata = op.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_sms)
    return schedule_metadata

def paged_mqa_logits(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        clean_logits: bool = True) -> torch.Tensor:
    logits = op.paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits)
    return logits
