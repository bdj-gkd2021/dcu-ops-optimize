import torch
from typing import Optional, Dict, Any, List, NamedTuple, Tuple
import functools
import os
import math
from . import op

def fp8_gemm(a: Tuple[torch.Tensor, torch.Tensor],
                     b: Tuple[torch.Tensor, torch.Tensor],
                     d: torch.Tensor,
                     c: Optional[torch.Tensor] = None,
                     recipe: Optional[Tuple[int, int, int]] = None,
                     compiled_dims: str = "nk",
                     disable_ue8m0_cast: bool = False):
    batch = b[0].shape[0] if b[0].dim() == 3 else 1
    m = a[0].shape[1] if a[0].dim() == 3 else a[0].shape[0]
    k = a[0].shape[2] if a[0].dim() == 3 else a[0].shape[1]

    b1 = b[0].shape[1] if b[0].dim() == 3 else b[0].shape[0]
    b2 = b[0].shape[2] if b[0].dim() == 3 else b[0].shape[1]
    n = b1 if b2 == k else b2


    # if compiled_dims == 'nk':
    #     n = b[0].shape[1] if b[0].dim() == 3 else b[0].shape[0]
    # elif compiled_dims == 'kn':
    #     n = b[0].shape[2] if b[0].dim() == 3 else b[0].shape[1]
    # else:
    #     raise ValueError(f"不支持的 compiled_dims: {compiled_dims}")
    # if (compiled_dims == 'nk' and b[0].stride(-1) == 1) or (compiled_dims == 'kn' and b[0].stride(-1) != 1):
    gemm_layout = 'TN'
    # else:
    #     gemm_layout = 'NN'
    if a[0].dtype == torch.int8:
        alpha = torch.tensor(1, dtype=torch.int32, device='cpu')
        beta = torch.tensor(0, dtype=torch.int32, device='cpu')
    elif a[0].dtype == torch.float8_e4m3fn:
        alpha = torch.tensor(1, dtype=torch.float32, device='cpu')
        beta = torch.tensor(0, dtype=torch.float32, device='cpu')
    else:
        raise ("不支持类型")
    op.fp8_gemm(b[0], a[0], b[1], a[1], d, n, m, k, batch, gemm_layout, alpha, beta, None)

    
