import torch
import pandas as pd
import random
import math
import torch
import triton
from typing import Type
from deepgemm import m_grouped_fp8_gemm_nt_contiguous
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency


def align(x, y):
    return ((x + y - 1) // y) * y

def to_fp8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.float8_e4m3fn)


def check_fp8_precision(ref_d: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor,
                        rtol: float = 1e-2, atol: float = 5e-2) -> bool:
    """FP8 精度检查：仅对有效行（m_indices >= 0）比较，并输出误差统计。"""
    valid_mask = (m_indices >= 0).unsqueeze(1).expand_as(ref_d)
    ref_valid = ref_d.masked_select(valid_mask)
    d_valid = d.masked_select(valid_mask)
    if ref_valid.numel() == 0:
        print("FP8 精度检查: 无有效行，跳过")
        return True
    abs_err = (ref_valid.float() - d_valid.float()).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    ref_abs = ref_valid.float().abs().clamp(min=1e-7)
    rel_err = (abs_err / ref_abs)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()
    print(f"  FP8 精度统计: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
          f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}")
    ok = torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol)
    return ok

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    #每个group实际的M
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    #将M对齐到128
    aligned_ms = [align(actual_m, 256) for actual_m in actual_ms]
    #group的M的总和
    m = sum(aligned_ms)
    print("align_m after m : ", m)
    # a 是整体input
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)  
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    ref_d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        #表示M属于哪个group id
        m_indices[start:actual_end] = i        #[000...-1-1-1, 111...-1-1]           
        m_indices[actual_end:aligned_end] = -1

        # ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    
    # ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)
    a_f8 = to_fp8(a)
    b_f8 = to_fp8(b)
    # FP8 使用正 scale，常用 (0.5, 2) 范围
    a_scale = torch.rand(m, device='cuda', dtype=torch.float) * 1.5 + 0.5
    b_scale = torch.rand((num_groups, n), device='cuda', dtype=torch.float) * 1.5 + 0.5

    group_a = [a_f8, a_scale]
    group_b = [b_f8, b_scale]
    return m, group_a, group_b, m_indices, d, ref_d


def torch_gemm(
    A: torch.Tensor,          # [B, M, K]
    B: torch.Tensor,          # [B_ext, N, K] （注意这里按“行”参与点积，对应原代码的 B[n, k] 索引）
    C: torch.Tensor,          # [B, M, N]
    scaleA: torch.Tensor,     # [B, M]
    scaleB: torch.Tensor,     # [B_ext, N]
    alpha: float,
    beta: float,
    skipA: int = 1,
    skipB: int = 1,
    m_index_h: torch.Tensor = None  # [M]，元素为 e_id，-1 表示跳过
) -> torch.Tensor:

    M, K = A.shape
    B_ext, N, Kb = B.shape

    D = C.clone()
    # 为了数值稳定，按 float32 做累加与缩放，再 cast 回 D.dtype
    for b in range(1):
        for m_idx in range(0, M, skipA):
            e_id = int(m_index_h[m_idx].item())
            if e_id == -1:
                continue
            bp = b + e_id
            if bp < 0 or bp >= B_ext:
                # 越界直接跳过（与原逻辑一致，不写入）
                continue

            a_row = A[m_idx, :].to(torch.float32)       # [K]
            sa = scaleA[m_idx].to(torch.float32)

            for n_idx in range(0, N, skipB):
                b_row = B[bp, n_idx, :].to(torch.float32)   # [K]
                sb = scaleB[bp, n_idx].to(torch.float32)

                s = torch.dot(a_row, b_row)                 # sum over k
                s = s * sa * sb

                val = alpha * s + beta * C[m_idx, n_idx].to(torch.float32)
                D[m_idx, n_idx] = val.to(D.dtype)

    return D

def test_m_grouped_gemm_contiguous(warmup_iters=10, test_iters=100, acc_check=True) -> None:
    num_groups = 4
    # 多组 (M, N, K) 做 FP8 精度与性能测试
    expected_m_per_group_list = [64, 128]
    n_list = [128, 1024]
    k_list = [128, 1024]
    out_dtype = torch.bfloat16
    for k_val in k_list:
        for n_val in n_list:
            for me in expected_m_per_group_list:
                expected_m_per_group, n, k = me, n_val, k_val
                m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k)
                print("m:{} n:{} k:{} (fp8)".format(m, n, k))
                a_f8, a_scale = a
                b_f8, b_scale = b
                w = pack_int8_weight_enk_to_w6_low_latency(b_f8)
                b = [w, b_scale]
                for _ in range(1):
                    m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)
                torch.cuda.synchronize()

                if (acc_check):
                    ref_d = torch_gemm(a_f8, b_f8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices)
                    ok = check_fp8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
                    if not ok:
                        print("FP8 精度检查不合格！！！")
                    else:
                        print("FP8 精度检查合格！！！")

                elapsed_time = triton.testing.do_bench(
                    lambda: m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices),
                    quantiles=None, return_mode="mean"
                ) * 1000
                print(f"deepGemm contiguous FP8 算子 - M{m}xN{n}xK{k}, 平均耗时: {elapsed_time:.2f} us\n")

test_m_grouped_gemm_contiguous(acc_check=True)
