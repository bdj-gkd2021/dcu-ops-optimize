import torch
import time
from lmslim import quant_ops 
from typing import Optional, Type
import pandas as pd
import os
import json
from deepgemm import m_grouped_i8_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous
import torch
import random
import triton
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin
from vllm.platforms import current_platform

device_name = current_platform.get_device_name().replace(" ", "_")
num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def align(x, y):
    return ((x + y - 1) // y) * y

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    #每个group实际的M
    actual_ms = [int(expected_m_per_group) for _ in range(num_groups)]
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
    a_int8 = to_int8(a)
    b_int8 = to_int8(b)
    a_scale = torch.randn(m, device='cuda', dtype=torch.float)
    b_scale = torch.randn((num_groups, n), device='cuda', dtype=torch.float)

    group_a = [a_int8, a_scale]
    group_b = [b_int8, b_scale]
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
    num_groups = 16
    expected_m_per_group_list = [64]
    n_list = [512]
    k_list = [512]
    out_dtype = torch.bfloat16
    for i in range(0,len(k_list),1):
        for me in expected_m_per_group_list:
            expected_m_per_group, n, k = me, n_list[0], k_list[0]
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k)
            print("m:{} n:{} k:{}".format(m, n, k))
            a_int8, a_scale = a
            b_int8, b_scale = b
            bm_int8 = []
            w = weight8bit_nt_kpack2_marlin(b_int8)
            b = [w, b_scale]
            for _ in range(1):
                m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices)
            torch.cuda.synchronize()

            if (acc_check):
                ref_d = torch_gemm(a_int8, b_int8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices)
                if not torch.allclose(ref_d, d, rtol=1e-2, atol=5e-2):
                    print("精度检查不合格！！！") 
                else:
                    print("精度检查合格！！！")
            
            elapsed_time = 0
            elapsed_time = triton.testing.do_bench(lambda:m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices),quantiles=None, return_mode="mean")*1000
            print(f"deepGemm continue算子速度测试结果 ({out_dtype}) - M{m}xN{n}xK{k}, 平均耗时: {elapsed_time:.2f} us")

test_m_grouped_gemm_contiguous(acc_check=False)


def deepgemm_contiguous_w8a8_benchmark(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    verify_precision: bool = False,
):
    m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m_per_group, n, k
    )
    a_int8, a_scale = a
    b_int8, b_scale = b
    w = weight8bit_nt_kpack2_marlin(b_int8)
    b = [w, b_scale]

    # warm up
    for _ in range(5):
        m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices)
    torch.cuda.synchronize()

    asm_cost_us = (
        triton.testing.do_bench(
            lambda: m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices),
            quantiles=None,
            return_mode="mean",
        )
        * 1000
    )

    if verify_precision:
        ref_d = torch_gemm(a_int8, b_int8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices)
        if not torch.allclose(ref_d, d, rtol=1e-2, atol=5e-2):
            print("精度检查不合格！！！")
        else:
            print("精度检查合格")

    computes = 2 * m * n * k
    tflops = computes / (asm_cost_us / 1e6) / 1e12
    data_bytes = m * k + num_groups * n * k + m * n * 2
    data_bytes += m * 4 + num_groups * n * 4  # scales float32
    bandwidth_gbs = data_bytes / (asm_cost_us / 1e6) / 1e9

    return asm_cost_us, tflops, bandwidth_gbs, m, m


def main_perf():
    num_groups = 16
    expected_m_per_group_list = [128, 256]
    n_list = [4096, 7168]
    k_list = [7168, 2048]
    out_dtype = torch.bfloat16

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    df_valid_m = []
    asm_costs = []
    df_tflops = []
    df_bandwidth = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(len(k_list)):
            n, k = n_list[i], k_list[i]
            print(
                "expected_m_per_group:{} n:{} k:{} ".format(
                    expected_m_per_group, n, k
                )
            )
            asm_cost_us, tflops, bandwidth_gbs, m, valid_m = deepgemm_contiguous_w8a8_benchmark(
                num_groups=num_groups,
                expected_m_per_group=expected_m_per_group,
                n=n,
                k=k,
                out_dtype=out_dtype,
                verify_precision=False,
            )

            asm_costs.append(asm_cost_us)
            df_E.append(num_groups)
            df_m.append(m)
            df_n.append(n)
            df_k.append(k)
            df_expected_m.append(expected_m_per_group)
            df_valid_m.append(valid_m)
            df_tflops.append(tflops)
            df_bandwidth.append(bandwidth_gbs)

            print("------------------------------------------------------")
            print(
                "deepgemm contiguous asm cost: {:.2f} us, {:.2f} TFlops".format(
                    asm_cost_us, tflops
                )
            )
            print("deepgemm contiguous asm bandwidth: {:.2f} GB/s".format(bandwidth_gbs))
            print("------------------------------------------------------")

    df = pd.DataFrame(
        {
            "E": df_E,
            "m": df_m,
            "n": df_n,
            "k": df_k,
            "expected_m": df_expected_m,
            "valid_m": df_valid_m,
            "asm deepgemm 耗时(us)": asm_costs,
            "asm deepgemm TFlops": df_tflops,
            "asm deepgemm Bandwidth(GB/s)": df_bandwidth,
        }
    )
    excel_name = "DeepGEMM_contiguous_groupgemm_w8a8.xlsx"
    df.to_excel(excel_name, index=False)
    print("表格已保存到 {} 文件中。".format(excel_name))


if __name__ == "__main__":
    main_perf()