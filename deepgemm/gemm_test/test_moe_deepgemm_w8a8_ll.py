#!/usr/bin/env python3
"""
Test script for deepgemm W8A8 low-latency masked kernel on DCU.

Tests m_grouped_w8a8_gemm_nt_masked_ll (low-latency path with 6-D packed weights).

LL kernel dimension restrictions (from low_latency_fp8_masked_utils.h):
  - K in {1536, 2048, 3072, 6144, 7168}
  - N in {3072, 4096, 6144, 7168}
  - E in {1, 16, 32}

Weight packing: pack_int8_weight_enk_to_w6_low_latency (6-D layout, NOT Marlin).
Scales: 2-D [E, M] and [E, N] when block_wise=False.
"""

import torch
import os
import time

os.environ["LD_LIBRARY_PATH"] = "/zkjh/drivers/ib_plugin/topo_lib/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from deepgemm.m_group_gemm import (
    m_grouped_w8a8_gemm_nt_masked_ll,
    pack_int8_weight_enk_to_w6_low_latency,
)
from lmslim.layers.gemm.int8_utils import per_token_quant_int8


def align_up(x: int, align: int = 256) -> int:
    return ((x + align - 1) // align) * align


def native_w8a8_per_channel_batch_matmul(
    q_a_all: torch.Tensor,
    weight: torch.Tensor,
    qa_scale_all: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    PyTorch reference for correctness validation.
    q_a_all:      [E, M_alloc, K], int8
    weight:       [E, N, K], int8
    qa_scale_all: [E, M_alloc]   (2-D, per-token)
    w_scale:      [E, N]         (2-D, per-channel)
    return:       [E, M_alloc, N]
    """
    A = q_a_all.to(torch.float32)
    B = weight.to(torch.float32)
    C = torch.bmm(A, B.transpose(1, 2))
    C = qa_scale_all.float().unsqueeze(-1) * C * w_scale.float().unsqueeze(1)
    return C.to(output_dtype)


def test_precision():
    """Precision test: compare deepgemm LL masked GEMM output with PyTorch reference."""
    print("\n" + "=" * 80)
    print("PRECISION TEST: deepgemm m_grouped_w8a8_gemm_nt_masked_ll vs PyTorch reference")
    print("=" * 80)

    torch.cuda.empty_cache()

    # MiniMax-M2.5 EP=8+TP=8: E=32 fixed, test both GEMM layers
    E = 32
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024]
    cu_values = [int(v) for v in os.environ.get("CU_VALUES", "64 80 128 256").split()]

    # (K, N, label)
    configs = [
        (3072, 3072, "GEMM1 EP=8 only (K=3072,N=3072)"),
        (3072, 384,  "GEMM1 EP=8+TP=8 (K=3072,N=384)"),
        (192,  3072, "GEMM2 EP=8+TP=8 (K=192,N=3072)"),
    ]

    print(f"CU values to test: {cu_values}")

    for K, N, label in configs:
        print(f"\n{'=' * 40} {label} {'=' * 40}")
        for M in M_values:
            print(f"\n--- M = {M} ---")
            torch.cuda.empty_cache()
            torch.manual_seed(42)

            M_alloc = align_up(M, 256)

            hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
            weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
            weight_scale = (torch.randn((E, N), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
            masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

            q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
            q_a_all = q_a_all.contiguous()
            qa_scale_all = qa_scale_all.contiguous()

            # LL kernel uses 2-D scales (squeeze last dim)
            qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()
            w_scale_2d = weight_scale.contiguous()

            # PyTorch reference
            ref = native_w8a8_per_channel_batch_matmul(
                q_a_all, weight, qa_scale_2d, w_scale_2d, output_dtype=out_dtype
            )

            # LL 6-D weight packing
            weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)

            for test_cu in cu_values:
                out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
                m_grouped_w8a8_gemm_nt_masked_ll(
                    (q_a_all, qa_scale_2d),
                    (weight_ll, w_scale_2d),
                    out,
                    masked_m,
                    M_alloc,
                    block_wise=False,
                    cu=test_cu,
                )
                torch.cuda.synchronize()

                precision_ok = True
                max_abs = 0.0
                max_rel = 0.0
                bad_expert = -1

                for e in range(E):
                    ref_e = ref[e, :M, :]
                    out_e = out[e, :M, :]
                    abs_diff = (ref_e.float() - out_e.float()).abs()
                    cur_max_abs = abs_diff.max().item()
                    cur_max_rel = (abs_diff / ref_e.float().abs().clamp_min(1e-6)).max().item()
                    max_abs = max(max_abs, cur_max_abs)
                    max_rel = max(max_rel, cur_max_rel)
                    if not torch.allclose(ref_e, out_e, rtol=1e-2, atol=1e-2):
                        precision_ok = False
                        bad_expert = e
                        break

                cos_sim = torch.nn.functional.cosine_similarity(
                    ref[:, :M, :].flatten().float(), out[:, :M, :].flatten().float(), dim=0
                ).item()

                status = "PASS" if precision_ok else f"FAIL (expert {bad_expert})"
                print(f"  CU={test_cu:>3}: cos={cos_sim:.6f}, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}  {status}")


def test_performance():
    """Performance test: measure latency for different batch sizes."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST: deepgemm m_grouped_w8a8_gemm_nt_masked_ll throughput")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, K, N = 32, 3072, 3072
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024, 4096]
    n_warmup = 5
    n_measure = 20

    print(f"\n{'M':>6} | {'Latency (ms)':>14} | {'Throughput (tokens/s)':>22} | Status")
    print("-" * 65)

    for M in M_values:
        torch.manual_seed(42)
        M_alloc = align_up(M, 256)

        hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
        weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
        weight_scale = (torch.randn((E, N), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
        masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

        q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
        q_a_all = q_a_all.contiguous()
        qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()

        weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)

        out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()

        for _ in range(n_warmup):
            m_grouped_w8a8_gemm_nt_masked_ll(
                (q_a_all, qa_scale_2d),
                (weight_ll, weight_scale),
                out,
                masked_m,
                M_alloc,
                block_wise=False,
                cu=128,
            )
        torch.cuda.synchronize()

        latencies = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            m_grouped_w8a8_gemm_nt_masked_ll(
                (q_a_all, qa_scale_2d),
                (weight_ll, weight_scale),
                out,
                masked_m,
                M_alloc,
                block_wise=False,
                cu=128,
            )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        throughput = (E * M) / (avg_latency / 1000)

        print(f"{M:>6} | {avg_latency:>14.3f} | {throughput:>22.0f} | PASS")


if __name__ == "__main__":
    import sys
    tests = {
        "precision": test_precision,
        "performance": test_performance,
    }

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {list(tests.keys())}")
    else:
        print("Run tests individually:")
        print("  python test_moe_deepgemm_w8a8_ll.py precision")
        print("  python test_moe_deepgemm_w8a8_ll.py performance")
