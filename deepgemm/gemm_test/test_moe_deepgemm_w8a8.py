#!/usr/bin/env python3
"""
Test script for deepgemm W8A8 kernels on DCU.

Tests both masked and contiguous grouped GEMM kernels used in SGLang's EP path.

Key findings:
1. Uses `weight8bit_nt_kpack2_marlin` for weight packing (not `weight8bit_nt_kpack2_marlin1`)
2. EP path uses full unsharded N = 3072 (not TP-sharded 384)
3. Input is pre-quantized int8 activations + per-token scale applied outside kernel
4. Weight scale: [E, N, 1] float32 (masked) or [E, N] float32 (contiguous)
5. Activation scale: [E, M_alloc, 1] float32 (masked) or [M] float32 (contiguous)

Dimensions (MiniMax M2):
- E = 256 experts (local)
- hidden (K) = 3072
- intermediate = 192 (TP-sharded to 384 per rank, full = 3072 for EP)
- top_k = 8
"""

import torch
import os
import time
import random

os.environ["LD_LIBRARY_PATH"] = "/zkjh/drivers/ib_plugin/topo_lib/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from deepgemm import m_grouped_w8a8_gemm_nt_masked, m_grouped_i8_gemm_nt_contiguous
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import weight8bit_nt_kpack2_marlin
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
    qa_scale_all: [E, M_alloc, 1]
    w_scale:      [E, N, 1]
    return:       [E, M_alloc, N]
    """
    A = q_a_all.to(torch.float32)
    B = weight.to(torch.float32)
    if qa_scale_all.dim() == 2:
        qa_scale_all = qa_scale_all.unsqueeze(-1)
    if w_scale.dim() == 2:
        w_scale = w_scale.unsqueeze(-1)
    C = torch.bmm(A, B.transpose(1, 2))
    C = qa_scale_all.float() * C * w_scale.float().transpose(1, 2)
    return C.to(output_dtype)


def test_precision():
    """Precision test: compare deepgemm masked GEMM output with PyTorch reference."""
    print("\n" + "=" * 80)
    print("PRECISION TEST: deepgemm m_grouped_w8a8_gemm_nt_masked vs PyTorch reference")
    print("=" * 80)

    torch.cuda.empty_cache()

    # EP path: N = 3072 (full unsharded intermediate * 2 for gate+up)
    # Test both E=256 (single rank) and E=32 (EP-8 mode: 256/8=32 experts per rank)
    E_values = [256, 32]
    K, N = 3072, 3072
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024]

    for E in E_values:
        print(f"\n{'=' * 40} E = {E} {'=' * 40}")
        for M in M_values:
            print(f"\n--- M = {M} ---")
            torch.cuda.empty_cache()
            torch.manual_seed(42)

            M_alloc = align_up(M, 256)

            hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
            weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
            # Use non-trivial per-channel weight scales to properly test the weight dequantization path.
            # weight_scale=1.0 would mask bugs where the kernel ignores weight dequantization.
            weight_scale = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
            masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

            q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
            q_a_all = q_a_all.contiguous()
            qa_scale_all = qa_scale_all.contiguous()

            ref = native_w8a8_per_channel_batch_matmul(
                q_a_all, weight, qa_scale_all, weight_scale, output_dtype=out_dtype
            )

            weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

            out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all),
                (weight_marlin, weight_scale),
                out,
                masked_m,
                M_alloc,
               # config={'MODE': 1000},
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
                # Tolerance accounts for BF16 output precision limits.
                # BF16 rounding error is ~0.39% max; kernel matches FP32 reference within BF16 precision.
                if not torch.allclose(ref_e, out_e, rtol=1e-2, atol=1e-2):
                    precision_ok = False
                    bad_expert = e
                    break

            cos_sim = torch.nn.functional.cosine_similarity(
                ref[:, :M, :].flatten().float(), out[:, :M, :].flatten().float(), dim=0
            ).item()

            print(f"  deepgemm: shape={out.shape}, min={out[:, :M, :].min().item():.2f}, max={out[:, :M, :].max().item():.2f}")
            print(f"  PyTorch ref: shape={ref.shape}, min={ref[:, :M, :].min().item():.2f}, max={ref[:, :M, :].max().item():.2f}")
            print(f"  Cosine Similarity: {cos_sim:.6f}")
            print(f"  max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")

            if precision_ok:
                print(f"  Status: PASS (all {E} experts match)")
            else:
                print(f"  Status: FAIL (expert {bad_expert} mismatch)")


def test_performance():
    """Performance test: measure latency for different batch sizes."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST: deepgemm m_grouped_w8a8_gemm_nt_masked throughput")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, K, N = 256, 3072, 3072
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
        # Use non-trivial per-channel weight scales (same rationale as test_precision).
        weight_scale = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
        masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

        q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
        q_a_all = q_a_all.contiguous()
        qa_scale_all = qa_scale_all.contiguous()

        weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

        out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()

        for _ in range(n_warmup):
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all), (weight_marlin, weight_scale), out, masked_m, M_alloc,# config={'MODE': 1000}
            )
        torch.cuda.synchronize()

        latencies = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all), (weight_marlin, weight_scale), out, masked_m, M_alloc, #config={'MODE': 1000}
            )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        throughput = (E * M) / (avg_latency / 1000)

        print(f"{M:>6} | {avg_latency:>14.3f} | {throughput:>22.0f} | PASS")


def test_weight_packing():
    """Test weight8bit_nt_kpack2_marlin packing format."""
    print("\n" + "=" * 80)
    print("TEST: weight8bit_nt_kpack2_marlin packing format")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, N, K = 4, 3072, 3072

    w_orig = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
    w_packed_list = [weight8bit_nt_kpack2_marlin(w_orig[i]).contiguous() for i in range(E)]
    w_packed = torch.stack(w_packed_list)

    print(f"Original weight: {w_orig.shape}")
    print(f"Packed weight:   {w_packed.shape}  [E, N/16, K*16] = [{E}, {N//16}, {K*16}]")
    print(f"Element count match: {w_packed.numel() == w_orig.numel()}")


def test_n_values():
    """Test different N values to find which ones work correctly."""
    print("\n" + "=" * 80)
    print("TEST: N value compatibility")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, M, K = 4, 8, 3072
    M_alloc = 256
    out_dtype = torch.bfloat16

    N_values = [256, 384, 512, 768, 1024, 2048, 3072]

    print(f"\n{'N':>6} | {'Cosine Sim':>12} | {'AllClose':>10} | Status")
    print("-" * 45)

    for N in N_values:
        torch.manual_seed(42 + N)
        hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
        weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
        # Use non-trivial per-channel weight scales (same rationale as test_precision).
        weight_scale = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
        masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

        q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)

        A = q_a_all.float()
        B = weight.float()
        ref = (qa_scale_all.float() * torch.bmm(A, B.transpose(1, 2)) * weight_scale.float().transpose(1, 2)).to(out_dtype)

        w_packed = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

        out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
        try:
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all), (w_packed, weight_scale), out, masked_m, M_alloc, config={'MODE': 1000}
            )
            torch.cuda.synchronize()
            cos_sim = torch.nn.functional.cosine_similarity(
                ref[:, :M, :].flatten().float(), out[:, :M, :].flatten().float(), dim=0
            ).item()
            # Tolerance accounts for BF16 output precision limits (~0.39% max rounding error).
            allclose = torch.allclose(ref[:, :M, :].float(), out[:, :M, :].float(), rtol=1e-2, atol=1e-2)
            status = "PASS" if allclose else "FAIL"
            print(f"{N:>6} | {cos_sim:>12.6f} | {str(allclose):>10} | {status}")
        except Exception as e:
            print(f"{N:>6} | {'ERROR':>12} | {'ERROR':>10} | {str(e)[:80]}")


def test_n_values_detailed():
    """Detailed N value scan around the N=384 failure boundary.

    Tests N in steps of 32 from 128 to 1024, plus a few larger values,
    to find the exact boundary of which N values the masked kernel supports.
    Also tests with multiple E and M values to rule out confounding factors.
    """
    print("\n" + "=" * 80)
    print("TEST: Detailed N value scan (masked kernel)")
    print("=" * 80)

    torch.cuda.empty_cache()

    K = 3072
    out_dtype = torch.bfloat16
    # Step by 32 around the failure zone, then by 64, then by 256
    N_values = (
        list(range(128, 640, 32))    # 128, 160, 192, ..., 608
        + list(range(640, 1280, 64))  # 640, 704, ..., 1216
        + [2048, 3072, 4096]          # larger values
    )

    # Test with two (E, M) combos to check if failure is N-specific
    configs = [
        {"E": 4, "M": 8, "label": "E=4,M=8"},
        {"E": 32, "M": 128, "label": "E=32,M=128"},
    ]

    for cfg in configs:
        E, M, label = cfg["E"], cfg["M"], cfg["label"]
        M_alloc = align_up(M, 256)

        print(f"\n--- {label} (K={K}, M_alloc={M_alloc}) ---")
        print(f"{'N':>6} | {'Cosine Sim':>12} | {'max_rel':>10} | {'AllClose':>10} | Status")
        print("-" * 60)

        for N in N_values:
            torch.manual_seed(42 + N)
            try:
                hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
                weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
                weight_scale = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
                masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

                q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)

                A = q_a_all.float()
                B = weight.float()
                ref = (qa_scale_all.float() * torch.bmm(A, B.transpose(1, 2)) * weight_scale.float().transpose(1, 2)).to(out_dtype)

                w_packed = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

                out = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
                m_grouped_w8a8_gemm_nt_masked(
                    (q_a_all, qa_scale_all), (w_packed, weight_scale), out, masked_m, M_alloc, config={'MODE': 1000}
                )
                torch.cuda.synchronize()

                cos_sim = torch.nn.functional.cosine_similarity(
                    ref[:, :M, :].flatten().float(), out[:, :M, :].flatten().float(), dim=0
                ).item()

                abs_diff = (ref[:, :M, :].float() - out[:, :M, :].float()).abs()
                max_rel = (abs_diff / ref[:, :M, :].float().abs().clamp_min(1e-6)).max().item()

                allclose = torch.allclose(ref[:, :M, :].float(), out[:, :M, :].float(), rtol=1e-2, atol=1e-2)
                status = "PASS" if allclose else "FAIL"
                print(f"{N:>6} | {cos_sim:>12.6f} | {max_rel:>10.6f} | {str(allclose):>10} | {status}")

                del hidden_states, weight, weight_scale, masked_m, q_a_all, qa_scale_all, w_packed, out, ref
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"{N:>6} | {'ERROR':>12} | {'ERROR':>10} | {'ERROR':>10} | {str(e)[:60]}")
                torch.cuda.empty_cache()


def test_api_summary():
    """Summary of both masked and contiguous kernel APIs."""
    print("\n" + "=" * 80)
    print("SUMMARY: DeepGEMM W8A8 Kernel APIs")
    print("=" * 80)
    print("""
=== Masked Kernel: m_grouped_w8a8_gemm_nt_masked ===
  m_grouped_w8a8_gemm_nt_masked(
      a: tuple[torch.Tensor, torch.Tensor],  # (input_int8 [E,M_alloc,K], input_scale [E,M_alloc,1])
      b: tuple[torch.Tensor, torch.Tensor],  # (weight_packed [E,N/16,K*16], weight_scale [E,N,1])
      d: torch.Tensor,                        # output [E, M_alloc, N] bf16
      masked_m: torch.Tensor,                 # [E] int32, valid tokens per expert
      expected_m_per_group: int,              # M_alloc (aligned to 256)
      config: Optional[Dict] = {'MODE': 1000}
  ) -> torch.Tensor

=== Contiguous Kernel: m_grouped_i8_gemm_nt_contiguous ===
  m_grouped_i8_gemm_nt_contiguous(
      a: tuple[torch.Tensor, torch.Tensor],  # (input_int8 [M,K], input_scale [M])
      b: tuple[torch.Tensor, torch.Tensor],  # (weight_packed [E,N/16,K*16], weight_scale [E,N])
      output: torch.Tensor,                   # [M, N] bf16
      m_indices: torch.Tensor,                # [M] int32, expert id per token (-1 = skip)
      config: Optional[Dict] = {'MODE': 1000}
  ) -> torch.Tensor

Weight packing (both kernels):
  weight8bit_nt_kpack2_marlin(weight[e])  # [N, K] -> [N/16, K*16]

EP path dimensions (MiniMax M2):
  - E = 256 (local experts)
  - K = 3072 (hidden size)
  - N = 3072 (2 * intermediate_size, full unsharded)
  - M_alloc aligned to 256

Note: weight8bit_nt_kpack2_marlin1 is NOT compatible with these kernels.
      Use weight8bit_nt_kpack2_marlin instead.
""")


def _generate_contiguous_data(num_experts: int, m_per_expert: int, n: int, k: int, out_dtype: torch.dtype):
    """Generate test data for contiguous kernel.

    Returns:
        m_total: total M (sum of aligned m_per_expert)
        a_int8: [m_total, k] int8
        a_scale: [m_total] float32
        weight: [num_experts, n, k] int8
        weight_scale: [num_experts, n] float32
        m_indices: [m_total] int32
        ref_output: [m_total, n] bf16 (PyTorch reference)
    """
    m_aligned = align_up(m_per_expert, 256)
    m_total = num_experts * m_aligned

    torch.manual_seed(42)
    hidden_states = (torch.randn((m_total, k), device='cuda', dtype=out_dtype) * 5).contiguous()
    weight = torch.randint(-127, 127, (num_experts, n, k), dtype=torch.int8, device='cuda').contiguous()
    weight_scale = (torch.randn((num_experts, n), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()

    m_indices = torch.empty(m_total, device='cuda', dtype=torch.int32)
    for e in range(num_experts):
        start = e * m_aligned
        m_indices[start:start + m_per_expert] = e
        m_indices[start + m_per_expert:start + m_aligned] = -1

    q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
    a_int8 = q_a_all.contiguous()
    a_scale = qa_scale_all.squeeze(-1).contiguous()

    ref_output = torch.zeros((m_total, n), device='cuda', dtype=out_dtype)
    for e in range(num_experts):
        start = e * m_aligned
        end = start + m_per_expert
        if end <= start:
            continue
        a_e = a_int8[start:end].float()
        w_e = weight[e].float()
        s_a = a_scale[start:end].unsqueeze(-1).float()
        s_w = weight_scale[e].float()
        ref_output[start:end] = (s_a * torch.mm(a_e, w_e.transpose(0, 1)) * s_w).to(out_dtype)

    return m_total, a_int8, a_scale, weight, weight_scale, m_indices, ref_output


def test_contiguous_precision():
    """Precision test: compare contiguous kernel output with PyTorch reference."""
    print("\n" + "=" * 80)
    print("PRECISION TEST: deepgemm m_grouped_i8_gemm_nt_contiguous vs PyTorch reference")
    print("=" * 80)

    torch.cuda.empty_cache()

    # Test both E=256 (single rank) and E=32 (EP-8 mode: 256/8=32 experts per rank)
    E_values = [256, 32]
    K, N = 3072, 3072
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024]

    for E in E_values:
        print(f"\n{'=' * 40} E = {E} {'=' * 40}")
        for M in M_values:
            print(f"\n--- M_per_expert = {M} ---")
            torch.cuda.empty_cache()

            m_total, a_int8, a_scale, weight, weight_scale, m_indices, ref_output = \
                _generate_contiguous_data(E, M, N, K, out_dtype)

            m_aligned = align_up(M, 256)

            weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

            out = torch.zeros((m_total, N), device='cuda', dtype=out_dtype).contiguous()
            m_grouped_i8_gemm_nt_contiguous(
                (a_int8, a_scale),
                (weight_marlin, weight_scale),
                out,
                m_indices,
                config={'MODE': 1000},
            )
            torch.cuda.synchronize()

            precision_ok = True
            max_abs = 0.0
            bad_expert = -1

            for e in range(E):
                start = e * m_aligned
                end = start + M
                ref_e = ref_output[start:end]
                out_e = out[start:end]
                abs_diff = (ref_e.float() - out_e.float()).abs()
                cur_max_abs = abs_diff.max().item()
                max_abs = max(max_abs, cur_max_abs)
                if not torch.allclose(ref_e, out_e, rtol=1e-2, atol=1e-2):
                    precision_ok = False
                    bad_expert = e
                    break

            cos_sim = torch.nn.functional.cosine_similarity(
                ref_output.flatten().float(), out.flatten().float(), dim=0
            ).item()

            print(f"  deepgemm: shape={out.shape}, min={out.min().item():.2f}, max={out.max().item():.2f}")
            print(f"  PyTorch ref: shape={ref_output.shape}, min={ref_output.min().item():.2f}, max={ref_output.max().item():.2f}")
            print(f"  Cosine Similarity: {cos_sim:.6f}")
            print(f"  max_abs={max_abs:.6g}")
            print(f"  weight_scale range: [{weight_scale.min().item():.4f}, {weight_scale.max().item():.4f}]")

            if precision_ok:
                print(f"  Status: PASS (all {E} experts match)")
            else:
                print(f"  Status: FAIL (expert {bad_expert} mismatch)")


def test_contiguous_performance():
    """Performance test: measure contiguous kernel latency."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST: deepgemm m_grouped_i8_gemm_nt_contiguous throughput")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, K, N = 256, 3072, 3072
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024, 4096]
    n_warmup = 5
    n_measure = 20

    print(f"\n{'M':>6} | {'Latency (ms)':>14} | {'Throughput (tokens/s)':>22} | Status")
    print("-" * 65)

    for M in M_values:
        m_total, a_int8, a_scale, weight, weight_scale, m_indices, _ = \
            _generate_contiguous_data(E, M, N, K, out_dtype)

        m_aligned = align_up(M, 256)
        weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])
        out = torch.zeros((m_total, N), device='cuda', dtype=out_dtype).contiguous()

        for _ in range(n_warmup):
            m_grouped_i8_gemm_nt_contiguous((a_int8, a_scale), (weight_marlin, weight_scale), out, m_indices, config={'MODE': 1000})
        torch.cuda.synchronize()

        latencies = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            m_grouped_i8_gemm_nt_contiguous((a_int8, a_scale), (weight_marlin, weight_scale), out, m_indices, config={'MODE': 1000})
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        throughput = (E * M) / (avg_latency / 1000)

        print(f"{M:>6} | {avg_latency:>14.3f} | {throughput:>22.0f} | PASS")


def test_contiguous_m_indices():
    """Test that m_indices correctly routes tokens to experts and skips padding."""
    print("\n" + "=" * 80)
    print("TEST: contiguous kernel m_indices routing and padding")
    print("=" * 80)

    torch.cuda.empty_cache()

    E, K, N, M = 4, 3072, 3072, 8
    M_alloc = align_up(M, 256)
    out_dtype = torch.bfloat16

    m_total, a_int8, a_scale, weight, weight_scale, m_indices, ref_output = \
        _generate_contiguous_data(E, M, N, K, out_dtype)

    weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])
    out = torch.zeros((m_total, N), device='cuda', dtype=out_dtype).contiguous()
    m_grouped_i8_gemm_nt_contiguous(
        (a_int8, a_scale), (weight_marlin, weight_scale), out, m_indices, config={'MODE': 1000}
    )
    torch.cuda.synchronize()

    print(f"m_indices: first 20 = {m_indices[:20].tolist()}")
    print(f"m_indices unique: {torch.unique(m_indices).tolist()}")

    all_pass = True
    for e in range(E):
        start = e * M_alloc
        valid_end = start + M
        pad_end = start + M_alloc

        valid_match = torch.allclose(ref_output[start:valid_end].float(), out[start:valid_end].float(), rtol=1e-2, atol=1e-2)
        pad_is_zero = (out[valid_end:pad_end] == 0).all().item()

        status = "PASS" if (valid_match and pad_is_zero) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Expert {e}: valid_region={status}, padding_zero={'PASS' if pad_is_zero else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    import sys
    tests = {
        "precision": test_precision,
        "performance": test_performance,
        "contiguous_precision": test_contiguous_precision,
        "contiguous_performance": test_contiguous_performance,
        "contiguous_m_indices": test_contiguous_m_indices,
        "weight_packing": test_weight_packing,
        "n_values": test_n_values,
        "n_values_detailed": test_n_values_detailed,
        "api_summary": test_api_summary,
    }

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {list(tests.keys())}")
    else:
        print("Run tests individually to avoid GPU state contamination:")
        print("  python test_moe_deepgemm_w8a8.py precision")
        print("  python test_moe_deepgemm_w8a8.py performance")
        print("  python test_moe_deepgemm_w8a8.py contiguous_precision")
        print("  python test_moe_deepgemm_w8a8.py contiguous_performance")
        print("  python test_moe_deepgemm_w8a8.py contiguous_m_indices")
        print("  python test_moe_deepgemm_w8a8.py weight_packing")
        print("  python test_moe_deepgemm_w8a8.py n_values")
        print("  python test_moe_deepgemm_w8a8.py n_values_detailed")
        print("  python test_moe_deepgemm_w8a8.py api_summary")