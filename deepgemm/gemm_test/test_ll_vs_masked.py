#!/usr/bin/env python3
"""
Compare m_grouped_w8a8_gemm_nt_masked_ll vs m_grouped_w8a8_gemm_nt_masked.

Both paths share the same input/weight/scales; only weight packing and scale
dimensionality differ.  Dimensions are chosen to satisfy LL kernel restrictions:
  E in {1,16,32}, N in {3072,4096,6144,7168}, K in {1536,2048,3072,6144,7168}.
"""

import torch
import os
import time

os.environ["LD_LIBRARY_PATH"] = "/zkjh/drivers/ib_plugin/topo_lib/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from deepgemm import m_grouped_w8a8_gemm_nt_masked
from deepgemm.m_group_gemm import (
    m_grouped_w8a8_gemm_nt_masked_ll,
    pack_int8_weight_enk_to_w6_low_latency,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import weight8bit_nt_kpack2_marlin
from lmslim.layers.gemm.int8_utils import per_token_quant_int8


def align_up(x: int, align: int = 256) -> int:
    return ((x + align - 1) // align) * align


def parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(item) for item in value.replace(",", " ").split()]


def make_decode_masked_m(batch_size: int, top_k: int, experts: int, device: str = "cuda") -> torch.Tensor:
    """Random decode-style routing counts: batch_size tokens, top_k experts per token."""
    counts = torch.zeros((experts,), dtype=torch.int32)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000 + batch_size * 17 + experts)
    for _ in range(batch_size):
        routed = torch.randperm(experts, generator=gen)[:top_k]
        counts[routed] += 1
    return counts.to(device=device)


def test_compare():
    """Compare LL output vs masked output (same inputs, different packing)."""
    print("\n" + "=" * 80)
    print("COMPARISON: m_grouped_w8a8_gemm_nt_masked_ll vs m_grouped_w8a8_gemm_nt_masked")
    print("=" * 80)

    torch.cuda.empty_cache()

    # Must satisfy LL restrictions: E in {1,16,32}, N in {3072,4096,6144,7168}
    E_values = [32, 16]
    K, N = 3072, 3072
    out_dtype = torch.bfloat16
    M_values = [8, 128, 1024]

    for E in E_values:
        print(f"\n{'=' * 40} E = {E} {'=' * 40}")
        print(f"{'M':>6} | {'cos_ll_vs_ref':>14} | {'cos_masked_vs_ref':>14} | {'cos_ll_vs_masked':>16} | Status")
        print("-" * 72)

        for M in M_values:
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            M_alloc = align_up(M, 256)

            # ---- common data ----
            hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
            weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
            masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

            q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
            q_a_all = q_a_all.contiguous()
            qa_scale_all = qa_scale_all.contiguous()

            # PyTorch reference (3-D scales)
            w_scale_3d = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
            A = q_a_all.float()
            B = weight.float()
            ref = (qa_scale_all.float() * torch.bmm(A, B.transpose(1, 2)) * w_scale_3d.float().transpose(1, 2)).to(out_dtype)

            # ---- LL path (6-D weights, 2-D scales) ----
            qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()
            w_scale_2d = w_scale_3d.squeeze(-1).contiguous()
            weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)

            out_ll = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
            m_grouped_w8a8_gemm_nt_masked_ll(
                (q_a_all, qa_scale_2d),
                (weight_ll, w_scale_2d),
                out_ll,
                masked_m,
                M_alloc,
                block_wise=False,
                cu=128,
            )
            torch.cuda.synchronize()

            # ---- masked path (Marlin weights, 3-D scales) ----
            weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

            out_masked = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all),
                (weight_marlin, w_scale_3d),
                out_masked,
                masked_m,
                M_alloc,
                config={'MODE': 1000},
            )
            torch.cuda.synchronize()

            # ---- compare ----
            flat_ref = ref[:, :M, :].flatten().float()
            flat_ll = out_ll[:, :M, :].flatten().float()
            flat_masked = out_masked[:, :M, :].flatten().float()

            cos_ll_ref = torch.nn.functional.cosine_similarity(flat_ll, flat_ref, dim=0).item()
            cos_masked_ref = torch.nn.functional.cosine_similarity(flat_masked, flat_ref, dim=0).item()
            cos_ll_masked = torch.nn.functional.cosine_similarity(flat_ll, flat_masked, dim=0).item()

            allclose_ll = torch.allclose(ref[:, :M, :], out_ll[:, :M, :], rtol=1e-2, atol=1e-2)
            allclose_masked = torch.allclose(ref[:, :M, :], out_masked[:, :M, :], rtol=1e-2, atol=1e-2)
            pass_ll = "PASS" if allclose_ll else "FAIL"
            pass_masked = "PASS" if allclose_masked else "FAIL"

            print(f"{M:>6} | {cos_ll_ref:>14.6f} | {cos_masked_ref:>14.6f} | {cos_ll_masked:>16.6f} | LL:{pass_ll} masked:{pass_masked}")

            if not allclose_ll:
                abs_diff = (ref[:, :M, :].float() - out_ll[:, :M, :].float()).abs()
                print(f"         LL max_abs={abs_diff.max().item():.4f}, masked max_abs={(ref[:,:M,:].float()-out_masked[:,:M,:].float()).abs().max().item():.4f}")


def _run_perf_config(E: int, K: int, N: int, M_values: list[int], out_dtype, n_warmup: int, n_measure: int, cu: int) -> None:
    """Run performance comparison for a single (K, N) config."""
    print(f"\n{'=' * 40} K={K}, N={N} {'=' * 40}")
    print(f"{'M':>6} | {'LL (ms)':>10} | {'masked (ms)':>12} | {'speedup':>7} | {'LL tokens/s':>14} | {'masked tokens/s':>16}")
    print("-" * 82)

    for M in M_values:
        torch.manual_seed(42)
        M_alloc = align_up(M, 256)

        hidden_states = (torch.randn((E, M_alloc, K), device='cuda', dtype=out_dtype) * 5).contiguous()
        weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device='cuda').contiguous()
        w_scale_3d = (torch.randn((E, N, 1), device='cuda', dtype=torch.float32).abs() + 0.01).contiguous()
        masked_m = torch.full((E,), M, dtype=torch.int32, device='cuda').contiguous()

        q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
        q_a_all = q_a_all.contiguous()
        qa_scale_all = qa_scale_all.contiguous()

        # LL preparation
        qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()
        w_scale_2d = w_scale_3d.squeeze(-1).contiguous()
        weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)

        # masked preparation
        weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

        out_ll = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()
        out_masked = torch.zeros((E, M_alloc, N), device='cuda', dtype=out_dtype).contiguous()

        # ---- LL performance ----
        for _ in range(n_warmup):
            m_grouped_w8a8_gemm_nt_masked_ll(
                (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
            )
        torch.cuda.synchronize()

        latencies_ll = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            m_grouped_w8a8_gemm_nt_masked_ll(
                (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
            )
            torch.cuda.synchronize()
            latencies_ll.append((time.perf_counter() - t0) * 1000)

        # ---- masked performance ----
        for _ in range(n_warmup):
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                out_masked, masked_m, M_alloc, config={'MODE': 1000},
            )
        torch.cuda.synchronize()

        latencies_masked = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            m_grouped_w8a8_gemm_nt_masked(
                (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                out_masked, masked_m, M_alloc, config={'MODE': 1000},
            )
            torch.cuda.synchronize()
            latencies_masked.append((time.perf_counter() - t0) * 1000)

        avg_ll = sum(latencies_ll) / len(latencies_ll)
        avg_masked = sum(latencies_masked) / len(latencies_masked)
        speedup = avg_masked / avg_ll
        tp_ll = (E * M) / (avg_ll / 1000)
        tp_masked = (E * M) / (avg_masked / 1000)

        # TFlops & bandwidth (matching test_deepgemm_masked_w8a8_bandw.py formula)
        valid_m = E * M
        computes = 2 * valid_m * N * K
        data_bytes = valid_m * K + K * N * E + valid_m * N * 2  # A(int8) + B(int8,all experts) + C(bf16)

        tflops_ll = computes / (avg_ll / 1000) / 1e12
        tflops_masked = computes / (avg_masked / 1000) / 1e12
        bw_ll = data_bytes / (avg_ll / 1000) / 1e9
        bw_masked = data_bytes / (avg_masked / 1000) / 1e9

        print(f"{M:>6} | {avg_ll:>10.3f} | {avg_masked:>12.3f} | {speedup:>7.2f} | {tp_ll:>14.0f} | {tp_masked:>16.0f}")
        print(f"       | {'LL TFlops:':>10} {tflops_ll:>6.2f}, {'BW:':>3} {bw_ll:>7.1f} GB/s | {'masked TFlops:':>14} {tflops_masked:>6.2f}, {'BW:':>3} {bw_masked:>7.1f} GB/s")


def test_performance():
    """Performance comparison: LL vs masked, covering both MoE GEMM layers."""
    print("\n" + "=" * 80)
    print("PERFORMANCE: m_grouped_w8a8_gemm_nt_masked_ll vs m_grouped_w8a8_gemm_nt_masked")
    print("=" * 80)

    torch.cuda.empty_cache()

    E = int(os.environ.get("E", "32"))
    out_dtype = torch.bfloat16
    M_values = [1, 4, 8, 16, 20, 32, 64, 128, 1024, 4096]
    n_warmup = 5
    n_measure = 20
    cu = int(os.environ.get("CU", "128"))

    # MiniMax-M2.5 MoE GEMM configs
    # EP=8 only:        local_E=32
    # TP=8 only:        E=256 (require BLOCK_N=128, K_SCALE_RANGE=64)
    # EP=8 + TP=8:      local_E=32 + local_N/local_K
    K_N_configs = [
        # ---- EP=8 only ----
        (3072, 3072),   # GEMM 1: gate+up (E=32, K=3072, N=3072)
        (1536, 3072),   # GEMM 2: down    (E=32, K=1536, N=3072)
        # ---- EP=8 + TP=8 (BLOCK_N=128, K_SCALE_RANGE=64 needed) ----
        (3072, 384),    # GEMM 1 + TP=8:  gate+up (E=32, K=3072, N=3072/8=384)
        (192,  3072),   # GEMM 2 + TP=8:  down    (E=32, K=1536/8=192, N=3072)
    ]

    print(f"Config: E={E}, CU={cu}, warmup={n_warmup}, measure={n_measure}")
    print("Note: (K=3072,N=384) and (K=192,N=3072) need BLOCK_N=128, K_SCALE_RANGE=64")
    print("      For TP-only (E=256): set E=256 env var (e.g. E=256 K=3072 N=384)")

    for K, N in K_N_configs:
        _run_perf_config(E, K, N, M_values, out_dtype, n_warmup, n_measure, cu)


def test_decode_performance():
    """Decode-style benchmark: B*top_k routed rows distributed across experts."""
    print("\n" + "=" * 80)
    print("DECODE DISTRIBUTION: m_grouped_w8a8_gemm_nt_masked_ll vs masked")
    print("=" * 80)

    torch.cuda.empty_cache()

    E_values = [32, 16]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    top_k = 8
    K, N = 3072, 3072
    out_dtype = torch.bfloat16
    n_warmup = 10
    n_measure = 50
    cu = int(os.environ.get("CU", "128"))

    print(f"Config: E_VALUES={E_values}, batch_sizes={batch_sizes}, top_k={top_k}, N={N}, K={K}, CU={cu}")

    for E in E_values:
        torch.manual_seed(42 + E)
        M_alloc = 256

        weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device="cuda").contiguous()
        w_scale_3d = (torch.randn((E, N, 1), device="cuda", dtype=torch.float32).abs() + 0.01).contiguous()
        w_scale_2d = w_scale_3d.squeeze(-1).contiguous()
        weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)
        weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

        print(f"\n{'=' * 40} E = {E}, top_k = {top_k} {'=' * 40}")
        print(
            f"{'B':>4} | {'routed':>6} | {'active':>6} | {'max_m':>5} | "
            f"{'LL (ms)':>9} | {'masked (ms)':>11} | {'speedup':>7} | "
            f"{'LL tok/s':>11} | {'masked tok/s':>13}"
        )
        print("-" * 92)

        for batch_size in batch_sizes:
            torch.manual_seed(100 + batch_size + E)

            masked_m = make_decode_masked_m(batch_size, top_k, E).contiguous()
            routed_tokens = int(masked_m.sum().item())
            active_experts = int((masked_m > 0).sum().item())
            max_m = int(masked_m.max().item())

            hidden_states = (torch.randn((E, M_alloc, K), device="cuda", dtype=out_dtype) * 5).contiguous()
            q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
            q_a_all = q_a_all.contiguous()
            qa_scale_all = qa_scale_all.contiguous()
            qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()

            out_ll = torch.zeros((E, M_alloc, N), device="cuda", dtype=out_dtype).contiguous()
            out_masked = torch.zeros((E, M_alloc, N), device="cuda", dtype=out_dtype).contiguous()

            for _ in range(n_warmup):
                m_grouped_w8a8_gemm_nt_masked_ll(
                    (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                    out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
                )
            torch.cuda.synchronize()

            latencies_ll = []
            for _ in range(n_measure):
                t0 = time.perf_counter()
                m_grouped_w8a8_gemm_nt_masked_ll(
                    (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                    out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
                )
                torch.cuda.synchronize()
                latencies_ll.append((time.perf_counter() - t0) * 1000)

            for _ in range(n_warmup):
                m_grouped_w8a8_gemm_nt_masked(
                    (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                    out_masked, masked_m, M_alloc, config={"MODE": 1000},
                )
            torch.cuda.synchronize()

            latencies_masked = []
            for _ in range(n_measure):
                t0 = time.perf_counter()
                m_grouped_w8a8_gemm_nt_masked(
                    (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                    out_masked, masked_m, M_alloc, config={"MODE": 1000},
                )
                torch.cuda.synchronize()
                latencies_masked.append((time.perf_counter() - t0) * 1000)

            avg_ll = sum(latencies_ll) / len(latencies_ll)
            avg_masked = sum(latencies_masked) / len(latencies_masked)
            speedup = avg_masked / avg_ll
            tp_ll = routed_tokens / (avg_ll / 1000)
            tp_masked = routed_tokens / (avg_masked / 1000)

            print(
                f"{batch_size:>4} | {routed_tokens:>6} | {active_experts:>6} | {max_m:>5} | "
                f"{avg_ll:>9.3f} | {avg_masked:>11.3f} | {speedup:>7.2f} | "
                f"{tp_ll:>11.0f} | {tp_masked:>13.0f}"
            )


def test_decode_d_performance():
    """D-end style decode benchmark using concurrency-sized decode batches.

    Supports single (K, N) via env vars, or multiple K values via K_VALUES env var.
    When K_VALUES is set, each K is paired with the same N.
    """
    print("\n" + "=" * 80)
    print("D-END DECODE SIMULATION: one generated token per active request")
    print("=" * 80)

    torch.cuda.empty_cache()

    E_values = parse_int_list_env("E_VALUES", [32, 16])
    batch_sizes = parse_int_list_env("BATCHES", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    top_k = int(os.environ.get("TOP_K", "8"))
    K_values = parse_int_list_env("K_VALUES", [int(os.environ.get("K", "3072"))])
    N = int(os.environ.get("N", "3072"))
    output_len = int(os.environ.get("OUTPUT_LEN", "256"))
    cu = int(os.environ.get("CU", "128"))
    out_dtype = torch.bfloat16
    n_warmup = int(os.environ.get("WARMUP", "10"))
    n_measure = int(os.environ.get("MEASURE", "50"))

    for K in K_values:
        print(
            f"\nConfig: E_VALUES={E_values}, BATCHES={batch_sizes}, top_k={top_k}, "
            f"K={K}, N={N}, output_len={output_len}, CU={cu}"
        )
        print(
            "Note: LL/masked latency columns measure one MoE GEMM step; "
            "total columns estimate repeated decode steps by multiplying output_len."
        )

        for E in E_values:
            torch.manual_seed(42 + E + K)

            weight = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device="cuda").contiguous()
            w_scale_3d = (torch.randn((E, N, 1), device="cuda", dtype=torch.float32).abs() + 0.01).contiguous()
            w_scale_2d = w_scale_3d.squeeze(-1).contiguous()
            weight_ll = pack_int8_weight_enk_to_w6_low_latency(weight)
            weight_marlin = torch.stack([weight8bit_nt_kpack2_marlin(weight[i]).contiguous() for i in range(E)])

            print(f"\n{'=' * 40} E = {E}, K = {K}, N = {N}, top_k = {top_k} {'=' * 40}")
            print(
                f"{'B/conc':>6} | {'routed':>6} | {'active':>6} | {'max_m':>5} | {'M_alloc':>7} | "
                f"{'LL (ms)':>9} | {'masked (ms)':>11} | {'speedup':>7} | "
                f"{'LL total(ms)':>12} | {'masked total(ms)':>16} | "
                f"{'LL step tok/s':>13} | {'masked step tok/s':>17}"
            )
            print("-" * 146)

            for batch_size in batch_sizes:
                torch.manual_seed(100 + batch_size + E + K)

                masked_m = make_decode_masked_m(batch_size, top_k, E).contiguous()
                routed_tokens = int(masked_m.sum().item())
                active_experts = int((masked_m > 0).sum().item())
                max_m = int(masked_m.max().item())
                M_alloc = align_up(max(max_m, 1), 256)

                hidden_states = (torch.randn((E, M_alloc, K), device="cuda", dtype=out_dtype) * 5).contiguous()
                q_a_all, qa_scale_all = per_token_quant_int8(hidden_states)
                q_a_all = q_a_all.contiguous()
                qa_scale_all = qa_scale_all.contiguous()
                qa_scale_2d = qa_scale_all.squeeze(-1).contiguous()

                out_ll = torch.zeros((E, M_alloc, N), device="cuda", dtype=out_dtype).contiguous()
                out_masked = torch.zeros((E, M_alloc, N), device="cuda", dtype=out_dtype).contiguous()

                for _ in range(n_warmup):
                    m_grouped_w8a8_gemm_nt_masked_ll(
                        (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                        out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
                    )
                torch.cuda.synchronize()

                latencies_ll = []
                for _ in range(n_measure):
                    t0 = time.perf_counter()
                    m_grouped_w8a8_gemm_nt_masked_ll(
                        (q_a_all, qa_scale_2d), (weight_ll, w_scale_2d),
                        out_ll, masked_m, M_alloc, block_wise=False, cu=cu,
                    )
                    torch.cuda.synchronize()
                    latencies_ll.append((time.perf_counter() - t0) * 1000)

                for _ in range(n_warmup):
                    m_grouped_w8a8_gemm_nt_masked(
                        (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                        out_masked, masked_m, M_alloc, config={"MODE": 1000},
                    )
                torch.cuda.synchronize()

                latencies_masked = []
                for _ in range(n_measure):
                    t0 = time.perf_counter()
                    m_grouped_w8a8_gemm_nt_masked(
                        (q_a_all, qa_scale_all), (weight_marlin, w_scale_3d),
                        out_masked, masked_m, M_alloc, config={"MODE": 1000},
                    )
                    torch.cuda.synchronize()
                    latencies_masked.append((time.perf_counter() - t0) * 1000)

                avg_ll = sum(latencies_ll) / len(latencies_ll)
                avg_masked = sum(latencies_masked) / len(latencies_masked)
                speedup = avg_masked / avg_ll
                total_ll = avg_ll * output_len
                total_masked = avg_masked * output_len
                tp_ll = batch_size / (avg_ll / 1000)
                tp_masked = batch_size / (avg_masked / 1000)

                # TFlops & bandwidth
                valid_m = routed_tokens
                computes = 2 * valid_m * N * K
                data_bytes = valid_m * K + K * N * E + valid_m * N * 2

                tflops_ll = computes / (avg_ll / 1000) / 1e12
                tflops_masked = computes / (avg_masked / 1000) / 1e12
                bw_ll = data_bytes / (avg_ll / 1000) / 1e9
                bw_masked = data_bytes / (avg_masked / 1000) / 1e9

                print(
                    f"{batch_size:>6} | {routed_tokens:>6} | {active_experts:>6} | {max_m:>5} | {M_alloc:>7} | "
                    f"{avg_ll:>9.3f} | {avg_masked:>11.3f} | {speedup:>7.2f} | "
                    f"{total_ll:>12.1f} | {total_masked:>16.1f} | "
                    f"{tp_ll:>13.0f} | {tp_masked:>17.0f}"
                )
                print(
                    f"       | {'LL TFlops:':>10} {tflops_ll:>6.2f}, {'BW:':>3} {bw_ll:>7.1f} GB/s | {'masked TFlops:':>14} {tflops_masked:>6.2f}, {'BW:':>3} {bw_masked:>7.1f} GB/s"
                )


def test_real_model():
    """Test MiniMax-M2.5 EP=8+TP=8: both GEMM layers, E=32 only.

    GEMM 1 (gate+up): E=32, K=3072, N=384   (TP splits N: 3072/8=384)
    GEMM 2 (down):    E=32, K=192,  N=3072  (TP splits K: 1536/8=192)

    Requires BLOCK_N=128, K_SCALE_RANGE=64 in the kernel.
    """
    print("\n" + "=" * 80)
    print("REAL MODEL: MiniMax-M2.5 EP=8+TP=8 (E=32 only)")
    print("=" * 80)
    print("GEMM 1 (gate+up): E=32, K=3072, N=384")
    print("GEMM 2 (down):    E=32, K=192,  N=3072")
    print("=" * 80)

    os.environ["E"] = "32"
    os.environ["E_VALUES"] = "32"  # suppress E=16
    for label, K, N in [
        ("GEMM 1 (gate+up)", 3072, 384),
        ("GEMM 2 (down)",    192,  3072),
    ]:
        os.environ["K"] = str(K)
        os.environ["N"] = str(N)
        print(f"\n{'#' * 60}")
        print(f"# {label}: E=32, K={K}, N={N}")
        print(f"{'#' * 60}")
        test_decode_d_performance()


if __name__ == "__main__":
    import sys
    tests = {
        "compare": test_compare,
        "performance": test_performance,
        "decode": test_decode_performance,
        "decode_d": test_decode_d_performance,
        "real_model": test_real_model,
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
        print("  python test_ll_vs_masked.py compare")
        print("  python test_ll_vs_masked.py performance")
        print("  python test_ll_vs_masked.py decode")
        print("  python test_ll_vs_masked.py decode_d")
        print("  python test_ll_vs_masked.py real_model")
        print("")
        print("Environment variables for decode_d / real_model:")
        print("  E_VALUES=\"32 16\"         # expert counts (default: 32 16)")
        print("  BATCHES=\"1 2 4 ...\"      # concurrency levels (default: 1 2 4 8 16 32 64 128 256 512)")
        print("  K=3072                    # K dimension (default: 3072)")
        print("  K_VALUES=\"3072 1536\"     # multiple K values for decode_d (default: unset)")
        print("  N=3072                    # N dimension (default: 3072)")
        print("  TOP_K=8                   # top-k routing (default: 8)")
        print("  OUTPUT_LEN=1024           # decode length for total time estimation (default: 256)")
        print("  CU=128                    # compute units for LL kernel (default: 128)")
        print("  WARMUP=10                 # warmup iterations (default: 10)")
        print("  MEASURE=50                # measurement iterations (default: 50)")
        print("")
        print("Examples:")
        print("  # Single GEMM layer")
        print("  OUTPUT_LEN=1024 CU=128 python test_ll_vs_masked.py decode_d")
        print("  # Both GEMM layers with EP=8 dimensions")
        print("  OUTPUT_LEN=1024 CU=80 python test_ll_vs_masked.py real_model")
        print("  # Test second GEMM (down projection) only")
        print("  K=1536 OUTPUT_LEN=1024 CU=128 python test_ll_vs_masked.py decode_d")
