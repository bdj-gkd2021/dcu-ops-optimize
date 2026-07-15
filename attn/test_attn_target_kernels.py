#!/usr/bin/env python3
import argparse
import math
import random

import torch

from flash_attn import flash_attn_with_kvcache, vllm_flash_attn_with_kvcache


CASES = [
    {"name": "case1", "batch": 1, "q_len": 1, "kv_len": 1025, "q_head": 16, "kv_head": 16, "head_dim": 64},
    {"name": "case2", "batch": 20, "q_len": 1, "kv_len": 16384, "q_head": 28, "kv_head": 4, "head_dim": 64},
    {"name": "case3", "batch": 1, "q_len": 1, "kv_len": 51200, "q_head": 10, "kv_head": 2, "head_dim": 64},
]


def reference_attention(q, k, v, causal=True):
    out_dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    repeat = q.shape[2] // k.shape[2]
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    score = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).tril(
            diagonal=k.shape[1] - q.shape[1]
        )
        score = score.masked_fill(~mask[None, None, :, :], float("-inf"))
    prob = torch.softmax(score, dim=-1, dtype=torch.float32)
    out = torch.einsum("bhqk,bkhd->bqhd", prob, v)
    return out.to(out_dtype)


def make_paged_cache_from_full(full_k, full_v):
    batch, kv_len, kv_head, head_dim = full_k.shape
    block_size = 64
    nblocks = math.ceil(kv_len / block_size)
    num_blocks = nblocks * batch * 3
    device = full_k.device
    k_paged = torch.zeros(num_blocks, block_size, kv_head, head_dim, device=device, dtype=full_k.dtype)
    v_paged = torch.zeros(num_blocks, block_size, kv_head, head_dim, device=device, dtype=full_v.dtype)
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device)[: batch * nblocks]
    block_table = block_table.view(batch, nblocks)

    for b in range(batch):
        for blk in range(nblocks):
            src_start = blk * block_size
            src_end = min(src_start + block_size, kv_len)
            dst = int(block_table[b, blk].item())
            span = src_end - src_start
            k_paged[dst, :span] = full_k[b, src_start:src_end]
            v_paged[dst, :span] = full_v[b, src_start:src_end]

    return k_paged, v_paged, block_table


def run_case(case, dtype, warmup, iters):
    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    batch = case["batch"]
    q_len = case["q_len"]
    kv_len = case["kv_len"]
    q_head = case["q_head"]
    kv_head = case["kv_head"]
    head_dim = case["head_dim"]

    q = torch.randn(batch, q_len, q_head, head_dim, device=device, dtype=dtype)
    k_contig = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    v_contig = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    k_paged, v_paged, block_table = make_paged_cache_from_full(k_contig, v_contig)
    k_ref = k_contig
    v_ref = v_contig
    ref = reference_attention(q, k_ref, v_ref, causal=True)

    def bench(fn):
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end) / iters

    out_contig, ms_contig = bench(
        lambda: flash_attn_with_kvcache(
            q,
            k_contig,
            v_contig,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=0,
            return_softmax_lse=False,
        )
    )

    out_paged, ms_paged = bench(
        lambda: vllm_flash_attn_with_kvcache(
            q,
            k_paged,
            v_paged,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=0,
            return_softmax_lse=False,
        )
    )

    contig_ok = torch.allclose(out_contig, ref, rtol=1e-2, atol=1e-2)
    paged_ok = torch.allclose(out_paged, ref, rtol=1e-2, atol=1e-2)
    contig_diff = (out_contig.float() - ref.float()).abs().max().item()
    paged_diff = (out_paged.float() - ref.float()).abs().max().item()

    print(
        f"{case['name']}: "
        f"contig={'PASS' if contig_ok else 'FAIL'} "
        f"paged={'PASS' if paged_ok else 'FAIL'} "
        f"contig_ms={ms_contig:.3f} "
        f"paged_ms={ms_paged:.3f} "
        f"contig_max_abs={contig_diff:.6g} "
        f"paged_max_abs={paged_diff:.6g}"
    )

    return contig_ok and paged_ok


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the target attention kernels.")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    passed = True
    for case in CASES:
        passed = run_case(case, dtype, args.warmup, args.iters) and passed
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
