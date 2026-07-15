import math

import pytest
import torch

from flash_attn import flash_attn_with_kvcache, vllm_flash_attn_with_kvcache


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


def make_paged_cache_from_full(full_k, full_v, page_size=64):
    batch, kv_len, kv_head, head_dim = full_k.shape
    nblocks = math.ceil(kv_len / page_size)
    num_blocks = nblocks * batch * 3
    device = full_k.device
    k_paged = torch.zeros(num_blocks, page_size, kv_head, head_dim, device=device, dtype=full_k.dtype)
    v_paged = torch.zeros(num_blocks, page_size, kv_head, head_dim, device=device, dtype=full_v.dtype)
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device)[: batch * nblocks]
    block_table = block_table.view(batch, nblocks)

    for b in range(batch):
        for blk in range(nblocks):
            src_start = blk * page_size
            src_end = min(src_start + page_size, kv_len)
            dst = int(block_table[b, blk].item())
            span = src_end - src_start
            k_paged[dst, :span] = full_k[b, src_start:src_end]
            v_paged[dst, :span] = full_v[b, src_start:src_end]

    return k_paged, v_paged, block_table


CONTIG_CASES = [
    {
        "name": "prefill_small_mha",
        "batch": 2,
        "q_len": 128,
        "kv_len": 128,
        "q_head": 6,
        "kv_head": 6,
        "head_dim": 64,
        "causal": True,
    },
    {
        "name": "prefill_mid_mqa",
        "batch": 2,
        "q_len": 64,
        "kv_len": 800,
        "q_head": 6,
        "kv_head": 1,
        "head_dim": 64,
        "causal": True,
    },
    {
        "name": "prefill_long_gqa",
        "batch": 2,
        "q_len": 16,
        "kv_len": 20000,
        "q_head": 6,
        "kv_head": 3,
        "head_dim": 64,
        "causal": True,
    },
    {
        "name": "decode_short_mha",
        "batch": 1,
        "q_len": 1,
        "kv_len": 1025,
        "q_head": 16,
        "kv_head": 16,
        "head_dim": 64,
        "causal": True,
    },
    {
        "name": "decode_mid_gqa",
        "batch": 1,
        "q_len": 1,
        "kv_len": 1514,
        "q_head": 10,
        "kv_head": 2,
        "head_dim": 64,
        "causal": True,
    },
    {
        "name": "decode_long_mqa",
        "batch": 1,
        "q_len": 1,
        "kv_len": 51200,
        "q_head": 10,
        "kv_head": 2,
        "head_dim": 64,
        "causal": True,
    },
]


PAGED_CASES = [
    {"name": "decode_1025_mha", "batch": 1, "q_len": 1, "kv_len": 1025, "q_head": 16, "kv_head": 16, "head_dim": 64},
    {"name": "decode_16384_gqa", "batch": 20, "q_len": 1, "kv_len": 16384, "q_head": 28, "kv_head": 4, "head_dim": 64},
    {"name": "decode_17408_gqa", "batch": 8, "q_len": 1, "kv_len": 17408, "q_head": 16, "kv_head": 2, "head_dim": 64},
    {"name": "decode_3584_gqa1", "batch": 32, "q_len": 1, "kv_len": 3584, "q_head": 32, "kv_head": 8, "head_dim": 64},
    {"name": "decode_1514_gqa", "batch": 48, "q_len": 1, "kv_len": 1514, "q_head": 10, "kv_head": 2, "head_dim": 64},
    {"name": "decode_32402_gqa", "batch": 2, "q_len": 1, "kv_len": 32402, "q_head": 10, "kv_head": 2, "head_dim": 64},
    {"name": "decode_51200_gqa", "batch": 1, "q_len": 1, "kv_len": 51200, "q_head": 10, "kv_head": 2, "head_dim": 64},
    {"name": "decode_256_gqa", "batch": 1, "q_len": 1, "kv_len": 256, "q_head": 8, "kv_head": 4, "head_dim": 64},
]


def _run_contig_case(case, dtype):
    device = "cuda"
    torch.manual_seed(0)

    batch = case["batch"]
    q_len = case["q_len"]
    kv_len = case["kv_len"]
    q_head = case["q_head"]
    kv_head = case["kv_head"]
    head_dim = case["head_dim"]

    q = torch.randn(batch, q_len, q_head, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=case["causal"],
        num_splits=0,
        return_softmax_lse=False,
    )
    ref = reference_attention(q, k_cache, v_cache, causal=case["causal"])

    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), (
        f"{case['name']} contig mismatch: "
        f"max_abs={(out.float() - ref.float()).abs().max().item():.6g}"
    )


def _run_paged_case(case, dtype):
    device = "cuda"
    torch.manual_seed(0)

    batch = case["batch"]
    q_len = case["q_len"]
    kv_len = case["kv_len"]
    q_head = case["q_head"]
    kv_head = case["kv_head"]
    head_dim = case["head_dim"]

    q = torch.randn(batch, q_len, q_head, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch, kv_len, kv_head, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    k_paged, v_paged, block_table = make_paged_cache_from_full(k_cache, v_cache, page_size=64)

    out = vllm_flash_attn_with_kvcache(
        q,
        k_paged,
        v_paged,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
        max_seqlen_k=kv_len,
    )
    ref = reference_attention(q, k_cache, v_cache, causal=True)

    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), (
        f"{case['name']} paged mismatch: "
        f"max_abs={(out.float() - ref.float()).abs().max().item():.6g}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CONTIG_CASES, ids=[c["name"] for c in CONTIG_CASES])
def test_flash_attn_with_kvcache_matches_ref(case, dtype):
    _run_contig_case(case, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", PAGED_CASES, ids=[c["name"] for c in PAGED_CASES])
def test_vllm_flash_attn_with_kvcache_matches_ref(case, dtype):
    _run_paged_case(case, dtype)
