import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

import math
import time
import pytest
import torch
import torch.nn.functional as F


import pdb


# torch.set_printoptions(precision=4, profile="default", sci_mode=False)

from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
    varlen_fwd_unified,
)
from flash_attn import flash_attn_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size_n
from flash_attn.layers.rotary import apply_rotary_emb

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


logger = init_logger(__name__)
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )


    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )


def ref_attn(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None,
             softcap=0.0, alibi_slopes=None, use_alibi_sqrt=False, qq_bias=None,
             sinks=None, mm_prefix_range=None):
    """
    Pure PyTorch reference attention for a single sequence.
    q: [seqlen_q, nheads, d]
    k: [seqlen_k, nheads_k, d]
    v: [seqlen_k, nheads_k, d_v]
    mm_prefix_range: [MAX_MM_RANGES, 2] int32, per-sequence ranges
    sinks: [nheads] dtype, sink values per head
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[0]
    nheads_k = k.shape[1]
    context_len = seqlen_k - seqlen_q
    num_queries_per_kv = nheads // nheads_k

    # expand kv heads for GQA
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)

    q_ = q.unsqueeze(0).transpose(1, 2).float()   # [1, h, sq, d]
    k_ = k.unsqueeze(0).transpose(1, 2).float()   # [1, h, sk, d]
    v_ = v.unsqueeze(0).transpose(1, 2).float()   # [1, h, sk, dv]

    scores = torch.matmul(q_, k_.transpose(-2, -1)) * softmax_scale  # [1, h, sq, sk]

    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)

    # ALiBi
    if alibi_slopes is not None:
        slopes = alibi_slopes.float().view(1, nheads, 1, 1)  # [1, h, 1, 1]
        k_idx = torch.arange(seqlen_k, device=q.device).float().view(1, 1, 1, seqlen_k)
        q_idx = torch.arange(seqlen_q, device=q.device).float().view(1, 1, seqlen_q, 1) + context_len
        if use_alibi_sqrt:
            # triton: alibi_offset = -sqrt(max(0, q_abs - k))
            rel = (q_idx - k_idx).clamp(min=0)
            bias = -torch.sqrt(rel)
        else:
            # triton: alibi_offset = seq_offset - context_len = k - context_len
            bias = k_idx - context_len
        scores = scores + slopes * bias
    # print("scores before qq_bias [seq1, h0, token524, :5]:", scores[0, 0, 524, :5])
    # QQ bias (query-query region: col >= context_len)
    if qq_bias is not None:
        scores[:, :, :, context_len:] = (
            scores[:, :, :, context_len:]
            + qq_bias.float().unsqueeze(0).unsqueeze(0)
        )

    # Save scores before masking for mm_prefix_range restore
    scores_before_mask = scores.clone()

    # Causal mask
    if causal:
        q_idx = torch.arange(seqlen_q, device=q.device).view(seqlen_q, 1)
        k_idx = torch.arange(seqlen_k, device=q.device).view(1, seqlen_k)
        mask = k_idx <= (q_idx + context_len)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Sliding window
    window_left, window_right = window_size
    if window_left >= 0 or window_right >= 0:
        q_idx = torch.arange(seqlen_q, device=q.device).view(seqlen_q, 1).float() + context_len
        k_idx = torch.arange(seqlen_k, device=q.device).view(1, seqlen_k).float()
        mask = torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool)
        if window_left >= 0:
            mask &= (k_idx >= q_idx - window_left)
        if window_right >= 0:
            mask &= (k_idx <= q_idx + window_right)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # MM prefix range: override causal/sliding_window mask for bidirectional region
    if mm_prefix_range is not None:
        # q_idx: absolute position of each query token
        q_idx = torch.arange(seqlen_q, device=q.device).view(seqlen_q, 1).long() + context_len
        k_idx = torch.arange(seqlen_k, device=q.device).view(1, seqlen_k).long()
        override_mask = torch.zeros(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool)
        for r in range(mm_prefix_range.shape[0]):
            start = mm_prefix_range[r, 0].item()
            end   = mm_prefix_range[r, 1].item()
            if start >= end:
                continue
            q_in_range = (q_idx >= start) & (q_idx <= end)  # [sq, 1]
            k_in_range = (k_idx >= start) & (k_idx <= end)  # [1, sk]
            override_mask |= (q_in_range & k_in_range)      # [sq, sk]
        scores = torch.where(
            override_mask.unsqueeze(0).unsqueeze(0),
            scores_before_mask,
            scores
        )

    # Sinks
    if sinks is not None:
        outs = []
        for h in range(nheads):
            # triton: M = tl.load(sink_ptr + ...), no scale applied
            # scale is applied to S via: S += scale * tl.dot(Q, K)
            # so sink is compared directly against scaled scores
            sink_val = sinks[h].float().item()  # no softmax_scale here
            s = scores[0, h]                    # [sq, sk], already scaled
            row_max = torch.maximum(
                s.amax(dim=-1),
                torch.full_like(s[:, 0], sink_val)
            )                                                      # [sq]
            exp_s = torch.exp(s - row_max.unsqueeze(-1))           # [sq, sk]
            exp_s = torch.nan_to_num(exp_s, nan=0.0, posinf=0.0)
            exp_sink = torch.exp(sink_val - row_max)               # [sq]
            denom = exp_s.sum(dim=-1) + exp_sink                   # [sq]
            attn_h = exp_s / denom.unsqueeze(-1)                   # [sq, sk]
            out_h = torch.matmul(attn_h, v_[0, h])                # [sq, dv]
            outs.append(out_h)
        out = torch.stack(outs, dim=1)                             # [sq, h, dv]
        return out.to(q.dtype)

    attn = torch.nn.functional.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v_)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


# ---------------------------------------------------------------------------
# Paged KV cache builder
# ---------------------------------------------------------------------------

def make_paged_kv(k_list, v_list, block_size, device, dtype):
    """
    Pack per-sequence K/V tensors into paged KV cache format.
    Returns k_cache [total_blocks, block_size, nheads_k, d],
            v_cache [total_blocks, block_size, nheads_k, dv],
            block_table [num_seqs, max_blocks_per_seq]
    """
    num_kv_heads = k_list[0].shape[1]
    head_size_k = k_list[0].shape[2]
    head_size_v = v_list[0].shape[2]

    blocks = []
    block_table_rows = []
    for k_seq, v_seq in zip(k_list, v_list):
        seqlen = k_seq.shape[0]
        num_blocks_seq = math.ceil(seqlen / block_size)
        row = []
        for b in range(num_blocks_seq):
            start = b * block_size
            end = min(start + block_size, seqlen)
            k_block = torch.zeros(block_size, num_kv_heads, head_size_k, device=device, dtype=dtype)
            v_block = torch.zeros(block_size, num_kv_heads, head_size_v, device=device, dtype=dtype)
            k_block[:end - start] = k_seq[start:end]
            v_block[:end - start] = v_seq[start:end]
            blocks.append((k_block, v_block))
            row.append(len(blocks) - 1)
        block_table_rows.append(row)

    max_blocks = max(len(r) for r in block_table_rows)
    num_seqs = len(block_table_rows)

    k_cache = torch.stack([b[0] for b in blocks])
    v_cache = torch.stack([b[1] for b in blocks])

    block_table = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
    for i, row in enumerate(block_table_rows):
        block_table[i, :len(row)] = torch.tensor(row, dtype=torch.int32)

    return k_cache, v_cache, block_table


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False, is_e5m2: bool = False) -> None:
    torch_dtype = x.dtype
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if is_e5m2:
        assert cos_diff < 1e-2
    elif use_fp8:
        assert cos_diff < 1e-3
    else:
        assert cos_diff < (1e-4 if torch_dtype == torch.bfloat16 else 1e-5)

# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["gqa"])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("window_size", [(-1, -1)])
@pytest.mark.parametrize("use_alibi_sqrt", [True, False])
@pytest.mark.parametrize("use_qq_bias", [True, False])  # seqlen_q > seqlen_k 时 skip
@pytest.mark.parametrize("use_sinks", [True, False])
@pytest.mark.parametrize("use_mm_prefix", [True, False])
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,block_size",
    [
        # --- 场景 1: 标准 Prefill (全量预填充) ---
        # 验证对角线处理、Is_causal 逻辑以及全量 Bias 覆盖
        (1, 512, 512, 128),     # 单 Batch 小尺寸，快速验证
        (4, 2048, 2048, 128),  # 匹配你日志的大尺寸，验证多 Batch 偏移
        (2, 1024, 1024, 128),   # 较小的 block_size，增加循环迭代次数

        # --- 场景 2: Decode 场景 (增量推理) ---
        # 验证 seqlen_q=1 时，如何正确从 KV Cache 的最后位置读取信息
        # 此时 qq_bias 实际上只退化为向量加法，最容易测出指针偏移错误
        (8, 1, 2048, 128),      # 高 Batch 的标准 Decode
        (1, 1, 4096, 128),     # 超长上下文 Decode，验证大索引寻址

        # --- 场景 3: Chunked Prefill / Speculative Decoding (分段/投机采样) ---
        # Q 小于 K，但大于 1。这是最难写的逻辑，验证 Is_causal 的动态截断
        (2, 128, 1024, 128),    # Q 是一小段，K 是长历史
        (4, 256, 512, 128),     # 验证 Q 和 K 比例较近时的处理

        # --- 场景 4: 边界非对称尺寸 (非 2 的幂次) ---
        # 专门用来抓那些“假设数据一定是 BlockSize 整数倍”的 Bug
        (1, 127, 127, 128),     # 刚好差 1 个填满 Block
        (2, 33, 1025, 128),     # 非常细碎的 Block 和不规则长度
    ],
)
def test_unified_attn_2d(
    batch_size, seqlen_q, seqlen_k, block_size,
    d, causal, window_size, softcap,
    mha_type, dtype,
    use_alibi_sqrt, use_qq_bias, use_sinks, use_mm_prefix,
):
    device = torch.device("cuda")
    torch.manual_seed(42)

    nheads = 8
    nheads_k = 1 if mha_type == "gqa" else nheads
    softmax_scale = d ** (-0.5)
    MAX_MM_RANGES = 2

    # skip invalid combos
    if use_alibi_sqrt and not causal:
        pytest.skip("alibi_sqrt only tested with causal=True")
    if use_qq_bias and seqlen_q > seqlen_k:
        pytest.skip("qq_bias requires seqlen_q <= seqlen_k")

    if use_mm_prefix and seqlen_q > seqlen_k:
        pytest.skip("mm_prefix not supported when seqlen_q > seqlen_k")

    # if use_mm_prefix and not causal:
    #     pytest.skip("mm_prefix_range is only meaningful with causal=True")

    q_list, k_list, v_list = [], [], []
    for _ in range(batch_size):
        q_list.append(torch.randn(seqlen_q, nheads, d, device=device, dtype=dtype))
        k_list.append(torch.randn(seqlen_k, nheads_k, d, device=device, dtype=dtype))
        v_list.append(torch.randn(seqlen_k, nheads_k, d, device=device, dtype=dtype))

    q_varlen = torch.cat(q_list, dim=0)
    cu_seqlens_q = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.cumsum(
        torch.tensor([seqlen_q] * batch_size, dtype=torch.int32), dim=0
    )
    seqused_k = torch.tensor([seqlen_k] * batch_size, device=device, dtype=torch.int32)
    k_cache, v_cache, block_table = make_paged_kv(k_list, v_list, block_size, device, dtype)

    # Build optional tensors
    alibi_slopes = None
    if use_alibi_sqrt:
        alibi_slopes = torch.rand(nheads, device=device, dtype=torch.float32) * 0.1

    qq_bias = None
    if use_qq_bias:
        qq_bias = torch.randn(seqlen_q, seqlen_q, device=device, dtype=torch.float32)

    sinks = None
    if use_sinks:
        sinks = torch.randn(nheads, device=device, dtype=dtype)

    mm_prefix_range = None
    if use_mm_prefix:
        mm_prefix_range = torch.zeros(batch_size, MAX_MM_RANGES, 2, device=device, dtype=torch.int32)
        for i in range(batch_size):
            prefix_len = min(32, seqlen_k // 4)
            mm_prefix_range[i, 0, 0] = 0
            mm_prefix_range[i, 0, 1] = prefix_len - 1

    # ---- Reference (torch) ----
    ref_outs = []
    for i in range(batch_size):
        ref_outs.append(
            ref_attn(
                q_list[i], k_list[i], v_list[i],
                causal=causal,
                window_size=window_size,
                softmax_scale=softmax_scale,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                use_alibi_sqrt=use_alibi_sqrt,
                qq_bias=qq_bias,
                sinks=sinks,
                mm_prefix_range=mm_prefix_range[i] if mm_prefix_range is not None else None,
            )
        )
    ref_out = torch.cat(ref_outs, dim=0)
    
    # ---- CUDA kernel ----
    cuda_out, cuda_lse = varlen_fwd_unified(
        q_varlen, k_cache, v_cache,
        cu_seqlens_q, seqused_k, block_table,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        use_alibi_sqrt=use_alibi_sqrt,
        qq_bias=qq_bias,
        s_aux=sinks,
        mm_prefix_range=mm_prefix_range,
        return_softmax_lse=True,
    )

    # # ---- Triton kernel ----
    # triton_out = torch.zeros_like(q_varlen)
    # unified_attention(
    #     q=q_varlen,
    #     k=k_cache,
    #     v=v_cache,
    #     out=triton_out,
    #     cu_seqlens_q=cu_seqlens_q,
    #     max_seqlen_q=seqlen_q,
    #     seqused_k=seqused_k,
    #     max_seqlen_k=seqlen_k,
    #     softmax_scale=softmax_scale,
    #     causal=causal,
    #     window_size=window_size,
    #     block_table=block_table,
    #     softcap=softcap,
    #     q_descale=None,
    #     k_descale=None,
    #     v_descale=None,
    #     alibi_slopes=alibi_slopes,
    #     use_alibi_sqrt=use_alibi_sqrt,
    #     qq_bias=qq_bias,
    #     sinks=sinks,
    #     mm_prefix_range=mm_prefix_range,
    # )

    cuda_max_diff = (cuda_out - ref_out).abs().max().item()
    # triton_max_diff = (triton_out - ref_out).abs().max().item()

    print(
        f"\n[{dtype} | causal={causal} | {mha_type} | bs={batch_size} "
        f"sq={seqlen_q} sk={seqlen_k} blk={block_size} | "
        f"alibi_sqrt={use_alibi_sqrt} qq_bias={use_qq_bias} "
        f"sinks={use_sinks} mm_prefix={use_mm_prefix}]"
        f"\n  CUDA   max_diff={cuda_max_diff:.4e}"
        # f"\n  Triton max_diff={triton_max_diff:.4e}"
    )

    cal_diff(cuda_out, ref_out, "out")


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

import time


def benchmark_unified_attention():

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float16
    d = 256
    block_size = 320

    warmup = 10
    repeat = 50

    window_size = (-1, -1)
    softcap = 0.0

    MAX_MM_RANGES = 2

    # GQA
    nheads = 8
    nheads_k = 1

    # workload shapes
    shapes = [
        # (8, 2048, 2048),
        # (4, 2048, 2048),
        (4, 1, 2048),
        # (4, 2048, 4096),
    ]

    # feature configs (C A Q S P)
    feature_configs = [
        (1,0,0,0,0),
        (1,1,0,0,0),
        (1,0,1,0,0),
        (1,0,0,1,0),
        (1,0,0,0,1),
        (1,1,1,0,0),
        (1,1,0,1,0),
        (1,1,0,0,1),
        (1,1,1,1,0),
        (1,1,1,0,1),
        (1,1,1,1,1),
    ]

    print("\nUnified Attention GQA Benchmark")
    print("=" * 120)

    print(
        f"{'BS':>3} {'SQ':>6} {'SK':>6} | "
        f"{'C':>1} {'A':>1} {'Q':>1} {'S':>1} {'P':>1} | "
        f"{'CUDA(ms)':>10} {'Triton(ms)':>11} | "
        f"{'CUDA TFLOPS':>11} {'Triton TFLOPS':>13} | {'Speedup':>8}"
    )

    print("-" * 120)

    def time_fn(fn):

        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeat):
            fn()

        torch.cuda.synchronize()

        return (time.perf_counter() - start) / repeat * 1000


    for batch_size, seqlen_q, seqlen_k in shapes:

        softmax_scale = d ** (-0.5)

        # -------------------
        # 创建 tensor (只做一次)
        # -------------------

        q_list, k_list, v_list = [], [], []

        for _ in range(batch_size):

            q_list.append(
                torch.randn(seqlen_q, nheads, d, device=device, dtype=dtype)
            )

            k_list.append(
                torch.randn(seqlen_k, nheads_k, d, device=device, dtype=dtype)
            )

            v_list.append(
                torch.randn(seqlen_k, nheads_k, d, device=device, dtype=dtype)
            )

        q_varlen = torch.cat(q_list, dim=0)

        cu_seqlens_q = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(
            torch.tensor([seqlen_q] * batch_size, dtype=torch.int32), dim=0
        )

        seqused_k = torch.tensor(
            [seqlen_k] * batch_size, device=device, dtype=torch.int32
        )

        k_cache, v_cache, block_table = make_paged_kv(
            k_list, v_list, block_size, device, dtype
        )

        triton_out = torch.zeros_like(q_varlen)

        for C, A, Q, S, P in feature_configs:

            causal = bool(C)
            use_alibi_sqrt = bool(A)
            use_qq_bias = bool(Q)
            use_sinks = bool(S)
            use_mm_prefix = bool(P)

            torch.cuda.empty_cache()

            # optional tensors

            alibi_slopes = None
            if use_alibi_sqrt:  # 先生成 alibi_slopes 再决定是否使用，确保相同随机数
                alibi_slopes = torch.rand(
                    nheads, device=device, dtype=torch.float32
                ) * 0.1

            qq_bias = None
            if use_qq_bias:
                qq_bias = torch.randn(
                    seqlen_q, seqlen_q, device=device, dtype=torch.float32
                )

            sinks = None
            if use_sinks:
                sinks = torch.randn(nheads, device=device, dtype=dtype)

            mm_prefix_range = None
            if use_mm_prefix:

                mm_prefix_range = torch.zeros(
                    batch_size, MAX_MM_RANGES, 2, device=device, dtype=torch.int32
                )

                for i in range(batch_size):

                    prefix_len = min(32, seqlen_k // 4)

                    mm_prefix_range[i, 0, 0] = 0
                    mm_prefix_range[i, 0, 1] = prefix_len - 1

            # ---------------- CUDA ----------------

            def run_cuda():

                varlen_fwd_unified(
                    q_varlen,
                    k_cache,
                    v_cache,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q=seqlen_q,
                    max_seqlen_k=seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    use_alibi_sqrt=use_alibi_sqrt,
                    qq_bias=qq_bias,
                    s_aux=sinks,
                    mm_prefix_range=mm_prefix_range,
                    return_softmax_lse=False,
                )

            # ---------------- Triton ----------------

            def run_triton():

                unified_attention(
                    q=q_varlen,
                    k=k_cache,
                    v=v_cache,
                    out=triton_out,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_table,
                    softcap=softcap,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    alibi_slopes=alibi_slopes,
                    use_alibi_sqrt=use_alibi_sqrt,
                    qq_bias=qq_bias,
                    sinks=sinks,
                    mm_prefix_range=mm_prefix_range,
                    seq_threshold_3D=128,
                )

            # compile warmup
            run_cuda()
            run_triton()
            torch.cuda.synchronize()

            cuda_ms = time_fn(run_cuda)
            triton_ms = time_fn(run_triton)

            # FLOPs
            flops = 4.0 * batch_size * nheads * seqlen_q * seqlen_k * d

            cuda_tflops = flops / cuda_ms / 1e9
            triton_tflops = flops / triton_ms / 1e9

            print(
                f"{batch_size:3d} {seqlen_q:6d} {seqlen_k:6d} | "
                f"{C} {A} {Q} {S} {P} | "
                f"{cuda_ms:10.3f} {triton_ms:11.3f} | "
                f"{cuda_tflops:11.2f} {triton_tflops:13.2f} | "
                f"{triton_ms/cuda_ms:8.2f}x"
            )

    print("=" * 120)


if __name__ == "__main__":
    benchmark_unified_attention()