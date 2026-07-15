#pragma once
#include "flash_fwd_b16_pa.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////


#include "kvcache/int8_kvcache_qk_gemm_prefetch_v.h"
#include "kvcache/int8_kvcache_pv_gemm_prefetch_k.h"
#include "kvcache/int8_kvcache_softmax.h"
#include "kvcache/int8_kvcache_acco_reduce.h"

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, typename Params>
inline __device__ void compute_attn_mha_1rowblock_splitkv_int8(const Params &params, const int bidb, const int bidh, const int WARP_ID) {
    using Element          = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;

    #ifndef NDEBUGING
    // ElementAccum * qk_ptr = static_cast<ElementAccum*>(params.qk_ptr);
    int * qk_ptr = static_cast<int*>(params.qk_ptr);
    #endif
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK_int8  = Kernel_traits::kBlockK_int8;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;

    flash::BlockInfo</*Varlen=*/true, /*Is_Kvcache*/true> binfo(params, bidb);

    // recompute the true actual_seqlen_k and num_split according to split_id, especially the last block
    int split_id;
    if constexpr (Split) {
        split_id   = blockIdx.y;
        int num_splits = max(1, floor_div(binfo.actual_seqlen_k, params.partition_size));
        binfo.actual_seqlen_k = (split_id == num_splits - 1)
            ? binfo.actual_seqlen_k - split_id * params.partition_size: params.partition_size;
        binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
        if (split_id >= num_splits) return;
    }

    const int WARP_NUM = (kBlockN)/(WARP_N);

    // kvcache doesn't has mask, no need to balance workload
    const int m_block = blockIdx.x;

    // when groups is more than 32, this may lead to incorrect results
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;
    extern __shared__ Element smem[];

    // decide lds partition
    // load Q --> QK gemm load K --> PV gemm load V, no conflicts
    #if defined(KVCACHE_USE_4STAGES_PINGPANG) && (defined(__gfx936__) || defined(__gfx938__))
        Element_k* q_lds = reinterpret_cast<Element_k*>(smem);
        Element_k* k_lds = q_lds;
        Element_k* v_lds = q_lds;
        ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
        ElementAccum* max_lds   = reinterpret_cast<ElementAccum*>(v_lds + (4096 - 256));
        /*默认占用 32KB, 最后一个 4x128 不预取, 留给 max_lds 做多 wave 之间的 reduce, 这里选的是 0 号 wave lds 最后一段空间(一共 8KB, 4096 个 half), 512 B 足够了, 即使 256 个 Half*/
    #else
        Element_k* q_lds = reinterpret_cast<Element_k*>(smem) + 512/*1KB, 512 halfs, configured for max_lds*/;
        Element_k* k_lds = q_lds;
        // Element_k* v_lds_int8 = reinterpret_cast<Element_k*>(q_lds);
        Element_k* v_lds = q_lds;
        // prepare lds for max and acc whiling reducing results across 4 waves
        // max and acc_o_lds has no conflicts while using lds
        ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
        ElementAccum* max_lds   = acc_o_lds;
        // float* scales_q = reinterpret_cast<float*>(smem);
        // __shared__ ElementAccum acc_o_lds[kHeadDim * 4];
        // __shared__ ElementAccum max_lds[WARP_NUM * WARP_M];
    #endif

    vec4_int8 q_reg[(kHeadDim/kBlockK_int8)*((WARP_M*kBlockK_int8)/(32*kBlockK_int8))*2][4];  //ds_read mini size is 32*32,2 is seq, 4 is head dim

    const int n_block_min = 0;
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);

    // acquire stride over seqlen dimension
    const int query_seqlen_stride  = params.q_row_stride;
    const int kcache_seqlen_stride = params.k_row_stride;
    const int vcache_seqlen_stride = params.v_row_stride;
    const int scale_kcache_seqlen_stride = kcache_seqlen_stride / kHeadDim;

    // compute block table
    const int page_block_size = params.page_block_size;
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    // if split, block_table begin from the new split!
    block_table = block_table + (Split ? ceil_div(split_id * params.partition_size, page_block_size) : 0);
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // const index_t row_offset_v = block_table == nullptr
    //     ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
    //     + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
    //     : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    const int64_t row_offset_q = bidb * params.q_batch_stride + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    #if defined(__gfx936__) || defined(__gfx938__)
        constexpr bool USE_CACHE_SWIZZLE = false;
    #else
        constexpr bool USE_CACHE_SWIZZLE = true; // for gfx928, cache swizzle have significant influence
    #endif

    auto gQ = prepare_for_buffer_load<kHeadDim, Element_k, false>(reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element_k, false>(reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV, Element_k, false>(reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_k);

    float *scales_q_ptr = reinterpret_cast<ElementAccum*>(params.scales_q_ptr);
    float *scales_k_ptr = reinterpret_cast<ElementAccum*>(params.scales_k_ptr);
    float *scales_v_ptr = reinterpret_cast<ElementAccum*>(params.scales_v_ptr);
    
    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int q_seq_idx = (lane_id & 15) * 2;
    float scales_q[M_MMAC_COUNT];
    int scales_q_global_offset = row_offset_q / kHeadDim;
    for (int m_idx=0; m_idx<M_MMAC_COUNT; m_idx++){
        int scales_q_offset = min(scales_q_global_offset + q_seq_idx + m_idx, params.total_scale_q);
        scales_q[m_idx] = scales_q_ptr[scales_q_offset];
    }

    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse;
        scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse;
    }

    float gAlibi;
    if constexpr (Has_alibi) {
        gAlibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + threadIdx.x & 63; /* 参考官方写法 offset(offset + (bid * nheads + hid) * 32 + tid % 32) */
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff; /*DCU 不支持 16bit 和 8bit 的比较指令*/
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32)/*前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + WARP_ID/*当前 block 内的 warp id*/;
        // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might exit early and no one saves the rng states.
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }


    int8_kvcache_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_NUM, Element, Element_k, STAGES, REUSE_KV_TIMES, M_MMAC_COUNT>(gQ, q_lds, q_reg, WARP_ID, (binfo.actual_seqlen_q - m_block * kBlockM));
    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M/32] = {-INFINITY};
    vec2_Accum<ElementAccum> scores_sum[WARP_M/32] = {0};
    // 由于当前编译器无法自动生成 v_mov_b64 指令, 主动用 builtin 还会被转译成 v_mov_b32, 因此用内联汇编控制
    #if defined(__gfx936__) || defined(__gfx938__)
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4];
        if constexpr (kHeadDimV == 128) { // kHeadDim 128 是主要优化目标
            if constexpr (M_MMAC_COUNT == 1) {
                inline_vgpr4_init_zero_4x2x4(acc_o);
            } else {
                inline_vgpr4_init_zero_4x4x4(acc_o);
            }
            __builtin_amdgcn_sched_barrier(0);
        } else { // 非 kHeaddim 128, 交给编译器后续的优化了
            #pragma unroll
            for (int i = 0; i < (kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32)); ++i) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #if defined(__gfx936__)
                        acc_o[i][min_tile_n * 2 + min_tile_m].u64[0] = __builtin_hcu_mov_b64(0);
                        acc_o[i][min_tile_n * 2 + min_tile_m].u64[1] = __builtin_hcu_mov_b64(0);
                        #elif defined(__gfx938__)
                        asm volatile("v_mov_b64 %0, 0x0"
                            : "=v"(acc_o[i][min_tile_n * 2 + min_tile_m].u64[0])
                            :);
                        asm volatile("v_mov_b64 %0, 0x0"
                            : "=v"(acc_o[i][min_tile_n * 2 + min_tile_m].u64[1])
                            :);
                        #endif
                    }
                }
            }
        }
    #else // gfx928
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4] = {0};
    #endif
    
    constexpr bool PREFETCH_K = false; // KV lds 使用量较大, 暂不适合使用 prefetch K
    int block_table_idx_cur = block_table_idx;
    // int scales_k_global_offset = block_table[block_table_idx_cur] * params.scales_k_batch_stride + (bidh / params.h_h_k_ratio) * params.scales_k_head_stride;
    int64_t scales_k_global_offset = row_offset_k / kHeadDim;
    int table_diff = 0;
    int offset_diff = 0;
    // These are the iterations where we don't need masking on S
    // Separated processing of masked and unmasked would result in vgpr spill
    // Can be reviewed in the future
    int n_block_loop = n_block_min;
    float scales_k[2][4];
    float scales_v[2][4];
    int pre_scales_k_offset[2][4];
    #pragma unroll
    for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            pre_scales_k_offset[min_tile_n][k] = (lane_id/16*2 + k*8 + min_tile_n + WARP_N*WARP_ID)*scale_kcache_seqlen_stride;
        }
    }
    
    for (; n_block_loop < n_block_max - 1;) {
        int warp_id_vec = threadIdx.x / 64; // warp id in a block
        int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);
        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;
        scales_k_global_offset += (int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride))/kHeadDim;

        if constexpr ((not PREFETCH_K) and (STAGES > 1)) {
            int8_kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_N, Element, Element_k, STAGES, WARP_NUM>(gK, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        }

        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_k[min_tile_n][k] = scales_k_ptr[scales_k_offset];
                // scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
            }
        }

        union_vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4];
        int8_kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK_int8, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gK, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, kcache_seqlen_stride, warp_seqkv_limit);
        vec4_Accum<ElementAccum> s_reg_fp32[(WARP_M/32)*(WARP_N/32)][4];

        // 将 s_reg 的值转换为 fp32
        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
                #pragma unroll
                for (int i = 0; i < (WARP_M/32)*(WARP_N/32); ++i) {
                    #pragma unroll
                    for (int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        s_reg_fp32[i][min_tile_n*2 + min_tile_m].f32[k] = static_cast<float>(s_reg[i][min_tile_n*2 + min_tile_m].int32[k]) * scales_q[min_tile_m] * scales_k[min_tile_n][k];
                    }
                }
            }
        }

        if constexpr (Has_alibi) {
            int8_kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k, (m_block * kBlockM), binfo.actual_seqlen_q, gAlibi);
        }

        // kvcache_apply_mask<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_seqkv_limit);

        if constexpr (Is_local) {
            int8_kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k,
                                                                                (m_block * kBlockM + warp_id * WARP_M),
                                                                                    binfo.actual_seqlen_q, params.window_size_left,
                                                                                    params.window_size_right);
        }

        int8_kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg_fp32, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            int8_kvcache_apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, M_MMAC_COUNT>(s_reg_fp32, binfo.actual_seqlen_k - warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M/32)*(WARP_N/32)][4];
        // convertType: float2half
        int8_kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg_fp32);

        // 如果要预取 K 的话, 需要提前偏移
        if constexpr (PREFETCH_K) {
            n_block_loop++;
        }

        block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        offset_diff             = block_table_offset_next - block_table_offset_cur;

        *(uint64_t*)&gK += ((int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride)) * sizeof(Element_k));

        int8_kvcache_pv_gemm_prefetch_k_3stage<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        *(uint64_t*)&gV += ((int64_t(table_diff) * int64_t(params.v_batch_stride) + int64_t(offset_diff) * int64_t(params.v_row_stride)) * sizeof(Element_k));

        if constexpr (not PREFETCH_K) {
            n_block_loop++;
        }

    }

    {
        int warp_offset_in_seqkv = n_block_loop * kBlockN + WARP_ID * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;
        scales_k_global_offset += (int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride))/kHeadDim;

        if constexpr ((not PREFETCH_K) and (STAGES > 1)) {
            int8_kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_N, Element, Element_k, STAGES, WARP_NUM>(gK, k_lds, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);
        }

        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_k[min_tile_n][k] = scales_k_ptr[scales_k_offset];
            }
        }
        
        union_vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4];
        int8_kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK_int8, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gK, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, kcache_seqlen_stride, warp_seqkv_limit);
        vec4_Accum<ElementAccum> s_reg_fp32[(WARP_M/32)*(WARP_N/32)][4];
    
        // 将 s_reg 的值转换为 fp32
        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
                #pragma unroll
                for (int i = 0; i < (WARP_M/32)*(WARP_N/32); ++i) {
                    #pragma unroll
                    for (int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        s_reg_fp32[i][min_tile_n*2 + min_tile_m].f32[k] = static_cast<float>(s_reg[i][min_tile_n*2 + min_tile_m].int32[k]) * scales_q[min_tile_m] * scales_k[min_tile_n][k];
                    }
                }
            }
        }
        // __syncthreads();
        // print_qk(m_block, bidb, bidh);

        if constexpr (Has_alibi) {
            int8_kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k, (m_block * kBlockM), binfo.actual_seqlen_q, gAlibi);
        }

        if constexpr (true) {
            int8_kvcache_apply_mask<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_seqkv_limit);
        }

        if constexpr (Is_local) {
            int8_kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k,
                                                                                (m_block * kBlockM + WARP_ID * WARP_M),
                                                                                    binfo.actual_seqlen_q, params.window_size_left,
                                                                                    params.window_size_right);
        }

        int8_kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg_fp32, scores_max, scores_sum, acc_o, max_lds, WARP_ID, params.scale_softmax_log2);

        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            int8_kvcache_apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, M_MMAC_COUNT>(s_reg_fp32, binfo.actual_seqlen_k - warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M/32)*(WARP_N/32)][4];
        // convertType: float2half
        int8_kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg_fp32);

        int8_kvcache_pv_gemm_prefetch_k_3stage<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);

    }
    __syncthreads();

    if constexpr (WARP_NUM > 1) {
        // reduce acc_o across 4 waves
        kvcache_acco_reduce<REUSE_KV_TIMES, kHeadDimV, kBlockK, WARP_M, M_MMAC_COUNT, WARP_NUM, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, WARP_ID, lane_id);
    }

    /**********************************************************************************************************************************/

    kvcache_epilugue_rescale_acco<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    kvcache_epilogue_store_max_sum<Split, false, WARP_M / 32, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, WARP_ID, threadIdx.x, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);

    /**************************************************************************************************************************************/
    kvcache_epilogue_store_output<Params, kHeadDimV, kHeadDimV, Split, false/*Is_16x32*/, typename Kernel_traits::SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, 0, WARP_ID, lane_id);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool Is_softcap, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv_int8(const Params &params) {
    // block id in sequence dimension
    const int m_block = blockIdx.x;

    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);
    flash::compute_attn_mha_1rowblock_splitkv_int8<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_K, Return_softmax, Has_alibi, Split, M_MMAC_COUNT, REUSE_KV_TIMES, Flash_fwd_params>(params, bidb, bidh, warp_id);
}

} // namespace flash
