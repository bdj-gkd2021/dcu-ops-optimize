#pragma once
#include "flash_fwd_b16_fa.h"


namespace flash {

#include "fwd/int8_pv_gemm_prefetch_k.h"
#include "fwd/int8_qk_gemm_prefetch_v.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_int8_prefix_prefill_1rowblock(const Params &params, const int bidb, const int __bidh, const int m_block, const int WARP_ID) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kBlockK_int8  = Kernel_traits::kBlockK_int8;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimVSplit;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr int SplitD    = Kernel_traits::SplitD;
    constexpr int kHeadDimVOrigin = Kernel_traits::kHeadDimV;
    // 获取 splitD 结果
    const int bidh = __bidh / SplitD;

    // 获取当前 TG 处理的任务大小
    const flash::BlockInfo</*Varlen=*/!Is_even_MN, false/*Is_kvcache*/> binfo(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    // 分配 lds
    extern __shared__ Element smem[];
    Element_k* q_lds = (Element_k*)&(smem);
    Element_k* v_lds = (Element_k*)&(smem);
    Element_k* k_lds = reinterpret_cast<Element_k*>(v_lds + (2 * STAGES * 32 * kBlockK));

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride = (Layout == 1) ? params.q_row_stride: params.q_row_stride;
    int seqlen_k_stride = (Layout == 1) ? params.k_row_stride: params.k_row_stride;
    int seqlen_v_stride = (Layout == 1) ? params.v_row_stride: params.v_row_stride;
    int seqlen_o_stride = (Layout == 1) ? params.o_row_stride: params.o_row_stride;
    int scale_seqlen_q_stride = seqlen_q_stride / kHeadDim;
    int scale_seqlen_k_stride = seqlen_k_stride / kHeadDim;
    size_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    size_t row_offset_lse;
    // 获取页表信息
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    if constexpr (Layout == 1) { /*bshd layout, lse is num_heads, total_q*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * seqlen_o_stride;
    }
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %ld | row_offset_k: %ld | row_offset_v: %ld | row_offset_o: %ld | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif

    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto gQ = prepare_for_buffer_load<kHeadDim, Element_k>(reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element_k>(reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDim, Element_k>(reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v);

    float *scales_q_ptr = reinterpret_cast<float*>(params.scales_q_ptr);
    float *scales_k_ptr = reinterpret_cast<float*>(params.scales_k_ptr);
    float *scales_v_ptr = reinterpret_cast<float*>(params.scales_v_ptr);
    Element_k *k_ptr = reinterpret_cast<Element_k*>(params.k_ptr);

    // attention 变体: Alibi
    float gAlibi;
    if constexpr (Has_alibi) {
        gAlibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + threadIdx.x & 63; /* 参考官方写法 offset(offset + (bid * nheads + hid) * 32 + tid % 32) */
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff; /*hcu 不支持 16bit 和 8bit 的比较指令*/
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32)/*前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + WARP_ID/*当前 block 内的 warp id*/;
        // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might exit early and no one saves the rng states.
        if (m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int q_seq_idx = WARP_ID * 32 + (lane_id & 15) * 2;
    int64_t scales_k_global_offset = row_offset_k / kHeadDim;
    int scales_q_offset = row_offset_q / kHeadDim + q_seq_idx*scale_seqlen_q_stride;
    // __shared__ float scales_q_lds[128];
    // if (q_seq_idx < binfo.actual_seqlen_q){
    //     scales_q_lds[q_seq_idx] = scales_q_ptr[scales_q_offset];
    //     scales_q_lds[q_seq_idx+1] = scales_q_ptr[scales_q_offset+1];
    // }
    // float scales_q[2];
    // if (q_seq_idx + 1 < params.seqlen_q){
    //     scales_q[0] = scales_q_ptr[scales_q_offset];
    //     scales_q[1] = scales_q_ptr[scales_q_offset+1*scale_seqlen_q_stride];
    // }
    float scales_q[2];
    for (int m=0; m<2; m++) {
        if (scales_q_offset+m*scale_seqlen_q_stride < params.total_scale_q) {
            scales_q[m] = scales_q_ptr[scales_q_offset+m*scale_seqlen_q_stride];
        }
    }
    int table_diff = 0;
    int offset_diff = 0;
    // 预取 Q 的数据到寄存器
    vec4_int8 q_reg[(kHeadDim/kBlockK_int8)*((WARP_M*kBlockK_int8)/(32*kBlockK_int8))*2][4];
    Is_even_MN
    ? int8_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK_int8, WARP_M, Element_k, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride)
    : int8_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK_int8, WARP_M, Element_k, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride, (binfo.actual_seqlen_q - m_block * kBlockM));

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K     = false;

    /***************************************************************************************************************************/
    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/
    /***************************************************************************************************************************/
    constexpr bool Aggressive = (kHeadDim == 128 or kHeadDim == 64);
    auto QK_GEMM_FUNC = Aggressive
        ? &int8_qk_gemm_prefetch_v_headdim128<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK_int8, WARP_M, WARP_N, STAGES, Element, Element_k, ElementAccum, Is_even_MN>
        : &int8_qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK_int8, WARP_M, WARP_N, STAGES, Element, Element_k, ElementAccum, Is_even_MN>;

    auto PV_GEMM_FUNC = Aggressive
        ? &int8_pv_gemm_prefetch_k_headdim128<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, Element_k, ElementAccum, Is_even_MN>
        : &int8_pv_gemm_prefetch_k<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, Element_k, ElementAccum, Is_even_MN>;
    // mask 循环中不需要做 prefetch K, 因此 prefetch K 固定为 false
    auto PV_GEMM_FUNC_IN_MASK = Aggressive
        ? &int8_pv_gemm_prefetch_k_headdim128<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, Element_k, ElementAccum, Is_even_MN>
        : &int8_pv_gemm_prefetch_k<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, Element_k, ElementAccum, Is_even_MN>;

    // constexpr int n_masking_steps = (!Is_causal && !Is_local) ? 1 : flash::ceil_div(kBlockM, kBlockN);
    // These are the iterations where we don't need masking on S
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max/*n_block_max - n_masking_steps*/; ++n_block_loop) {

        const int seqlen_kv_limit = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // c mini tile is 32 * 32
        vec4_int32 s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4] = {0};

        if constexpr (STAGES > 1) {
            Is_even_MN
            ? int8_prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK_int8, WARP_NUM, WARP_N, Element_k, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : int8_prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK_int8, WARP_NUM, WARP_N, Element_k, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_kv_limit);
        }

        scales_k_global_offset += (int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride))/kHeadDim;
        float scales_k[kBlockN / WARP_N][2][4];
        float scales_v[kBlockN / WARP_N][2][4];
        #pragma unroll
        for (int i = 0; i < (kBlockN/32); ++i) {
            #pragma unroll
            for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int64_t scales_k_offset = scales_k_global_offset + (lane_id/16*2 + k*8 + min_tile_n + i*32)*scale_seqlen_k_stride;
                    scales_k[i][min_tile_n][k] = scales_k_ptr[scales_k_offset];
                    scales_v[i][min_tile_n][k] = scales_v_ptr[scales_k_offset];
                }
            }
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        vec4_Accum<ElementAccum> s_reg_fp32[(WARP_M/32)*(kBlockN / WARP_N)][4];
        // // 将 s_reg 的值转换为 fp32
        #pragma unroll
        for (int i = 0; i < (WARP_M/32)*(kBlockN/32); ++i) {
            #pragma unroll
            for (int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                #pragma unroll
                for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        s_reg_fp32[i][min_tile_n*2 + min_tile_m].f32[k] = static_cast<float>(s_reg[i][min_tile_n*2 + min_tile_m][k]) * scales_q[min_tile_m] * scales_k[i][min_tile_n][k];
                    }
                }
            }
        }

        if constexpr (Has_alibi) {
            apply_alibi<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg_fp32, n_block_loop * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, binfo.actual_seqlen_q, gAlibi);
        }

        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg_fp32, seqlen_kv_limit); }
        } else {
            if constexpr (Is_causal) {
                apply_mask_causal<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg_fp32, n_block_loop * kBlockN, binfo.actual_seqlen_k,
                                                                                m_block * kBlockM + WARP_ID * WARP_M,
                                                                                binfo.actual_seqlen_q);
            } else if constexpr (Is_local) {
                apply_mask_local<Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg_fp32, n_block_loop * kBlockN, binfo.actual_seqlen_k,
                                                                                m_block * kBlockM + WARP_ID * WARP_M,
                                                                                binfo.actual_seqlen_q, params.window_size_left,
                                                                                params.window_size_right);
            }
        }

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg_fp32, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        if constexpr (Is_dropout) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg_fp32, seqlen_kv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        // convertType: float2half
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg_fp32);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        offset_diff             = block_table_offset_next - block_table_offset_cur;
        Is_even_MN
        ? PV_GEMM_FUNC(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(int64_t*)&gK += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element_k);
        *(int64_t*)&gV += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element_k);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, false/*Is_dropout*/, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    if (params.softmax_lse_ptr != nullptr) {

        fwd_epilogue_store_lse<WARP_M, Is_even_MN, SplitD, false/*Is_Interleaved*/, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, WARP_ID, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, false/*Is_Interleaves*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, WARP_ID, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_int8_prefix_prefill_kernel(const Params params) {

    const int bidh = blockIdx.y;

    const int bidb = blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = blockIdx.x;
    compute_attn_int8_prefix_prefill_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
}

} // namespace flash