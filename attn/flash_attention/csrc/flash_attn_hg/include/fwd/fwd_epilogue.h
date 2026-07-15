#include "numeric_types.h"
#include "intrinsic.h"


template<int WARP_M, int kBlockK, int kHeadDimV, bool Is_dropout, typename ElementAccum>
__forceinline__ __device__ void fwd_epilugue_rescale_acco(
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
    vec2_Accum<ElementAccum> lse[WARP_M / 32],
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32],
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32],
    const ElementAccum scale_softmax,
    const ElementAccum rp_dropout) {
    // Epilogue
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            ElementAccum sum = scores_sum[mi].f32[min_tile_m];
            ElementAccum inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse[mi].f32[min_tile_m] = (sum == 0.f || sum != sum) ? INFINITY : scores_max[mi].f32[min_tile_m] * scale_softmax + __logf(sum);
            ElementAccum scale = Is_dropout ? inv_sum * rp_dropout: inv_sum;
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < (kBlockK / 32); ++ni) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int mmac_id = min_tile_n * 2 + min_tile_m;
                    #pragma unroll
                    for(int pv_n_loop = 0; pv_n_loop < (kHeadDimV / kBlockK); ++pv_n_loop) {
                        const int pv_tile_id = pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + ni * (WARP_M / 32) + mi;
                        #if defined(__gfx936__) || defined(__gfx938__)
                            for(int vec_id = 0; vec_id < 2; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].u64[vec_id] = hcu_pk_mul_f32(
                                    acc_o[pv_tile_id][mmac_id].u64[vec_id],
                                    scale_pair
                                );
                            }
                        #else
                            for(int vec_id = 0; vec_id < 4; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].f32[vec_id] *= scale;
                            }
                        #endif
                    }
                }
            }
        }
    }
}



template<int WARP_M, bool Is_even_MN, bool SplitD, bool Is_Interleaved, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_lse(
    vec2_Accum<ElementAccum> lse[WARP_M / 32],
    void *softmax_lse_ptr,
    int row_offset_lse,
    int warp_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit) {

    ElementAccum * gLSE = reinterpret_cast<ElementAccum*>(softmax_lse_ptr) + row_offset_lse;
    #if (DEBUG_LEVEL >= 1)
        ElementAccum * scores_sum_ptr = reinterpret_cast<ElementAccum*>(scores_sum_ptr) + row_offset_lse;
        ElementAccum * scores_max_ptr = reinterpret_cast<ElementAccum*>(scores_max_ptr) + row_offset_lse;
    #endif
    const bool write_lse = SplitD > 1 ? (lane_id >> 4) == 0 and headdim_split_id == 0: (lane_id >> 4) == 0;
    if (write_lse) {
        #pragma unroll
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                const int row = Is_Interleaved
                    ? warp_id * WARP_M + mi * 32 + (lane_id & 15) + min_tile_m * 16
                    : warp_id * WARP_M + mi * 32 + (lane_id & 15) * 2 + min_tile_m;
                if constexpr (Is_even_MN) {
                    gLSE[row] = lse[mi].f32[min_tile_m];
                    #if (DEBUG_LEVEL >= 1)
                        scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                        scores_max_ptr[row] = scores_max[mi].f32[min_tile_m];
                    #endif
                } else {
                    if (row < seqlen_q_limit) {
                        gLSE[row] = lse[mi].f32[min_tile_m];
                        #if (DEBUG_LEVEL >= 1)
                            scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                            scores_max_ptr[row] = scores_max[mi].f32[min_tile_m];
                        #endif
                    }
                }
            }
        }
    }
}



template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_output(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {

    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;

    #if defined(__gfx938__)
        constexpr bool Is_Interleaved_ = Is_Interleaved and kHeadDimV == 128;
    #else
        constexpr bool Is_Interleaved_ = Is_Interleaved;
    #endif

    if constexpr (Is_Interleaved_) {
    #if defined(__gfx938__)
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int tile32x32_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                        int s_offset = k_loop * kBlockK;
                        int seqlen_q_offset = (warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m);
                        int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 8;
                        union_vec4_f16x2<Element> v_data;
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            constexpr bool is_bf16 = std::is_same<Element, bhalf_t>::value;
                            v_data.f16x2[vec_index][0] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                            v_data.f16x2[vec_index][1] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                        }

                        auto lds = (__attribute__((address_space(3))) float*)(0);
                        int lds_write_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + pv_lane_head_dim_idx * 4 + pv_lane_seq_idx * 4) * 4;
                        __builtin_amdgcn_sched_barrier(0);
                        inlineasm_ds_write_b128(lds_write_offset, v_data.f32);
                        flash::wait_lds_data_arrived<false>(0);
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 2; ++vec_index) {
                            int lds_load_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + vec_index * 8 + pv_lane_head_dim_idx + pv_lane_seq_idx * 4) * 4;
                            asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:%2\n":: "v"(v_data.data[vec_index]), "v"(lds_load_offset), "B"(4));
                        }
                        flash::wait_lds_data_arrived<false>(0);

                        if constexpr (Is_even_MN) {
                            *(vec4_fp32*)(o_ptr + v_offset + s_offset + k_tile_idx * 32) = v_data.f32;
                        } else {
                            if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                *(vec4_fp32*)(o_ptr + v_offset + s_offset + k_tile_idx * 32) = v_data.f32;
                            }
                        }
                    }
                }
            }
        }
    #else
        #pragma unroll
        for (int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for (int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for (int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        #pragma unroll 2
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            const int pv_tile_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                            const int mmac_id    = min_tile_m + min_tile_n * 2;
                            int seqlen_q_offset  = warp_id * WARP_M + warp_m_idx * 32 + min_tile_m * 16 + pv_lane_seq_idx;
                            int s_offset = k_tile_idx * 32 + min_tile_n * 16;
                            int v_offset = seqlen_q_offset * seqlen_o_stride + k_loop * kBlockK + pv_lane_head_dim_idx * 4;
                            union_vec2_f16x2<Element> v_data;
                            #pragma unroll
                            for (int vec_index = 0; vec_index < 2; ++vec_index) {
                                v_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(acc_o[pv_tile_id][mmac_id].f32x2[vec_index]);
                            }
                            if constexpr (not Is_even_MN) {
                                if (m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                    *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = v_data;
                                }
                            } else {
                                *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = v_data;
                            }
                        }
                    }
                }
            }
        }
    #endif
    } else {
        auto gO = prepare_for_buffer_load<kHeadDimV, Element, TcpSwizzle>(o_ptr);
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            int tile32x32_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                            int s_offset = k_loop * kBlockK;
                            int s_offset_constexpr = k_tile_idx * 32 + vec_index * 8; /*overflow for s_offset_constexpr*/
                            int seqlen_q_offset = (warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m);
                            int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 2;
                            vec2_Element<Element> v_data;
                            // convert float -> bf16/fp16
                            if constexpr (std::is_same<Element, bhalf_t>::value) {
                            #if 1
                                v_data[0] = DownCast<ElementAccum, Element, true>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                v_data[1] = DownCast<ElementAccum, Element, true>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                            #else
                                v_data[0] = inlineasm_float2bfloat16_ushort_nonan(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                v_data[1] = inlineasm_float2bfloat16_ushort_nonan(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                            #endif
                            }
                            else if constexpr (std::is_same<Element, half_t>::value) {
                            #ifdef USE_CVT_PKRTZ_FP16_FP32
                                    *(vec2_Element<Element>*)&v_data = DownCastPair<ElementAccum, Element>(
                                    acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index],
                                    acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]
                                );
                            #else
                                v_data[0] = DownCast<ElementAccum, Element>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                v_data[1] = DownCast<ElementAccum, Element>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                            #endif
                            }
                            // write to global memory
                            if constexpr (Is_even_MN) {
                                inline_buffer_store_dword<vec2_Element<Element>, 1>(v_data, v_offset, gO, s_offset, /* immediate integer */s_offset_constexpr);
                            } else {
                                if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                    *(vec2_Element<Element>*)(o_ptr + v_offset + s_offset + s_offset_constexpr) = v_data;
                                }
                            }
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
        if constexpr (Is_even_MN) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
        }
    }

}