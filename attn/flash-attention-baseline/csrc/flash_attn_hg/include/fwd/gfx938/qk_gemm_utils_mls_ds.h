#pragma once
#include "intrinsic_mls_ds.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void  prefetch_q_to_vgpr_mls_ds(
        vec4_uint q_ptr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        int warp_id,
        int seqlen_q_stride,
        int max_seq_q_offset=0) {

    // 编译期可知变量
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int Q_LDS_LOAD_NUM  = kBlockM * kBlockK / (32 * 32);
    constexpr int Q_LOAD_REQUESTS = Q_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // LDS 起始地址
    int q_lds_base = reinterpret_cast<size_t>(q_lds);
    flash::wait_lds_data_arrived<true>(0);

    // MLS
    vec4_uint q_srsrc;
    q_srsrc[2] = seqlen_q_stride; // stride
    q_srsrc[3] = 0;

    int stage_id = 0;
    {
        int k_loop = 0;
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_ptr + (k_loop * kBlockK + warp_id * 32 * seqlen_q_stride) * ELEMENT_BYTES);
        if constexpr (true) {
            int nm_filter = inline_min_max<0, 32>(32 * warp_id + 32 - max_seq_q_offset);
            q_srsrc[3] = max_seq_q_offset % kBlockM == 0 ? 0: nm_filter << 8; // set only once
        }
        int lds_offset = (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        union union_vec4_uint q_rsrc_bits;
        q_rsrc_bits.v32 = q_srsrc;
        size_t lds_addr_warp = reinterpret_cast<size_t>(q_lds) + lds_offset;
        matrix_load_b16_lds_trans_builtin<32, 32, 1, 0>(lds_addr_warp, q_rsrc_bits.i32, 0);
    }

    stage_id ^= 1;
    #pragma unroll
    for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_ptr + (k_loop * kBlockK + warp_id * 32 * seqlen_q_stride) * ELEMENT_BYTES);
        if constexpr (true) {
            int nm_filter = inline_min_max<0, 32>(32 * warp_id + 32 - max_seq_q_offset);
            q_srsrc[3] = max_seq_q_offset % kBlockM == 0 ? 0: nm_filter << 8; // set only once
        }
        int lds_offset = (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        union union_vec4_uint q_rsrc_bits;
        q_rsrc_bits.v32 = q_srsrc;
        size_t lds_addr_warp = reinterpret_cast<size_t>(q_lds) + lds_offset;
        matrix_load_b16_lds_trans_builtin<32, 32, 1, 0>(lds_addr_warp, q_rsrc_bits.i32, 0);

        stage_id ^= 1;
        // DS
        buffer_load_lds_dwordx1_wait<Q_LOAD_REQUESTS>();
        __builtin_amdgcn_sched_barrier(0);

        int lds_load_offset = q_lds_base + (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        #ifdef __gfx938__
            DS_READ_MATRIX_32X32_B16(lds_load_offset, q_reg[(k_loop - 1) * 2].f16, q_reg[(k_loop - 1) * 2 + 1].f16, true);
        #endif
        // __syncthreads();
        flash::wait_lds_data_arrived<true>(0);
    }

    {
        stage_id ^= 1;
        // DS
        buffer_load_lds_dwordx1_wait<0>();
        __builtin_amdgcn_sched_barrier(0);
        constexpr int k_loop = kHeadDim / kBlockK - 1;
        int lds_load_offset = q_lds_base + (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        #ifdef __gfx938__
            DS_READ_MATRIX_32X32_B16(lds_load_offset, q_reg[k_loop * 2].f16, q_reg[k_loop * 2 + 1].f16, true);
        #endif
    }
    __builtin_amdgcn_s_waitcnt(0);
    // __syncthreads();
    flash::wait_lds_data_arrived<true>(0);
}




template<int kHeadDim, int kBlockN, int kBlockK, int WARP_NUM, int WARP_N, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefetch_k_to_lds_mls_ds(
        vec4_uint k_ptr,
        Element* k_lds,
        int warp_id,
        int seqlen_k_stride,
        int max_seq_k_offset=0) {
    constexpr int kHeadDim_ = (kHeadDim == 192) ? 128 : kHeadDim;
    constexpr int ELEMENT_BYTES = sizeof(Element);

    int stage_id = 0;
    int n_loop   = 0;
    int k_load   = 0;
    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    if constexpr (true) {
        int nm_filter = inline_min_max<0, 32>(n_loop * WARP_N + 32 - max_seq_k_offset);
        if constexpr (kHeadDim == 192) {
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop * WARP_N * seqlen_k_stride + warp_id * 32 + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
        } else {
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop * WARP_N * seqlen_k_stride + warp_id * 32) * ELEMENT_BYTES);
        }
        k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
    }
    int lds_offset = (stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 32) * ELEMENT_BYTES;
    flash::wait_all_warp_arrived();
    union union_vec4_uint k_rsrc_bits;
    k_rsrc_bits.v32 = k_srsrc;
    size_t lds_addr_warp = reinterpret_cast<size_t>(k_lds) + lds_offset;
    matrix_load_b16_lds_trans_builtin<32, 32, 0, 0>(lds_addr_warp, k_rsrc_bits.i32, 0);

}
