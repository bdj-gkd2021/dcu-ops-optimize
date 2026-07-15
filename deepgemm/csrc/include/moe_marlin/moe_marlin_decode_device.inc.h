// SPDX-License-Identifier: Apache-2.0
#ifndef MOE_MARLIN_DECODE_DEVICE_INC_H
#define MOE_MARLIN_DECODE_DEVICE_INC_H

// MoE Marlin GEMM2 decode (MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_FP8) —
// verbatim slice of moe/csrc_for_aiter/moe_w8a8_opt.h. Requires the following symbols
// from the moe_marlin/ sibling headers to be already visible (do NOT include them
// from this file; the host translation unit includes them first):
//   * union_vec_opt<Element, N>          (moe_w8a8_utils.h)
//   * vec8_Element<scalar_t>             (numeric_types.h — deepgemm's local naming;
//                                         upstream moe/csrc_for_aiter calls it ``vec_element_8``)
//   * vec4_fp32                          (numeric_types.h)
//   * vec<T, N>                          (intrinsic_2.h)
//   * tcp_cache_swizzle_func<...>        (moe_w8a8_utils.h)
//   * inline_buffer_load_dword(...)      (moe_w8a8_utils.h)
//   * b32_to_b16<scalar_t>(...)          (moe_w8a8_utils.h)
//   * gemm_nt_marlin_decode_2_fp8<...>   (moe_w8a8_utils.h)
//   * mmac_fp8<Element>(...)             (intrinsic_2.h) [transitively via gemm_nt_marlin_decode_2_fp8]
//
// Tile shape locked to BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, WARP_M=16, WARP_N=32,
// WARP_K=64, STAGES=2 by the host launcher in moe_marlin_decode_launch.inc.h
// (mode 38 in moe/csrc_for_aiter/moe_w8a8_config.h).
//
// gfx938 only: mmac_fp8 dispatches to __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts;
// other archs return 0 (kernel will execute but produce zeros).

template<
  typename scalar_t,
  typename Element,
  int WARP_NUM,
  int BLOCK_SIZE_M,
  int BLOCK_SIZE_N,
  int BLOCK_SIZE_K,
  int WARP_M,
  int WARP_N,
  int WARP_K,
  int GROUP_N,
  int GROUP_K,
  int STAGES,
  bool mul_topk_weight>
__global__ void __launch_bounds__(512,1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN_FP8(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  scalar_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t*  sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n,
  uint32_t size_k,
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn,
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t real_topk
) {
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.x;

  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return;

  const uint32_t input_offset = bidz * BLOCK_SIZE_K;
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  Element* input_lds = (Element*)&(smem);
  Element* qweight_lds = input_lds;
  scalar_t* output_lds = reinterpret_cast<scalar_t*>(&(smem));

  float* b_scale_lds = (float*)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m];

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index_safe = std::min(uint32_t(token_ids * real_topk + topk_ids), size_m - 1);
      float input_scale_value = *(input_scale + token_index_safe * stride_asm);
      weight_dot_a_scale[idx] = topk_weights[token_index_safe] * input_scale_value;
  }

  float weight_dot_a_scale_value[WARP_M / mfma_m];
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    weight_dot_a_scale_value[min_tile_m] = weight_dot_a_scale[min_tile_m];
  }

  constexpr  int  n_loop_num = 1;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t* g_output;
  g_output = output + output_offset;

  // 与 prefill：moe_marlin_prefill_device.inc.h expert_id==-1 路径一致 — 使用 numeric_types.h 的 vec8_Element
  if(expert_id == -1){
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;
    vec8_Element<scalar_t> zero_element_8{};

    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))] ;
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids   = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk + topk_ids;

      if(topk_ids < real_topk ) {
        *reinterpret_cast<vec8_Element<scalar_t>*>(&g_output[(token_index) * size_n + n_idx * 8]) = zero_element_8;
      }
    }
    return;
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n)*4 ];

  #pragma unroll
  for(int n_loop=0;n_loop < n_loop_num;n_loop++){
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      vec<uint,4> b_scale_ptr_prepared =  tcp_cache_swizzle_func<64,float>(b_scale_ptr + BLOCK_SIZE_N* n_loop);

      inline_buffer_load_dword(b_scale[n_loop][min_tile_n*4+0],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+0);
      inline_buffer_load_dword(b_scale[n_loop][min_tile_n*4+1],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+4);
      inline_buffer_load_dword(b_scale[n_loop][min_tile_n*4+2],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+8);
      inline_buffer_load_dword(b_scale[n_loop][min_tile_n*4+3],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+12);
    }
  }

  int token_index_store[BLOCK_SIZE_M  * BLOCK_SIZE_N /  (WARP_NUM * 512)];
  int tok_ids_store[BLOCK_SIZE_M  * BLOCK_SIZE_N /  (WARP_NUM * 512)];
  {
    int tid = threadIdx.x;
    int N_thread = BLOCK_SIZE_N / 8;

    int n_idx = threadIdx.x % N_thread;
    int m_idx = threadIdx.x / N_thread;
    int it_num = (WARP_NUM*64) / N_thread;
    for(; m_idx < BLOCK_SIZE_M; m_idx += it_num) {
      int it =  m_idx / it_num;
      const int32_t sorted_token_ids_element_store = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ m_idx, int(sorted_token_lens - 1))];
      int token_ids_store = sorted_token_ids_element_store & 0x00FFFFFF;
      tok_ids_store[it] =  (sorted_token_ids_element_store & 0xFF000000) >> 24;
      token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};

  gemm_nt_marlin_decode_2_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
  (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, real_topk);

  __syncthreads();

  for(int n_loop=0;n_loop < n_loop_num;n_loop++){

    if(warp_k_id == 0 || warp_k_num == 1)
    {
      #pragma unroll
      for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
        #pragma unroll
        for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
          #pragma unroll
          for(int reg_id = 0; reg_id < 4; reg_id++){
            float value = C_reg[n_loop][min_tile_m * (WARP_N/mfma_n) + min_tile_n][reg_id] * weight_dot_a_scale_value[min_tile_m] * b_scale[n_loop][min_tile_n*4 + reg_id];
            int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16))/2 * 8;
            output_lds[ index] = b32_to_b16<scalar_t>(value);
          }
        }
      }
    }
    __syncthreads();

    {
      const int tid = threadIdx.x;
      constexpr int N_thread = BLOCK_SIZE_N / 8;

      int n_idx = threadIdx.x % N_thread;
      int m_idx = threadIdx.x / N_thread;
      int it_num = (WARP_NUM*64) / N_thread;
      for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
        int it =  m_idx / it_num;
        if(tok_ids_store[it] < real_topk ) {
          *reinterpret_cast<vec8_Element<scalar_t>*>(&g_output[bidz * size_m * top_k  * size_n + token_index_store[it] * size_n + n_idx * 8 + BLOCK_SIZE_N* n_loop ]) =
          *reinterpret_cast<vec8_Element<scalar_t>*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 8]);
        }
      }
    }
  }
}

#endif /* MOE_MARLIN_DECODE_DEVICE_INC_H */
