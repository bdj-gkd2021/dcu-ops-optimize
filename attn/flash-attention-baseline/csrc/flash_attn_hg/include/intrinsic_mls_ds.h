#pragma once
#include <vector>
#include "numeric_types.h"
#include "intrinsic.h"

// ======================================================= MLS ===========================================================

#define VA_LIMIT_BITS(x) (0xffffffffffff & x)

template<int INSTM, int INSTNM, int T, int R>
__forceinline__ __device__ void matrix_load_b16_lds_trans_builtin(size_t lds_addr_warp, vec4_int rsrc, int moffset) {
#if defined(__gfx938__)
    int soffset = lds_addr_warp + 0x80000000;

    if constexpr (INSTM == 32 && INSTNM == 16) {
        __builtin_hcu_matrix_load_32x16_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    } else if constexpr (INSTM == 32 && INSTNM == 32) {
        __builtin_hcu_matrix_load_32x32_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    } else if constexpr (INSTM == 64 && INSTNM == 16) {
        __builtin_hcu_matrix_load_64x16_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    }
    (void)moffset;
#endif
}

template<int INSTM, int INSTNM, int T, int R>
__forceinline__ __device__ void matrix_load_b16_lds_builtin(size_t lds_addr_warp, vec4_int rsrc, int moffset) {
#if defined(__gfx938__)
    int soffset = lds_addr_warp + 0x00000000;

    if constexpr (INSTM == 32 && INSTNM == 16) {
        __builtin_hcu_matrix_load_32x16_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    } else if constexpr (INSTM == 32 && INSTNM == 32) {
        __builtin_hcu_matrix_load_32x32_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    } else if constexpr (INSTM == 64 && INSTNM == 16) {
        __builtin_hcu_matrix_load_64x16_b16(rsrc, (__attribute__((address_space(3))) short*)(soffset), 0, T, R, 0, 0);
    }
    (void)moffset;
#endif
}

// ======================================================= DS ===========================================================

#define DS_READ_MATRIX_32X32_B16(OFFSET, REG, REG1, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_trans_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    }

#define DS_READ_MATRIX_32X16_B16(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


#define DS_READ_MATRIX_32X16_B16_ALT2(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }

#define DS_READ_MATRIX_32X32_B16_ALT2(OFFSET, REG, REG1, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            "ds_read_matrix_trans_format %1, m0 offset:1024 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            "ds_read_matrix_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    }


template<int min_value, int max_value>
__forceinline__ __device__ int inline_min_max(int source) {
    /*
        To avoid usage of v_med3_i32
            ----> to avoid usage of __builtin_amdgcn_readfirstlane
                ----> to avoid usage of 5 nops for mls data hazard
    */
    return max(min_value, min(max_value, source));
    // int result;
    // asm volatile("s_max_i32 %0, %1, %2\n\t"
    //              "s_min_i32 %0, %0, %3\n"
    //            : "=s"(result)
    //            : "s"(source), "n"(min_value), "n"(max_value)
    //            :);
    // return result;
}

// ======================================================= def ===========================================================

template<typename VEC>
__forceinline__ __device__ void ds_mpermute_kdim_for_mmac(VEC& data) {
    asm volatile("ds_mpermute_dwordx2 %0, %0 offset:6\n":: "v"(data));
}

template<typename VEC>
__forceinline__ __device__ void ds_mpermute_kdim_for_mmac_wait(VEC& data) {
    asm volatile("ds_mpermute_dwordx2 %0, %0 offset:6\n\ts_waitcnt lgkmcnt(0)":: "v"(data));
}


// ======================================================= mmac ===========================================================
template<class T, class AccumType>
inline __device__ vec4_fp32 mmac_4interleave(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
{
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
}

template<>
inline __device__ vec4_fp32 mmac_4interleave<half_t, float>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(v1, v2, v3, 1, 0);
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
inline __device__ vec4_fp32 mmac_4interleave<bhalf_t, float>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(v1, v2, v3, 1, 0);
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}
