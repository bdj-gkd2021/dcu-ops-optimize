#pragma once
#include "numeric_types.h"

// DTK: __builtin_hcu_matrix_load_*_b8 第二参为 addrspace(3) char*（本机 clang 报错 short* 与 char* 不匹配）；b16 用 short* 见 intrinsic_mls_ds.h。
// 改法、soffset（trans +0x80000000）、调用方式与验证：见仓库根目录 ROCM指令迁移到DTK.md §4。
// Inline asm with "s"(vec4_uint) can lower srsrc to VGPR and fail with invalid operand; builtins keep srsrc in the correct class.

template<int r, int t>
__forceinline__ __device__ void matrix_load_128x16_b8_lds_trans_builtin(size_t lds_addr_warp, vec4_int rsrc, int /*matrix_offset*/) {
#if defined(__gfx938__)
    int soffset = static_cast<int>(lds_addr_warp) + 0x80000000;
    // Third arg must be compile-time constant (same pattern as matrix_load_b16); call sites use matrix_offset==0.
    __builtin_hcu_matrix_load_128x16_b8(
        rsrc,
        (__attribute__((address_space(3))) char*)(soffset),
        0,
        t,
        r,
        0,
        0);
#endif
}

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_128x16_b8_lds_trans(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx938__)
    union union_vec4_uint u;
    u.v32 = srsrc;
    size_t lds_addr_warp = reinterpret_cast<size_t>(shared_addr) + static_cast<size_t>(lds_offset);
    matrix_load_128x16_b8_lds_trans_builtin<r, t>(lds_addr_warp, u.i32, matrix_offset);
#endif
}

#define DS_READ_MATRIX_64x16_B8(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


template<int r, int t>
__forceinline__ __device__ void matrix_load_64x32_b8_lds_rearrange_builtin(size_t lds_addr_warp, vec4_int rsrc, int /*matrix_offset*/) {
#if defined(__gfx938__)
    int soffset = static_cast<int>(lds_addr_warp);
    __builtin_hcu_matrix_load_64x32_b8(
        rsrc,
        (__attribute__((address_space(3))) char*)(soffset),
        0,
        t,
        r,
        0,
        0);
#endif
}

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_64x32_b8_lds_rearrange(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx938__)
    union union_vec4_uint u;
    u.v32 = srsrc;
    size_t lds_addr_warp = reinterpret_cast<size_t>(shared_addr) + static_cast<size_t>(lds_offset);
    matrix_load_64x32_b8_lds_rearrange_builtin<r, t>(lds_addr_warp, u.i32, matrix_offset);
#endif
}


#define DS_READ_MATRIX_32x32_B8(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


#define DS_READ_MATRIX_32x32_B8_ALT2(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x1\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x1\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }

template<class T, class AccumType>
inline __device__ vec4_fp32 mmac_4interleave_b8(const vec8_Element<T> &v1, const vec8_Element<T> &v2, const vec4_fp32 &v3)
{
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1, v2, v3, 1, 0);
}