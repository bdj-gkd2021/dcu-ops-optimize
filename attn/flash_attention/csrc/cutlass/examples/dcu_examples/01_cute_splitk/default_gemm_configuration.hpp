/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"



namespace cutlass {
namespace gemm {
namespace device {
using namespace cute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmConfigurationToCutlass3Types configuration exists.");
};

///////////////////////////////////////////////////////////////////////////////
//config A和B的copy范式
namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB;

using DefaultTileShape = Shape<_128, _128, _16>;

using TiledMma_16x16x16 = TiledMMA<
      MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT>,
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Layout<Shape<_1,_1,_1>>//,
           >; 

using TiledMma_32x16x16 = TiledMMA<
      MMA_Atom<GFX928_32x32x16_F32F16F16F32_NT_ALT>,
      Layout<Shape<_2,_2,_1>>, 
      Layout<Shape<_1,_1,_1>>>; 

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::ColumnMajor, 8, SizeK>
{
  // Smem
  // 连续的8个元素作为一个单位 128bytes = 64个元素 相当于进行swizzle 64x16的共享内存layout
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _16>,
                       Stride< _1, _64>>{}));

  using SmemCopyAtomDefault = Copy_Atom<UniversalCopy<half_t>, half_t>;
  using SmemCopyAtomDsRead  = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_ALT, half_t>;

  // Gmem
  // 如果是128x128x16的线程 相当于A矩阵256个线程处理 128x16 的block块
  // threadLayout可以设置为 16 x 16 的线程块  valLayout设置为 8 x 1

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _16>,
                           Stride< _1, _16>>{},
                    Layout<Shape < _8, _1>>{}));
};

/// Operand A - Row-major (K-Major)
template <int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::RowMajor, 8, SizeK>
{
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _16>,
                       Stride< _16, _1>>{}));

  using SmemCopyAtomDefault = Copy_Atom<UniversalCopy<half_t>, half_t>;
  using SmemCopyAtomDsRead  = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_ALT, half_t>;


  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half_t>{},
                    Layout<Shape< _128, _2>,
                           Stride< _2, _1>>{},
                    Layout<Shape < _1, _8>>{}));
};
// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the DefaultOperands
// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};
}

//使用sm75进行模版特例化 表示amd gfx928架构

template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm75,
    half_t, LayoutA,
    half_t, LayoutB,
    half_t, LayoutC,
    float>
{
  using TileShape = detail::DefaultTileShape;
  //256个线程
  static constexpr int ThreadCount = 256;
  
  using DispatchPolicy = MainloopSm70TwoStage;
  //线程数量为2x2x64
  using TiledMma = typename detail::TiledMma_16x16x16;


  // A
  static constexpr int kAlignmentA = 8;
  using DefaultOperandA = detail::DefaultGemm_TensorOpGfx928_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;

  using SmemCopyAtom = typename DefaultOperandA::SmemCopyAtomDefault;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 8;
  using DefaultOperandB = detail::DefaultGemm_TensorOpGfx928_OperandB<
        half_t, LayoutB, kAlignmentB, 32>;

  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    half_t, TagToStrideA_t<LayoutA>,
    half_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  // using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
  //                             TagToStrideC_t<LayoutC>,
  //                             TagToStrideC_t<LayoutC>,
  //                             epilogue::thread::LinearCombination<half_t, 1, float, float>,
  //                             cutlass::gemm::EpilogueDefault>;

  using MNK = typename TiledMma::TiledShape_MNK;

  using SmemLayoutC = decltype(make_layout(make_shape(get<0>(MNK{}), get<1>(MNK{})),
                                          make_stride(Int<1>{}, get<0>(MNK{}))));

  // 如果SmemLayoutC是32x32 这时候R2S应该是使用threadLayout 8x32 valLayout 4x1 相应的s2r向量化数据应该是
  using CollectiveEpilogue = epilogue::collective::Epilogue<
                                TagToStrideC_t<LayoutC>,
                                TagToStrideC_t<LayoutC>,
                                epilogue::thread::LinearCombination<half_t, 1, float, float>,
                                SmemLayoutC,
                                Copy_Atom<UniversalCopy<float>,float>,                                // R2S with tiled_mma layout
                                decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{}, // S2R
                                                         Layout<Shape<_8, _32>,
                                                                Stride<_1, _8>>{},                   // Thread layout
                                                         Layout<Shape<_4, _1>>{})),                   // Value layout
                                Copy_Atom<UniversalCopy<half_t>, half_t>                            // R2G with S2R_dst layout
  >;
};


//
// DP4A - int8    Proof-of-concept
//

// SIMT Two Stage TN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  // NOTE: permuting MMA M mode lets us generate 128b smem loads (LDS.128) but has worst case bank conflicts
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16,_16,_1>>>;  // Tile of atoms (threads)

  // A (M,K)  K-major
  using ElementA = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomA = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementA>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomB = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementB>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

// SIMT Two Stage NT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

// SIMT Two Stage NT - idp2a
template <
  class ArchTag,
  class LayoutA, class LayoutB,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    half_t, LayoutA,
    half_t, LayoutB,
    ElementC, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP2A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = half_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,  _256>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;

  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::half_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1, _32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  N-major
  using ElementB = half_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,  _256>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;

  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::half_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, float, float>,
    cutlass::gemm::EpilogueDefault>;
};



} // namespace device
} // namespace gemm
} // namespace cutlass
