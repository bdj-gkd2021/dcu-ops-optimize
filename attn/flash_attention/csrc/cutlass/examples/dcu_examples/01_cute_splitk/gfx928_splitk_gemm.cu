/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Tests for device-wide GEMM interface
*/
#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "common/gemm_testbed_3x.hpp"

using namespace cute;

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 8;

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
constexpr int AlignmentB = 8;

using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 1;

using ElementAccumulator = float;
using ElementCompute = ElementAccumulator;

using TileShape_MNK = Shape<_128, _128, _32>;
using WarpShape_MNK = Shape<_2,_2,_1>;
using InstructionShape_MNK = Shape<_16,_16,_16>;
using ClusterShape_MNK = Shape<_1,_1,_1>;

using StageCountType = cutlass::gemm::collective::StageCountAuto;
using KernelScheduleType = cutlass::gemm::KernelMultistage;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using RoundStyle = cutlass::epilogue::collective::EpilogueScheduleAuto;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm75, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_MNK, WarpShape_MNK, InstructionShape_MNK, ClusterShape_MNK,
    StageCountType, KernelScheduleType
  >::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm75, cutlass::arch::OpClassTensorOp,  
    TileShape_MNK, ClusterShape_MNK, 
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    RoundStyle
  >::CollectiveOp;
  
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
using Testbed = test::gemm::device::Testbed3x<Gemm, cutlass::epilogue::thread::Identity>;

int main(int argc, char* argv[]){
  int m = 256;
  int n = 256;
  int k = 32;
  int slice_k = 1;

  if(argc >= 2){
    m = atoi(argv[1]);
  }
  if(argc >= 3){
    n = atoi(argv[2]);
  }
  if(argc >= 4){
    k = atoi(argv[3]);
  }
  if(argc >= 5){
    slice_k = atoi(argv[4]);
  }

  printf("cutlass 3.x_tensorcore:\n");
  Testbed testbed = {};
  ProblemShapeType problem_size;
  double alpha,beta;

  problem_size = ProblemShapeType{m, n, k, slice_k};
  alpha = 1.0;
  beta = 0.0;

  bool passed = testbed.run(
    problem_size,
    cutlass::from_real<ElementScalar>(alpha),
    cutlass::from_real<ElementScalar>(beta)
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

