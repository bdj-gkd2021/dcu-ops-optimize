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

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "default_gemm_configuration.hpp"

#include "cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace cute;

template<class ElementA,class ElementB,class ElementC,class ElementAcc>
void TEST(int Iterations,char perf)
{
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassSimt, cutlass::arch::Sm75,
    ElementA, cutlass::layout::ColumnMajor,
    ElementB, cutlass::layout::RowMajor,
    ElementC, cutlass::layout::ColumnMajor,
    ElementAcc>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      typename Config::CollectiveMainloop,
      typename Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  (test::gemm::device::TestAll<Gemm>());
  if(perf == 'Y')
    test::gemm::device::TestGemmPerf3x<Gemm>(Iterations);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){

  int iterations = 100;
  char perf = 'N';
  if (argc >= 2)
    sscanf(argv[1], "%c", &perf);
  if (argc >= 3)
    sscanf(argv[2], "%d", &iterations);
  
  TEST<int8_t,int8_t,int32_t,int32_t>(iterations,perf);
  TEST<half_t,half_t,half_t,float>(iterations,perf);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
