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

/////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    half_t, cutlass::layout::ColumnMajor,
    half_t, cutlass::layout::RowMajor,
    half_t, cutlass::layout::ColumnMajor,
    float>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  (test::gemm::device::TestAll<Gemm>());


  int iterations = 100;
  char perf = 'N';
  if (argc >= 2)
    sscanf(argv[1], "%c", &perf);
  if (argc >= 3)
    sscanf(argv[2], "%d", &iterations);

  int m = 4096;

  int n = 4096;

  int k = 128;

  if (argc >= 4)
    sscanf(argv[3], "%d", &m);
  if (argc >= 5)
    sscanf(argv[4], "%d", &n);
  if (argc >= 6)
    sscanf(argv[5], "%d", &k);
  
  if(perf == 'Y')
    test::gemm::device::TestGemmPerf3x<Gemm>(iterations,m,n,k);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
