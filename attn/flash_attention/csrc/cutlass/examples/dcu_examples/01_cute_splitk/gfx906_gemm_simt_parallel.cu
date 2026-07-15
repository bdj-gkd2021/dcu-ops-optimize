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
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "default_gemm_configuration.hpp"

#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "common/helper.h"

using namespace cute;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::ColumnMajor;

using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassSimt, cutlass::arch::Sm75, 
    ElementA, LayoutA, ElementB, LayoutB,
    ElementAcc, LayoutC, ElementAcc>; // 根据用户的需要，输出的workspace类型可以为高精度类型也可以为低精度类型

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                         typename Config::CollectiveMainloop,
                                         typename Config::CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using ElementD = typename Gemm::GemmKernel::ElementD;
using StrideD = typename Gemm::GemmKernel::StrideD;
using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;
using LayoutTagC = cutlass::detail::StrideToLayoutTagA_t<StrideC>;
using LayoutTagD = cutlass::detail::StrideToLayoutTagA_t<StrideD>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,       // 最终输出结果的类型                                 
      1,     
      ElementAcc,                                   
      ElementAcc>;
using ElementScalar = typename EpilogueOp::ElementScalar;
using ReduceGemmSplitKShape = cutlass::MatrixShape<4, 64>;
using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    typename EpilogueOp::ElementAccumulator,
    ElementAcc,        // 与workspace结果类型保持一致
    EpilogueOp::kCount>;
using ReductionKernel =
    cutlass::reduction::kernel::ReduceSplitK<ReduceGemmSplitKShape, EpilogueOp,
                                             ReductionOp>;
using ReduceGemmSplitK =
    cutlass::reduction::device::ReduceSplitK<ReductionKernel>;

void compare_result(cutlass::HostTensor<ElementC, LayoutTagD> tensor_D,
                    cutlass::HostTensor<ElementC, LayoutTagD> reference_D) {
  float max_error = 0.0;
  float max_error_splitk = 0.0;
  float max_error_reference = 0.0;
  float difference_ = 0.0;
  int error_nums = 0;
  auto extent = tensor_D.host_view().extent();
  for (int i = 0; i < extent[0]; i++) {
    for (int j = 0; j < extent[1]; j++) {
      float split_k = tensor_D.host_view().at({i, j});
      float reference = reference_D.host_view().at({i, j});
      float difference = split_k - reference;
      float relative_error =
          std::fabs(reference) < 1e-5
              ? difference
              : std::fabs(difference) / (std::fabs(reference) + 1e-6);
      if (std::fabs(difference) > 0) {
        error_nums++;
      }
      if (relative_error > max_error) {
        max_error = relative_error;
        max_error_splitk = split_k;
        max_error_reference = reference;
        difference_ = difference;
      }
    }
  }
  if (max_error == 0)
    printf("cutlass_result no error\n");
  else {
    printf("Max error=%f splitK:%f reference:%f Difference=%f error_num=%d\n",
           max_error, max_error_splitk, max_error_reference, difference_,
           error_nums);
  }
}

int run(int m,int n,int k,int slice_k) {
  int batch_count = 1;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, batch_count));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, batch_count));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, batch_count));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, batch_count));
  auto a_coord = cutlass::make_Coord(m * batch_count, k);
  auto b_coord = cutlass::make_Coord(k, n * batch_count);
  auto c_coord = cutlass::make_Coord(m * batch_count, n);
  
  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  cutlass::HostTensor<ElementC, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementC, LayoutTagD> reference_D;

  ProblemShapeType problem_size = ProblemShapeType{m, n, k, slice_k};
  double alpha = 1.0;
  double beta = 1.0;
  
  tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
  tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
  tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
  tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
  reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
  
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_A.host_view(),
    6118,
    ElementA(5),
    ElementA(-5),
    0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_B.host_view(),
    6117,
    ElementB(5),
    ElementB(-5),
    0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_C.host_view(),
    6116,
    ElementC(5),
    ElementC(-5),
    0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
    tensor_D.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
    reference_D.host_view());  // <- fill matrix D for reference on host with zeros

  cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
  reference_D.sync_device();

  auto arguments = typename Gemm::Arguments {
    cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
    problem_size,
    {
      tensor_A.device_data(), stride_a,
      tensor_B.device_data(), stride_b
    },
    {
      {cutlass::from_real<ElementScalar>(alpha), cutlass::from_real<ElementScalar>(beta)},
      reinterpret_cast<ElementAcc*>(tensor_C.device_data()), stride_c, reinterpret_cast<ElementAcc*>(tensor_D.device_data()), stride_d
      // tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
    }
  };
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Run the GEMM
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op.run();
  CUTLASS_CHECK(status);
  hipDeviceSynchronize();

  cutlass::gemm::GemmCoord problem_size_(m, n, k);

  /*
  Parallel mode中会根据slice和problem.k确定k方向的划块个数，
  实际Gemm launch的块个数不一定保证与slice一致，
  因此这里reduction规约的块数需要和Gemm launch的块数保持一致
  */
  batch_count = gemm_op.get_grid_shape(arguments).z;
  int splitk_gemm_stride = get<0>(problem_size); 
  cutlass::layout::RowMajor splitk_gemm_layout(splitk_gemm_stride);       
  void * workspace_gemm_ptr = workspace.get();
  cutlass::TensorRef<ElementAcc, cutlass::layout::RowMajor> workspace_gemm_tensorref(static_cast<ElementAcc *>(workspace_gemm_ptr), splitk_gemm_layout);
  cutlass::TensorRef<ElementC, cutlass::layout::RowMajor> tensor_d_tensorref(tensor_D.device_ref().data(), splitk_gemm_layout);
  cutlass::TensorRef<ElementC, cutlass::layout::RowMajor> tensor_c_tensorref(tensor_C.device_ref().data(), splitk_gemm_layout);    
  typename ReduceGemmSplitK::Arguments reduce_gemm_splitk_arguments{
    cutlass::MatrixCoord(get<1>(problem_size), get<0>(problem_size)),
    batch_count,
    size_t(get<0>(problem_size) * get<1>(problem_size)),
    workspace_gemm_tensorref,
    tensor_d_tensorref,
    tensor_c_tensorref,
    {cutlass::from_real<ElementScalar>(alpha), cutlass::from_real<ElementScalar>(beta)} 
  };
  ReduceGemmSplitK reduce_gemm_splitk_op;

  // ReductionOp
  status = reduce_gemm_splitk_op.initialize(reduce_gemm_splitk_arguments);
  CUTLASS_CHECK(status);
  status = reduce_gemm_splitk_op();
  CUTLASS_CHECK(status);

  // verify
  cutlass::reference::device::Gemm<ElementA,
                                   LayoutTagA,
                                   ElementB,
                                   LayoutTagB,
                                   ElementC,
                                   LayoutTagC,
                                   ElementAcc,
                                   ElementAcc> gemm_device;

  gemm_device(problem_size_,
              alpha,
              tensor_A.device_ref(),
              tensor_B.device_ref(),
              beta,
              tensor_C.device_ref(),
              reference_D.device_ref());

  tensor_D.sync_host();
  reference_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

  printf("problem size:%d %d %d slice_k=%d\n",m,n,k,slice_k);
  if (passed) {
    printf("check passed!\n");
  }
  else {
    printf("error!\n");
#if 0
    std::stringstream fname;
    fname << "error_Gemm_device_"
      << m << "x" << n << "x" << k << "x" << batch_count << "_"
      << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
      << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
      << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".csv";

    std::ofstream file(fname.str());
    file
      << "problem: " << ' ' << m << "x" << n << "x" << k << ", Batch count = " << batch_count
      << ", alpha: " << float(alpha) << ", beta: " << float(beta) << "\n\n";

    file
      << "A =\n" << tensor_A.host_view()
      << "\nB =\n" << tensor_B.host_view()
      << "\nC =\n" << tensor_C.host_view()
      << "\n\nReference =\n" << reference_D.host_view()
      << "\n\nComputed =\n" << tensor_D.host_view();
#else
    compare_result(tensor_D, reference_D);
#endif
  }
  printf("===================================\n");
  return 0;
}

int main(int argc, char* argv[]) {
  int m = 256;
  int n = 256;
  int k = 32;
  int slice_k = 1;
  if (argc >= 2) {
    m = atoi(argv[1]);
  }
  if (argc >= 3) {
    n = atoi(argv[2]);
  }
  if (argc >= 4) {
    k = atoi(argv[3]);
  }
  if (argc >= 5) {
    slice_k = atoi(argv[4]);
  }
  return run(m,n,k,slice_k);
}
