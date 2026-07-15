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
/* \file
   \brief Helper functions for mapping CUTLASS concepts to cuBLAS.
*/

#pragma once

#if CUTLASS_ENABLE_CUBLAS
#include <hipblas.h>

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/blas3.h"

#include "options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a cuBLAS status to cutlass::Status
Status get_cutlass_status(hipblasStatus_t cublas);

/// Converts a cuBLAS status to cutlass::profiler::Disposition
Disposition get_cutlass_disposition(hipblasStatus_t cublas_status);

/// Maps a CUTLASS tensor layout to a cuBLAS transpose operation
bool get_cublas_transpose_operation(
  hipblasOperation_t &operation,
  library::LayoutTypeID layout,
  library::ComplexTransform transform = library::ComplexTransform::kNone);

/// Maps a CUTLASS numeric type to a cuBLAS data type enumeration
bool get_cublas_datatype(hipblasDatatype_t &data_type, library::NumericTypeID element_type);

/// Gets the cublas algorithm given threadblock tile dimensions and math opcode class
hipblasGemmAlgo_t get_cublas_gemm_algo(
  int cta_m, 
  int cta_n, 
  int cta_k, 
  library::OpcodeClassID opcode_class);

/// Returns a status if cuBLAS can satisfy a particular GEMM description
Status cublas_satisfies(library::GemmDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular RankK description
Status cublas_satisfies(library::RankKDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular TRMM description
Status cublas_satisfies(library::TrmmDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular SYMM/HEMM description
Status cublas_satisfies(library::SymmDescription const &desc);

/// This is a helper class to create hipblasHandle_t automatically on CublasCreate object creation and 
/// to destroy hipblasHandle_t on CublasCreate object destruction. 
/// Additionally, it provides implicit cast from CublasCreate's object to hipblasHandle_t's object
class CublasCreate {
private:
	hipblasHandle_t handle;
	hipblasStatus_t status;

public:
	CublasCreate() {
		status = hipblasCreate(&handle);
	}

	~CublasCreate() {
		hipblasDestroy(handle);
	}

    /// Implicit cast CublasCreate object to hipblasHandle_t
    operator hipblasHandle_t() const { return handle; }

    /// returns hipblasStatus_t for handle creation
    hipblasStatus_t get_cublas_create_status() { return status; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Selects one or more cuBLAS algorithms.
static void select_cublas_algorithms(
  std::vector<hipblasGemmAlgo_t> &algorithms,
  Options const &options, 
  library::GemmDescription const &op_desc) {
    // for hipblas, only HIPBLAS_GEMM_DEFAULT is provided
    algorithms.push_back(HIPBLAS_GEMM_DEFAULT); 
}

/// Dispatcher to hipblasGemmEx() 
struct cublasGemmExDispatcher {

  //
  // Data members
  //
  library::GemmUniversalConfiguration configuration;
  library::GemmUniversalArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  hipblasOperation_t trans_A;
  hipblasOperation_t trans_B;
  hipblasDatatype_t data_type_A;
  hipblasDatatype_t data_type_B;
  hipblasDatatype_t data_type_C;
  hipblasDatatype_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  hipblasDatatype_t compute_type;
#endif

  hipblasGemmAlgo_t algo;
  Status status;
  
  //
  // Methods
  //

  cublasGemmExDispatcher( 
    library::GemmDescription const &op_desc,
    library::GemmUniversalConfiguration configuration_,
    library::GemmUniversalArguments arguments_,
    hipblasGemmAlgo_t algorithm = HIPBLAS_GEMM_DEFAULT
  );

  /// Executes GEMM using these arguments
  hipblasStatus_t operator()(hipblasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublas rank k update kernels 
struct cublasRankKDispatcher {

  //
  // Data members
  //
  library::RankKConfiguration configuration;
  library::RankKArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  hipblasOperation_t trans_A;
  hipblasFillMode_t uplo;
  hipblasDatatype_t data_type_A;
  hipblasDatatype_t data_type_C;
  hipblasDatatype_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  hipblasDatatype_t compute_type;
#endif

  int num_ranks;       //(rank-k or rank-2k)
  BlasMode blas_mode; //(symmetric or hermitian)
  Status status;
  
  //
  // Methods
  //

  cublasRankKDispatcher( 
    library::RankKDescription const &op_desc,
    library::RankKConfiguration configuration_,
    library::RankKArguments arguments_
  );

  /// Executes RankK using these arguments
  hipblasStatus_t operator()(hipblasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublasTrmm() 
struct cublasTrmmDispatcher {

  //
  // Data members
  //
  library::TrmmConfiguration configuration;
  library::TrmmArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  hipblasOperation_t trans_A;
  hipblasSideMode_t side;
  hipblasFillMode_t uplo;
  hipblasDiagType_t diag;
  hipblasDatatype_t data_type_A;
  hipblasDatatype_t data_type_B;
  hipblasDatatype_t data_type_D;
  hipblasDatatype_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  hipblasDatatype_t compute_type;
#endif

  Status status;
  
  //
  // Methods
  //

  cublasTrmmDispatcher( 
    library::TrmmDescription const &op_desc,
    library::TrmmConfiguration configuration_,
    library::TrmmArguments arguments_
  );

  /// Executes TRMM using these arguments
  hipblasStatus_t operator()(hipblasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublas symm/hemm update kernels 
struct cublasSymmDispatcher {

  //
  // Data members
  //
  library::SymmConfiguration configuration;
  library::SymmArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  hipblasSideMode_t side;
  hipblasFillMode_t uplo;
  hipblasDatatype_t data_type_A;
  hipblasDatatype_t data_type_B;
  hipblasDatatype_t data_type_C;
  hipblasDatatype_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  hipblasDatatype_t compute_type;
#endif
  
  BlasMode blas_mode; //(symmetric or hermitian)
  Status status;
  
  //
  // Methods
  //

  cublasSymmDispatcher( 
    library::SymmDescription const &op_desc,
    library::SymmConfiguration configuration_,
    library::SymmArguments arguments_
  );

  /// Executes Symm using these arguments
  hipblasStatus_t operator()(hipblasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

} // namespace profiler
} // namespace cutlass


#endif // #if CUTLASS_ENABLE_CUBLAS
