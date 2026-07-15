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

#include <stdexcept>

#if CUTLASS_ENABLE_CUBLAS
#include "cutlass/profiler/cublas_helpers.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a cuBLAS status to cutlass::Status
Status get_cutlass_status(hipblasStatus_t cublas) {

  switch (cublas) {
    case HIPBLAS_STATUS_SUCCESS: 
      return Status::kSuccess;
    case HIPBLAS_STATUS_INVALID_VALUE:
      return Status::kErrorInvalidProblem;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return Status::kErrorNotSupported;
    default: break;
  }
  return Status::kErrorInternal;
}

/// Converts a cuBLAS status to cutlass::profiler::Disposition
Disposition get_cutlass_disposition(hipblasStatus_t cublas_status) {

  if (cublas_status == HIPBLAS_STATUS_INVALID_VALUE) {
    return Disposition::kInvalidProblem;
  }
  else if (cublas_status == HIPBLAS_STATUS_NOT_SUPPORTED) {
    return Disposition::kNotSupported;
  }
  return Disposition::kFailed;
}

/// Maps a CUTLASS tensor layout to a cuBLAS transpose operation
bool get_cublas_transpose_operation(
  hipblasOperation_t &operation,
  library::LayoutTypeID layout, 
  library::ComplexTransform transform) {

  switch (layout) {
    case library::LayoutTypeID::kColumnMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = HIPBLAS_OP_N;
        return true;
      }
      else {
        return false;
      }
      break;
    case library::LayoutTypeID::kRowMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = HIPBLAS_OP_T;
        return true;
      }
      else if (transform == library::ComplexTransform::kConjugate) {
        operation = HIPBLAS_OP_C;
        return true;
      }
      break;
    default: break;
  }

  return false;
}

/// Maps a CUTLASS numeric type to a cuBLAS data type enumeration
bool get_cublas_datatype(hipblasDatatype_t &data_type, library::NumericTypeID element_type) {
  switch (element_type) {
  case library::NumericTypeID::kFE4M3:
#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
    data_type = CUDA_R_8F_E4M3;
    return true;
#endif
    break;
  
  case library::NumericTypeID::kFE5M2:
#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
    data_type = CUDA_R_8F_E5M2;
    return true;
#endif
    break;

  case library::NumericTypeID::kF16:
    data_type = HIPBLAS_R_16F;
    return true;
    
  case library::NumericTypeID::kBF16:
    data_type = HIPBLAS_R_16B;
    return true;
  
  case library::NumericTypeID::kTF32: 
    break;
  
  case library::NumericTypeID::kF32:
    data_type = HIPBLAS_R_32F;
    return true;
    
  case library::NumericTypeID::kF64: 
    data_type = HIPBLAS_R_64F;
    return true;
  
  case library::NumericTypeID::kS4: 
    break;
  
  case library::NumericTypeID::kS8: 
    data_type = HIPBLAS_R_8I;
    return true;
    
  case library::NumericTypeID::kS16: 
    break;
 
  case library::NumericTypeID::kS32: 
    data_type = HIPBLAS_R_32I;
    return true;
    
  case library::NumericTypeID::kS64: 
    break;
  
  case library::NumericTypeID::kU4: 
    break;
  
  case library::NumericTypeID::kU8: 
    data_type = HIPBLAS_R_8U;
    return true;
    
  case library::NumericTypeID::kU16: 
    break;
    
  case library::NumericTypeID::kU32: 
    data_type = HIPBLAS_R_32U;
    return true;
    
  case library::NumericTypeID::kU64: 
    break;

  case library::NumericTypeID::kB1: 
    break;

  case library::NumericTypeID::kCF32:
    data_type = HIPBLAS_C_32F;
    return true;

  case library::NumericTypeID::kCF64:
    data_type = HIPBLAS_C_64F;
    return true;
  
  case library::NumericTypeID::kInvalid:
  
  default: 
    break;
  }

  return false;
}

/// Maps a cutlass::SideMode to cuBLAS side mode
bool get_cublas_side_mode(hipblasSideMode_t& side, SideMode side_mode) {

  switch (side_mode) {
    case SideMode::kLeft: 
      side = HIPBLAS_SIDE_LEFT;
      return true;
    case SideMode::kRight: 
      side = HIPBLAS_SIDE_RIGHT;
      return true;
    default: break;
  }

  return false;
}

/// Maps a cutlass::FillMode to cuBLAS fill mode
bool get_cublas_fill_mode(hipblasFillMode_t& uplo, FillMode fill_mode) {

  switch (fill_mode) {
    case FillMode::kLower: 
      uplo = HIPBLAS_FILL_MODE_LOWER;
      return true;
    case FillMode::kUpper: 
      uplo = HIPBLAS_FILL_MODE_UPPER;
      return true;
    default: break;
  }

  return false;
}

/// Maps a cutlass::DiagType to cuBLAS diag type
bool get_cublas_diag_type(hipblasDiagType_t& diag, DiagType diag_type) {

  switch (diag_type) {
    case DiagType::kNonUnit: 
      diag = HIPBLAS_DIAG_NON_UNIT;
      return true;
    case DiagType::kUnit: 
      diag = HIPBLAS_DIAG_UNIT;
      return true;
    default: break;
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets the cublas algorithm given threadblock tile dimensions and math opcode class
hipblasGemmAlgo_t get_cublas_gemm_algo(int cta_m, int cta_n, int cta_k, library::OpcodeClassID opcode_class) {
  // return (opcode_class == library::OpcodeClassID::kSimt ? 
  //   HIPBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  // use HIPBLAS_GEMM_DEFAULT duo to only it's provided
  return HIPBLAS_GEMM_DEFAULT;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular GEMM description
Status cublas_satisfies(library::GemmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasGemmExDispatcher::cublasGemmExDispatcher(
  library::GemmDescription const &op_desc,
  library::GemmUniversalConfiguration configuration_,
  library::GemmUniversalArguments arguments_,
  hipblasGemmAlgo_t algorithm
):
  configuration(configuration_), arguments(arguments_), algo(algorithm), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_transpose_operation(trans_B, op_desc.B.layout, op_desc.transform_B));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    op_desc.B.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case HIPBLAS_R_32F:
      case HIPBLAS_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case HIPBLAS_R_64F:
      case HIPBLAS_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case HIPBLAS_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case HIPBLAS_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes GEMM using these arguments
hipblasStatus_t cublasGemmExDispatcher::operator()(hipblasHandle_t handle) {

  if (configuration.mode == library::GemmUniversalMode::kBatched) {
    return hipblasGemmStridedBatchedEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.batch_stride_A,
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.batch_stride_B,
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
      arguments.batch_stride_C,
      configuration.batch_count,
  // #if (__CUDACC_VER_MAJOR__ >= 11)
  //     compute_type,
  // #else
      compute_data_type,
  // #endif
      algo
    );
  }
  else {
    return hipblasGemmEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
  // #if (__CUDACC_VER_MAJOR__ >= 11)
  //     compute_type,
  // #else
      compute_data_type,
  // #endif
      algo
    );
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular RankK description
Status cublas_satisfies(library::RankKDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasRankKDispatcher::cublasRankKDispatcher(
  library::RankKDescription const &op_desc,
  library::RankKConfiguration configuration_,
  library::RankKArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  blas_mode = op_desc.blas_mode;
  num_ranks = op_desc.num_ranks;

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case HIPBLAS_R_32F:
      case HIPBLAS_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case HIPBLAS_R_64F:
      case HIPBLAS_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case HIPBLAS_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case HIPBLAS_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes RankK using these arguments
hipblasStatus_t cublasRankKDispatcher::operator()(hipblasHandle_t handle) {
 
  // SYRK and HERK
  if (num_ranks == 1) {
    if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_64F) {
      return hipblasDsyrk(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const double*>(arguments.alpha),
        static_cast<const double*>(arguments.A),
        int(configuration.lda),
        static_cast<const double*>(arguments.beta),
        static_cast<double*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
        return HIPBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      return hipblasSsyrk(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const float*>(arguments.alpha),
        static_cast<const float*>(arguments.A),
        int(configuration.lda),
        static_cast<const float*>(arguments.beta),
        static_cast<float*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_64F) {
      
        if (blas_mode == BlasMode::kHermitian) {
          return hipblasZherk(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            static_cast<const double*>(arguments.alpha),
            // static_cast<const hipDoubleComplex*>(arguments.A),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
            int(configuration.lda),
            static_cast<const double*>(arguments.beta),
            reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
            
            int(configuration.ldc)
          );
        }    
        else {
          return hipblasZsyrk(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
            int(configuration.lda),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.beta),
            reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }
  
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
        return HIPBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      if (blas_mode == BlasMode::kHermitian) {
        return hipblasCherk(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          static_cast<const float*>(arguments.alpha),
          reinterpret_cast<const hipblasComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const float*>(arguments.beta),
          reinterpret_cast<hipblasComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
      else {
        return hipblasCsyrk(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          reinterpret_cast<const hipblasComplex*>(arguments.alpha),
          reinterpret_cast<const hipblasComplex*>(arguments.A),
          int(configuration.lda),
          reinterpret_cast<const hipblasComplex*>(arguments.beta),
          reinterpret_cast<hipblasComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
    } else {
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
  } 

  // SYR2K and HER2K
  else if (num_ranks == 2) {
    if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_64F) {
      return hipblasDsyr2k(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const double*>(arguments.alpha),
        static_cast<const double*>(arguments.A),
        int(configuration.lda),
        static_cast<const double*>(arguments.B),
        int(configuration.ldb),
        static_cast<const double*>(arguments.beta),
        static_cast<double*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
        return HIPBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      return hipblasSsyr2k(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const float*>(arguments.alpha),
        static_cast<const float*>(arguments.A),
        int(configuration.lda),
        static_cast<const float*>(arguments.B),
        int(configuration.ldb),
        static_cast<const float*>(arguments.beta),
        static_cast<float*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_64F) {
      
        if (blas_mode == BlasMode::kHermitian) {
          return hipblasZher2k(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
            int(configuration.lda),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.B),
            int(configuration.ldb),
            static_cast<const double*>(arguments.beta),
            reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }    
        else {
          return hipblasZsyr2k(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
            int(configuration.lda),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.B),
            int(configuration.ldb),
            reinterpret_cast<const hipblasDoubleComplex*>(arguments.beta),
            reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }
  
    } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
        return HIPBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      if (blas_mode == BlasMode::kHermitian) {
        return hipblasCher2k(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          reinterpret_cast<const hipblasComplex*>(arguments.alpha),
          reinterpret_cast<const hipblasComplex*>(arguments.A),
          int(configuration.lda),
          reinterpret_cast<const hipblasComplex*>(arguments.B),
          int(configuration.ldb),
          static_cast<const float*>(arguments.beta),
          reinterpret_cast<hipblasComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
      else {
        return hipblasCsyr2k(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          reinterpret_cast<const hipblasComplex*>(arguments.alpha),
          reinterpret_cast<const hipblasComplex*>(arguments.A),
          int(configuration.lda),
          reinterpret_cast<const hipblasComplex*>(arguments.B),
          int(configuration.ldb),
          reinterpret_cast<const hipblasComplex*>(arguments.beta),
          reinterpret_cast<hipblasComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
    } else {
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
  }
  else {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular TRMM description
Status cublas_satisfies(library::TrmmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.D.element == library::NumericTypeID::kS4 || 
    desc.D.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasTrmmDispatcher::cublasTrmmDispatcher(
  library::TrmmDescription const &op_desc,
  library::TrmmConfiguration configuration_,
  library::TrmmArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_side_mode(side, op_desc.side_mode));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_diag_type(diag, op_desc.diag_type));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_cublas_datatype(data_type_D, op_desc.D.element));

  // if A is Transposed, then for cuBLAS that is inverted Fill Mode. 
  if (trans_A == HIPBLAS_OP_T || trans_A == HIPBLAS_OP_C) {
    if (uplo == HIPBLAS_FILL_MODE_LOWER)
      uplo = HIPBLAS_FILL_MODE_UPPER;
    else
      uplo = HIPBLAS_FILL_MODE_LOWER;
  }

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case HIPBLAS_R_32F:
      case HIPBLAS_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case HIPBLAS_R_64F:
      case HIPBLAS_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case HIPBLAS_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case HIPBLAS_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes TRMM using these arguments
hipblasStatus_t cublasTrmmDispatcher::operator()(hipblasHandle_t handle) {
  if (data_type_A == data_type_D && data_type_A == HIPBLAS_R_64F) {
    // TODO: do more check
    // hipblasDtrmm accpet different params than nv-version
    // in hipblasDtrmm，param B will be overwrited as result D
    if (arguments.D==nullptr){
      hipMalloc((void**)&arguments.D, sizeof(double)*configuration.problem_size.m()*configuration.problem_size.n());
    }
    hipMemcpy(arguments.D, arguments.B, sizeof(double)*configuration.problem_size.m()*configuration.problem_size.n(), hipMemcpyDeviceToDevice);
    return hipblasDtrmm(
      handle, side, uplo, trans_A, diag, 
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const double*>(arguments.alpha),
      static_cast<const double*>(arguments.A),
      int(configuration.lda),
      static_cast<double*>(arguments.D),
      int(configuration.ldd)
    );
  } else if (data_type_A == data_type_D && data_type_A == HIPBLAS_R_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
      return HIPBLAS_STATUS_NOT_SUPPORTED; 
#endif

    if (arguments.D==nullptr){
      hipMalloc((void**)&arguments.D, sizeof(float)*configuration.problem_size.m()*configuration.problem_size.n());
    }
    hipMemcpy(arguments.D, arguments.B, sizeof(float)*configuration.problem_size.m()*configuration.problem_size.n(), hipMemcpyDeviceToDevice);
    return hipblasStrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const float*>(arguments.alpha),
      static_cast<const float*>(arguments.A),
      int(configuration.lda),
      static_cast<float*>(arguments.D),
      int(configuration.ldd)
    );


  } else if (data_type_A == data_type_D && data_type_A == HIPBLAS_C_64F) {

    if (arguments.D==nullptr){
      hipMalloc((void**)&arguments.D, sizeof(hipDoubleComplex)*configuration.problem_size.m()*configuration.problem_size.n());
    }
    hipMemcpy(arguments.D, arguments.B, sizeof(hipDoubleComplex)*configuration.problem_size.m()*configuration.problem_size.n(), hipMemcpyDeviceToDevice);
    return hipblasZtrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
      reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
      int(configuration.lda),
      reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
      int(configuration.ldd)
    );

  } else if (data_type_A == data_type_D && data_type_A == HIPBLAS_C_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
      return HIPBLAS_STATUS_NOT_SUPPORTED; 
#endif

    if (arguments.D==nullptr){
      hipMalloc((void**)&arguments.D, sizeof(hipComplex)*configuration.problem_size.m()*configuration.problem_size.n());
    }
    hipMemcpy(arguments.D, arguments.B, sizeof(hipComplex)*configuration.problem_size.m()*configuration.problem_size.n(), hipMemcpyDeviceToDevice);
    return hipblasCtrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      reinterpret_cast<const hipblasComplex*>(arguments.alpha),
      reinterpret_cast<const hipblasComplex*>(arguments.A),
      int(configuration.lda),
      reinterpret_cast<hipblasComplex*>(arguments.D),
      int(configuration.ldd)
    );
  } else {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular Symm description
Status cublas_satisfies(library::SymmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.B.element == library::NumericTypeID::kBF16 || 
    desc.B.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  // only column major layout is supported in cuBLAS
  if (desc.A.layout != library::LayoutTypeID::kColumnMajor || 
      desc.transform_A != library::ComplexTransform::kNone) {
  
    return Status::kErrorNotSupported;
}

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasSymmDispatcher::cublasSymmDispatcher(
  library::SymmDescription const &op_desc,
  library::SymmConfiguration configuration_,
  library::SymmArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  blas_mode = op_desc.blas_mode;

  bool good = true;

  good = (good && get_cublas_side_mode(side, op_desc.side_mode));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case HIPBLAS_R_32F:
      case HIPBLAS_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case HIPBLAS_R_64F:
      case HIPBLAS_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case HIPBLAS_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case HIPBLAS_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes Symm using these arguments
hipblasStatus_t cublasSymmDispatcher::operator()(hipblasHandle_t handle) {
 
  // SYMM and HEMM
  if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_64F) {
    return hipblasDsymm(
      handle,
      side,
      uplo,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const double*>(arguments.alpha),
      static_cast<const double*>(arguments.A),
      int(configuration.lda),
      static_cast<const double*>(arguments.B),
      int(configuration.ldb),
      static_cast<const double*>(arguments.beta),
      static_cast<double*>(arguments.D),
      int(configuration.ldc)
    );
  } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_R_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
      return HIPBLAS_STATUS_NOT_SUPPORTED; 
#endif

    return hipblasSsymm(
      handle,
      side,
      uplo,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const float*>(arguments.alpha),
      static_cast<const float*>(arguments.A),
      int(configuration.lda),
      static_cast<const float*>(arguments.B),
      int(configuration.ldb),
      static_cast<const float*>(arguments.beta),
      static_cast<float*>(arguments.D),
      int(configuration.ldc)
    );
  } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_64F) {
    
      if (blas_mode == BlasMode::kHermitian) {
        return hipblasZhemm(
          handle,
          side,
          uplo,
          configuration.problem_size.m(),
          configuration.problem_size.n(),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
          int(configuration.lda),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.B),
          int(configuration.ldb),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.beta),
          reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }    
      else {
        return hipblasZsymm(
          handle,
          side,
          uplo,
          configuration.problem_size.m(),
          configuration.problem_size.n(),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.alpha),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.A),
          int(configuration.lda),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.B),
          int(configuration.ldb),
          reinterpret_cast<const hipblasDoubleComplex*>(arguments.beta),
          reinterpret_cast<hipblasDoubleComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }

  } else if (data_type_A == data_type_C && data_type_A == HIPBLAS_C_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != HIPBLAS_STATUS_SUCCESS)
      return HIPBLAS_STATUS_NOT_SUPPORTED; 
#endif

    if (blas_mode == BlasMode::kHermitian) {
      return hipblasChemm(
        handle,
        side,
        uplo,
        configuration.problem_size.m(),
        configuration.problem_size.n(),
        reinterpret_cast<const hipblasComplex*>(arguments.alpha),
        reinterpret_cast<const hipblasComplex*>(arguments.A),
        int(configuration.lda),
        reinterpret_cast<const hipblasComplex*>(arguments.B),
        int(configuration.ldb),
        reinterpret_cast<const hipblasComplex*>(arguments.beta),
        reinterpret_cast<hipblasComplex*>(arguments.D),
        int(configuration.ldc)
      );
    }
    else {
      return hipblasCsymm(
        handle,
        side,
        uplo,
        configuration.problem_size.m(),
        configuration.problem_size.n(),
        reinterpret_cast<const hipblasComplex*>(arguments.alpha),
        reinterpret_cast<const hipblasComplex*>(arguments.A),
        int(configuration.lda),
        reinterpret_cast<const hipblasComplex*>(arguments.B),
        int(configuration.ldb),
        reinterpret_cast<const hipblasComplex*>(arguments.beta),
        reinterpret_cast<hipblasComplex*>(arguments.D),
        int(configuration.ldc)
      );
    }
  } else {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

#endif // #if CUTLASS_ENABLE_CUBLAS
