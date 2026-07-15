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
   \brief Helper functions for mapping CUTLASS concepts to cuDNN.

*/

#pragma once
#if CUTLASS_ENABLE_CUDNN
#include <hip/hip_runtime.h>
#include <hipDNN.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/library/library.h"
#include "enumerated_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Converts a cuDNN status to cutlass::Status
Status get_cutlass_status(hipdnnStatus_t cudnn_status);

/// Converts a cuDNN status to cutlass::profiler::Disposition
Disposition get_cutlass_disposition(hipdnnStatus_t cudnn_status);

/// Checks hipdnnStatus_t converts to cutlas status and returns if Status::kSuccess o.w. throws exception
Status checkCudnnErr(hipdnnStatus_t cudnn_status);

/// Maps a CUTLASS conv mode to a cuDNN conv mode enumeration
bool get_cudnn_conv_mode(hipdnnConvolutionMode_t &cudnn_conv_mode, conv::Mode conv_mode);

/// Maps a CUTLASS layout type to a cuDNN data type enumeration
bool get_cudnn_layout(hipdnnTensorFormat_t &cudnn_layout, library::LayoutTypeID layout);

/// Maps a CUTLASS numeric type to a cuDNN data type enumeration
bool get_cudnn_datatype(hipdnnDataType_t &cudnn_element_type, library::NumericTypeID element_type);

/// Maps CUTLASS math OpcodeClassID and MathOperationID to cuDNN math_type
bool get_cudnn_mathtype(hipdnnMathType_t &cudnn_math_type, library::ConvDescription const &conv_desc);

/// Returns a status if cudnn can satisfy a particular Conv2d description
Status cudnn_satisfies(library::ConvDescription const &desc, library::Conv2dConfiguration const &configuration);

/// Returns a status if cudnn can satisfy a particular Conv3d description
Status cudnn_satisfies(library::ConvDescription const &desc, library::Conv3dConfiguration const &configuration);

/// Cudnn compute type seems to be hardcoded to float (To handle a possible cudnn issue)
float cast_cudnn_compute_type_to_float(library::NumericTypeID type, void const * src);


/// This is a helper class to create hipdnnHandle_t automatically on CudnnCreate object creation and 
/// to destroy hipdnnHandle_t on CudnnCreate object destruction. 
/// Additionally, it provides implicit cast from CudnnCreate's object to hipdnnHandle_t's object
class CudnnCreate {
private:
	hipdnnHandle_t handle;
	hipdnnStatus_t status;

public:
	CudnnCreate() {
		status = hipdnnCreate(&handle);
	}

	~CudnnCreate() {
		hipdnnDestroy(handle);
	}

    /// Implicit cast CudnnCreate object to hipdnnHandle_t
    operator hipdnnHandle_t() const { return handle; }

    /// returns hipdnnStatus_t for handle creation
    hipdnnStatus_t get_cudnn_create_status() { return status; }
};


namespace detail {

/// Dispatcher to cudnn convolution operators
struct cudnnConvDispatcher {

  //
  // Data members
  //
  //library::Conv2dConfiguration configuration;
  library::ConvArguments arguments;
  library::ConvKind conv_kind;

  // cudnn-specific data structures to fill cudnn API call arguments
  // cudnn activation, filter, and output descriptors
  hipdnnTensorDescriptor_t activation_desc;
  hipdnnFilterDescriptor_t filter_desc;
  hipdnnTensorDescriptor_t output_desc;
  hipdnnConvolutionDescriptor_t conv_desc;

  // cudnn datatypes
  hipdnnDataType_t data_type_activation;
  hipdnnDataType_t data_type_filter;
  hipdnnDataType_t data_type_output;

  // cudnn layouts
  hipdnnTensorFormat_t layout_activation;
  hipdnnTensorFormat_t layout_filter;
  hipdnnTensorFormat_t layout_output;

  // cudnn convolution mode
  hipdnnConvolutionMode_t conv_mode;
  
  // cudnn math type (tensorop, tensorop with conversion, simt)
  hipdnnMathType_t math_type;

  // cudnn compute data type
  hipdnnDataType_t compute_type;
  
  // cudnn compute type seems to be hardcoded to float (to handle a possible a cudnn issue)
  float alpha;
  float beta;

  // cudnn workspace
  size_t workspace_size_in_bytes = 0;
  cutlass::device_memory::allocation<char> workspace;
  
  // select cudnn's implicit gemm precomputed algorithm with tensor operations
  static hipdnnConvolutionFwdAlgo_t const fprop_algo = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static hipdnnConvolutionBwdDataAlgo_t const dgrad_algo = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static hipdnnConvolutionBwdFilterAlgo_t const wgrad_algo = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  Status status;
  
  //
  // Methods
  //

  // TODO: unify ctor cudnnConvDispatcher for conv2d and conv3d by unifying Conv2dConfiguration
  
  // ctor for conv2d 
  cudnnConvDispatcher( 
    library::ConvDescription const &op_desc,
    library::Conv2dConfiguration configuration,
    library::ConvArguments arguments_,
    hipdnnHandle_t handle
  ):
    //configuration(configuration_), 
    arguments(arguments_),
    conv_kind(op_desc.conv_kind), 
    status(Status::kSuccess) {

    bool good = true;

    // Get cudnn datatype, layout, and convolution mode from library::ConvDescription
    good = (good && get_cudnn_datatype(data_type_activation, op_desc.A.element));
    good = (good && get_cudnn_datatype(data_type_filter, op_desc.B.element));
    good = (good && get_cudnn_datatype(data_type_output, op_desc.C.element));
    good = (good && get_cudnn_layout(layout_activation, op_desc.A.layout));
    good = (good && get_cudnn_layout(layout_filter, op_desc.B.layout));
    good = (good && get_cudnn_layout(layout_output, op_desc.C.layout));
    good = (good && get_cudnn_conv_mode(conv_mode, configuration.problem_size.mode));
    // Get cudnn mathtype (hipdnnMathType_t)
    good = (good && get_cudnn_mathtype(math_type, op_desc));
    good = (good && get_cudnn_datatype(
      compute_type,
      op_desc.tile_description.math_instruction.element_accumulator));
    // Check cutlass Conv2d description has equivalent operator in cudnn
    if (!good) {
      status = Status::kErrorNotSupported;
      return;
    }
    // cudnn compute type seems to be hardcoded to float (to handle a possible a cudnn issue)
    alpha = cast_cudnn_compute_type_to_float(op_desc.element_epilogue, arguments.alpha);
    beta = cast_cudnn_compute_type_to_float(op_desc.element_epilogue, arguments.beta);

    // Create convolution descriptor object
    status = get_cutlass_status(hipdnnCreateConvolutionDescriptor(&conv_desc));

    // Configure convolution operator
    std::vector<int> padding {configuration.problem_size.pad_h, configuration.problem_size.pad_w};
    std::vector<int> stride {configuration.problem_size.stride_h, configuration.problem_size.stride_w};
    std::vector<int> dilation {configuration.problem_size.dilation_h, configuration.problem_size.dilation_w};

    status = get_cutlass_status(
      hipdnnSetConvolutionNdDescriptor(
        conv_desc,
        op_desc.conv_dim,
        padding.data(),
        stride.data(),
        dilation.data(),
        conv_mode,
        compute_type
    ));

    // Set groups
    status = get_cutlass_status(hipdnnSetConvolutionGroupCount(conv_desc, configuration.problem_size.groups));

    // Create activation, filter, and output descriptor objects
    status = get_cutlass_status(hipdnnCreateTensorDescriptor(&activation_desc));
    status = get_cutlass_status(hipdnnCreateFilterDescriptor(&filter_desc));
    status = get_cutlass_status(hipdnnCreateTensorDescriptor(&output_desc));

    // Set activation, filter, and output descriptor 
    status = get_cutlass_status(
      hipdnnSetTensor4dDescriptor(
        activation_desc,
        layout_activation,
        data_type_activation,
        configuration.problem_size.N,
        configuration.problem_size.C,
        configuration.problem_size.H,
        configuration.problem_size.W 
    ));

    status = get_cutlass_status(
      hipdnnSetFilter4dDescriptor(
        filter_desc,
        data_type_filter,
        layout_filter,
        configuration.problem_size.K,
        configuration.problem_size.C / configuration.problem_size.groups,
        configuration.problem_size.R,
        configuration.problem_size.S
    ));

    status = get_cutlass_status(
      hipdnnSetTensor4dDescriptor(
        output_desc,
        layout_output,
        data_type_output,
        configuration.problem_size.N,
        configuration.problem_size.K,
        configuration.problem_size.P,
        configuration.problem_size.Q
    ));

    // Set math instruction to tensor op
    status = get_cutlass_status(
      hipdnnSetConvolutionMathType(conv_desc, math_type));

    // Initialize workspace
    switch (conv_kind) {
      case library::ConvKind::kFprop:
        status =  get_cutlass_status(
          hipdnnGetConvolutionForwardWorkspaceSize(
            handle,
            activation_desc,
            filter_desc,
            conv_desc,
            output_desc,
            fprop_algo,
            &workspace_size_in_bytes
        )); break;
      case library::ConvKind::kDgrad:
        status =  get_cutlass_status(
          hipdnnGetConvolutionBackwardDataWorkspaceSize(
            handle,
            filter_desc,
            output_desc,
            conv_desc,
            activation_desc,
            dgrad_algo,
            &workspace_size_in_bytes
        )); break;
        case library::ConvKind::kWgrad:
        status =  get_cutlass_status(
          hipdnnGetConvolutionBackwardFilterWorkspaceSize(
            handle,
            activation_desc,
            output_desc,
            conv_desc,
            filter_desc,
            wgrad_algo,
            &workspace_size_in_bytes
        )); break;

    }

    workspace = cutlass::device_memory::allocation<char>(workspace_size_in_bytes);
  }


  // ctor for conv3d 
  cudnnConvDispatcher( 
    library::ConvDescription const &op_desc,
    library::Conv3dConfiguration configuration,
    library::ConvArguments arguments_,
    hipdnnHandle_t handle
  ):
    //configuration(configuration_), 
    arguments(arguments_),
    conv_kind(op_desc.conv_kind), 
    status(Status::kSuccess) {

    bool good = true;

    // Get cudnn datatype, layout, and convolution mode from library::ConvDescription
    good = (good && get_cudnn_datatype(data_type_activation, op_desc.A.element));
    good = (good && get_cudnn_datatype(data_type_filter, op_desc.B.element));
    good = (good && get_cudnn_datatype(data_type_output, op_desc.C.element));

    good = (good && get_cudnn_layout(layout_activation, op_desc.A.layout));
    good = (good && get_cudnn_layout(layout_filter, op_desc.B.layout));
    good = (good && get_cudnn_layout(layout_output, op_desc.C.layout));

    good = (good && get_cudnn_conv_mode(conv_mode, configuration.problem_size.mode));
    
    // cudnn compute type seems to be hardcoded to float (to handle a possible a cudnn issue)
    alpha = cast_cudnn_compute_type_to_float(op_desc.element_epilogue, arguments.alpha);
    beta = cast_cudnn_compute_type_to_float(op_desc.element_epilogue, arguments.beta);

    good = (good && get_cudnn_datatype(
      compute_type, 
      op_desc.tile_description.math_instruction.element_accumulator));

    // Check cutlass Conv2d description has equivalent operator in cudnn
    if (!good) {
      status = Status::kErrorNotSupported;
    }

    // Create convolution descriptor object
    status = get_cutlass_status(hipdnnCreateConvolutionDescriptor(&conv_desc));

    // Configure convolution operator
    std::vector<int> padding {configuration.problem_size.pad_d, configuration.problem_size.pad_h, configuration.problem_size.pad_w};
    std::vector<int> stride {configuration.problem_size.stride_d, configuration.problem_size.stride_h, configuration.problem_size.stride_w};
    std::vector<int> dilation {configuration.problem_size.dilation_d, configuration.problem_size.dilation_h, configuration.problem_size.dilation_w};

    status = get_cutlass_status(
      hipdnnSetConvolutionNdDescriptor(
        conv_desc,
        op_desc.conv_dim,
        padding.data(),
        stride.data(),
        dilation.data(),
        conv_mode,
        compute_type
    ));

    // Set groups
    status = get_cutlass_status(hipdnnSetConvolutionGroupCount(conv_desc, configuration.problem_size.groups));

    // Create activation, filter, and output descriptor objects
    status = get_cutlass_status(hipdnnCreateTensorDescriptor(&activation_desc));
    status = get_cutlass_status(hipdnnCreateFilterDescriptor(&filter_desc));
    status = get_cutlass_status(hipdnnCreateTensorDescriptor(&output_desc));

    // Set activation descriptor 
    std::vector<int> activation_extent {
      configuration.problem_size.N,
      configuration.problem_size.C,
      configuration.problem_size.D,
      configuration.problem_size.H,
      configuration.problem_size.W
    };

    std::vector<int> activation_stride {
      configuration.layout_activations.stride()[3],
      1,
      configuration.layout_activations.stride()[2],
      configuration.layout_activations.stride()[1],
      configuration.layout_activations.stride()[0]
    };

    status = get_cutlass_status(
      hipdnnSetTensorNdDescriptor(
        activation_desc,
        data_type_activation,
        op_desc.conv_dim + 2,
        activation_extent.data(),
        activation_stride.data()        
    ));

    // Set filter descriptor
    std::vector<int> filter_extent {
      configuration.problem_size.K,
      configuration.problem_size.C,
      configuration.problem_size.T,
      configuration.problem_size.R,
      configuration.problem_size.S
    };

    std::vector<int> filter_stride {
      configuration.layout_filters.stride()[3],
      1,
      configuration.layout_filters.stride()[2],
      configuration.layout_filters.stride()[1],
      configuration.layout_filters.stride()[0]
    };

    status = get_cutlass_status(
      hipdnnSetFilterNdDescriptor(
        filter_desc,
        data_type_filter,
        layout_filter,
        op_desc.conv_dim + 2,
        filter_extent.data() 
    ));


    // Set output descriptor
    std::vector<int> output_extent {
      configuration.problem_size.N,
      configuration.problem_size.K,
      configuration.problem_size.Z,
      configuration.problem_size.P,
      configuration.problem_size.Q
    };

    std::vector<int> output_stride {
      configuration.layout_output.stride()[3],
      1,
      configuration.layout_output.stride()[2],
      configuration.layout_output.stride()[1],
      configuration.layout_output.stride()[0]
    };

    status = get_cutlass_status(
      hipdnnSetTensorNdDescriptor(
        output_desc,
        data_type_output,
        op_desc.conv_dim + 2,
        output_extent.data(),
        output_stride.data() 
    ));

    // Set math instruction to tensor op
    status = get_cutlass_status(
      hipdnnSetConvolutionMathType(conv_desc, math_type));

    // Initialize workspace
    switch (conv_kind) {
      case library::ConvKind::kFprop:
        status =  get_cutlass_status(
          hipdnnGetConvolutionForwardWorkspaceSize(
            handle,
            activation_desc,
            filter_desc,
            conv_desc,
            output_desc,
            fprop_algo,
            &workspace_size_in_bytes
        )); break;
      case library::ConvKind::kDgrad:
        status =  get_cutlass_status(
          hipdnnGetConvolutionBackwardDataWorkspaceSize(
            handle,
            filter_desc,
            output_desc,
            conv_desc,
            activation_desc,
            dgrad_algo,
            &workspace_size_in_bytes
        )); break;
        case library::ConvKind::kWgrad:
        status =  get_cutlass_status(
          hipdnnGetConvolutionBackwardFilterWorkspaceSize(
            handle,
            activation_desc,
            output_desc,
            conv_desc,
            filter_desc,
            wgrad_algo,
            &workspace_size_in_bytes
        )); break;

    }

    workspace = cutlass::device_memory::allocation<char>(workspace_size_in_bytes);
  }

  /// Executes Conv2d operator from cudnn library
  hipdnnStatus_t operator()(hipdnnHandle_t handle) {

    switch (conv_kind) {
      case library::ConvKind::kFprop:
        return hipdnnConvolutionForward(
          handle,
          &alpha,
          activation_desc,
          activation(),
          filter_desc,
          filter(),
          conv_desc,
          fprop_algo,
          workspace.get(),
          workspace_size_in_bytes,
          &beta,
          output_desc,
          arguments.D
        );
      case library::ConvKind::kDgrad:
        return hipdnnConvolutionBackwardData(
          handle,
          &alpha,
          filter_desc,
          filter(),
          output_desc,
          output(),
          conv_desc,
          dgrad_algo,
          workspace.get(),
          workspace_size_in_bytes,
          &beta,
          activation_desc,
          arguments.D
        );
      case library::ConvKind::kWgrad:
        return hipdnnConvolutionBackwardFilter(
          handle,
          &alpha,
          activation_desc,
          activation(),
          output_desc,
          output(),
          conv_desc,
          wgrad_algo,
          workspace.get(),
          workspace_size_in_bytes,
          &beta,
          filter_desc,
          arguments.D
        );
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Activation Tensor
  void const * activation() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return arguments.A;
      case library::ConvKind::kDgrad : return arguments.C;
      case library::ConvKind::kWgrad : return arguments.B;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Filter Tensor
  void const *filter() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return arguments.B;
      case library::ConvKind::kDgrad : return arguments.B;
      case library::ConvKind::kWgrad : return arguments.C;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Output Tensor
  void const *output() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return arguments.C;
      case library::ConvKind::kDgrad : return arguments.A;
      case library::ConvKind::kWgrad : return arguments.A;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }
};

} // namespace detail
/////////////////////////////////////////////////////////////////////////////////////////////////
#endif //#if CUTLASS_ENABLE_CUDNN
} // namespace profiler
} // namespace cutlass
