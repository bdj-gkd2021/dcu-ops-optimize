# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

message(STATUS "Configuring hipblas ...")

if((DEFINED CUTLASS_ENABLE_CUBLAS AND NOT CUTLASS_ENABLE_CUBLAS) OR
   (DEFINED CUBLAS_ENABLED AND NOT CUBLAS_ENABLED))
  
  # Don't add cuBLAS if it's defined and false, assume it's not found.

  set(CUBLAS_FOUND OFF)
  message(STATUS "hipBLAS Disabled.")

elseif(NOT TARGET hipblas)
 
  find_path(
    _HIPBLAS_INCLUDE_DIR
    NAMES hipblas.h
    HINTS
      ${HIP_TOOLKIT_ROOT_DIR}/hipblas
    PATH_SUFFIXES
      include
    )

  find_library(
    _HIPBLAS_LIBRARY
    NAMES hipblas
    HINTS
      ${HIP_TOOLKIT_ROOT_DIR}/hipblas
    PATH_SUFFIXES
      lib64
      lib/x64
      lib
    )

  if(_HIPBLAS_INCLUDE_DIR AND _HIPBLAS_LIBRARY)

    message(STATUS "hipBLAS: ${_HIPBLAS_LIBRARY}")
    message(STATUS "hipBLAS: ${_HIPBLAS_INCLUDE_DIR}")
    
    set(CUBLAS_FOUND ON CACHE INTERNAL "hipblas Library Found")
    set(CUBLAS_LIBRARY ${_HIPBLAS_LIBRARY})
    set(CUBLAS_INCLUDE_DIR ${_HIPBLAS_INCLUDE_DIR})

  else()

    message(STATUS "hipblas not found.")
    set(CUBLAS_FOUND OFF CACHE INTERNAL "hipblas Library Found")

  endif()

endif()

set(CUTLASS_ENABLE_CUBLAS ${CUBLAS_FOUND} CACHE BOOL "Enable CUTLASS to build with cuBLAS library.")

if(CUTLASS_ENABLE_CUBLAS AND NOT CUBLAS_FOUND)
  message(FATAL_ERROR "CUTLASS_ENABLE_CUBLAS enabled but cuBLAS library could not be found.")
endif()

if(CUTLASS_ENABLE_CUBLAS AND NOT TARGET hipblas)

  if(WIN32)
    add_library(cublas STATIC IMPORTED GLOBAL)
  else()
    add_library(hipblas SHARED IMPORTED GLOBAL)
  endif()

  add_library(hip::hipblas ALIAS hipblas)

  set_property(
    TARGET hipblas
    PROPERTY IMPORTED_LOCATION
    ${CUBLAS_LIBRARY})
    
  target_include_directories(
    hipblas
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUBLAS_INCLUDE_DIR}>)

  find_library(
    _HIPBLASLT_LIBRARY
    NAMES hipblasLt
    HINTS
      ${HIP_TOOLKIT_ROOT_DIR}/hipblaslt
    PATH_SUFFIXES
      lib64
      lib/x64
      lib
    )

  if(_HIPBLASLT_LIBRARY AND NOT TARGET hipblasLt)

    add_library(hipblasLt SHARED IMPORTED GLOBAL)

    set_property(
      TARGET hipblasLt
      PROPERTY IMPORTED_LOCATION
      ${_CUBLASLT_LIBRARY})
  
    add_library(hip::hipblasLt ALIAS hipblasLt)

    target_link_libraries(hipblas INTERFACE hipblasLt)

  endif()

endif()

message(STATUS "Configuring hipBLAS ... done.")
