#include "hip/hip_runtime.h"
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

// #include "cutlass_unit_test.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cute/atom/copy_traits_sm75.hpp>

#include <cute/atom/copy_traits_gfx928.hpp>
#include <cutlass/numeric_conversion.h>

using namespace cute;

// #define GFX928_DS_READ_DS_M32x8_B32_TEST
#define GFX928_DS_READ_DS_M32x16_B16_ALT_TEST
// #define GFX928_DS_READ_DS_M32x16_B16_TEST

template <class T>
__global__ void
ldsm_test_device(T* g_in, T* g_out)
{
  constexpr int count = sizeof(T) / 4;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  // load input gmem -> smem
  __shared__ uint32_t smem[64 * count];
  for (int i = 0; i < count; ++i) {
    smem[tid + (stride * i)] = reinterpret_cast<uint32_t*>(g_in)[tid + (stride * i)];
  }

  __syncthreads();

  uint32_t reg[count];
  for (int i = 0; i < count; ++i) {
    reg[i] = 0;
  }
  uint32_t*  rmem_ptr = reinterpret_cast<uint32_t*>(reg);
  // int offset = tid * sizeof(uint128_t);
  // cute::GFX928_DS_READ_F32_TEST::copy(offset, rmem_ptr[0],rmem_ptr[1],rmem_ptr[2],rmem_ptr[3]);
  // store output rmem -> gmem
  for (int i = 0; i < count; ++i) {
    reinterpret_cast<uint32_t*>(g_out)[tid * 4 + i] = reg[i];
  }

}



template <class T, class TiledCopy, class SmemLayout>
__global__ void
ldsm_test_device_cute(T* g_in, T* g_out,
                      TiledCopy tiled_copy, SmemLayout smem_layout)
{

  using namespace cute;

  __shared__ T smem[size(smem_layout)];

  auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
  auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
  auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

  int tid = threadIdx.x;
  // Load input gmem -> smem
  for (int i = tid; i < size(t_smem); i += size(tiled_copy)) {
    t_smem(i) = t_g_in(i);
  }

  __syncthreads();
  
  // if(thread0())
  // {
  //   // print_tensor(t_g_in);
  //   print_tensor(t_smem);
  //   // print_tensor(t_g_out);
  // }

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  auto tXsX = thr_copy.partition_S(t_smem);   // (V,M,N)
  auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)

  auto tXrX = make_tensor<T>(layout(tXgX)); // (V,M,N)
  clear(tXrX);  // Just to make sure
  // if (thread0()) {
  //   print("thr_copy: ");
  //   print(thr_copy);
  //   print("\n");
  //   // print("tXsX: " ); print(tXsX.layout()); print("\n");
  //   // print("tXgX: " ); print(tXgX.layout()); print("\n");
  //   // print("tXrX: " ); print(tXrX.layout()); print("\n");
  // }
  // printf("threadIdx.x:%d  shared:%p\n",)
/*
  if (thread0()) {
    print("tXsX: " ); print(tXsX.layout()); print("\n");
    print("tXgX: " ); print(tXgX.layout()); print("\n");
    print("tXrX: " ); print(tXrX.layout()); print("\n");
  }
*/

  // // Copy smem -> rmem via tiled_copy (LDSM, LDS)

  copy(tiled_copy, tXsX, tXrX);
  // if (thread0())
  // {
  //   print_tensor(tXsX);
  //   print_tensor(tXrX);
  // }

  // // Output rmem -> gmem
  copy(tXrX, tXgX);

}

int main()
{
  //
  //  float ds_read
  //

  // {
  //   constexpr int count = 1024;

  //   thrust::host_vector<uint32_t> h_in(count);
  //   for (int i = 0; i < count; ++i) {
  //     h_in[i] = uint32_t(i);
  //   }
  //   thrust::device_vector<uint32_t> d_in = h_in;
  //   thrust::device_vector<uint32_t> d_out(count);
  //   ldsm_test_device<uint128_t><<<1, 64>>>(
  //     thrust::raw_pointer_cast(d_in.data()),
  //     thrust::raw_pointer_cast(d_out.data()));
  //   thrust::host_vector<uint32_t> h_out = d_out;
  //   for (int i = 0; i < 128; ++i) {
  //     printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
  //   }
  // }

  //
  //  half_t ds_read
  //
  // {
  //   constexpr int count = 1024;

  //   thrust::host_vector<uint16_t> h_in(count);
  //   for (int i = 0; i < count; ++i) {
  //     h_in[i] = uint16_t(i);
  //   }
  //   thrust::device_vector<uint16_t> d_in = h_in;
  //   thrust::device_vector<uint16_t> d_out(count);
  //   ldsm_test_device<uint128_t><<<1, 64>>>(
  //     thrust::raw_pointer_cast(d_in.data()),
  //     thrust::raw_pointer_cast(d_out.data()));
  //   thrust::host_vector<uint16_t> h_out = d_out;
  //   for (int i = 0; i < 128; ++i) {
  //     printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
  //   }
  // }
 
  //
  // CuTe LDSM
  //
  #ifdef GFX928_DS_READ_DS_M32x16_B16_ALT_TEST
  {
    constexpr int count = 1024;
    thrust::host_vector<uint16_t> h_in(count);
    for (int i = 0; i < count; ++i) {
      h_in[i] = uint16_t(i);
    }
    thrust::device_vector<uint16_t> d_in = h_in;
    thrust::device_vector<uint16_t> d_out(count);

    auto smem_layout = Layout<Shape <_32,_16>,
                              Stride< _1,_32>>{};

    auto tiled_copy = make_tiled_copy(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_ALT, uint16_t>{},
                                      Layout<Shape<_16,_4>>{},
                                      Layout<Shape< _2,_4>>{});

    ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      tiled_copy,
      smem_layout);
    thrust::host_vector<uint16_t> h_out = d_out;
    for (int i = 0; i < size(smem_layout); ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      // EXPECT_EQ(h_out[i], h_in[i]);
    }
    // CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x1_LDSM_N SUCCESS\n");
  }
  #endif

  #ifdef GFX928_DS_READ_DS_M32x16_B16_TEST
  {
    constexpr int count = 1024;
    thrust::host_vector<uint16_t> h_in(count);
    for (int i = 0; i < count; ++i) {
      h_in[i] = uint16_t(i);
    }
    thrust::device_vector<uint16_t> d_in = h_in;
    thrust::device_vector<uint16_t> d_out(count);

    auto smem_layout = Layout<Shape <_32,_16>,
                              Stride< _1,_32>>{};

 

    TiledMMA mma = make_tiled_mma(GFX928_32x32x16_F32F16F16F32_NT{},
                            Layout<Shape<_1,_1,_1>>{},  // Layout in Thr
                            Layout<Shape<_1,_1,_1>>{});   // Layout in Val

    auto tiled_copy = make_tiled_copy_A(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, uint16_t>{},
                                 mma);
    #if 1
        print_layout(smem_layout);
        print(tiled_copy);
        using test_copy = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, uint16_t>;
        print_layout(test_copy::ValLayoutSrc{});
        print_layout(test_copy::ValLayoutDst{});
        printf("block size:%d\n",int(size(tiled_copy)));
    #endif

    ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      tiled_copy,
      smem_layout);
    thrust::host_vector<uint16_t> h_out = d_out;
    for (int i = 0; i < size(smem_layout); ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      // EXPECT_EQ(h_out[i], h_in[i]);
    }
    // CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x1_LDSM_N SUCCESS\n");
  }
  #endif

  
  #ifdef GFX928_DS_READ_DS_M32x8_B32_TEST
  {
    constexpr int count = 1024;
    thrust::host_vector<uint32_t> h_in(count);
    for (int i = 0; i < count; ++i) {
      h_in[i] = uint32_t(i);
    }
    thrust::device_vector<uint32_t> d_in = h_in;
    thrust::device_vector<uint32_t> d_out(count);

    auto smem_layout = Layout<Shape <_32,_8>,
                              Stride< _1,_32>>{};

    // auto tiled_copy = make_tiled_copy(Copy_Atom<GFX928_DS_READ_DS_M32x8_B32, uint32_t>{},
    //                                   Layout<Shape<_32,_2>>{},
    //                                   Layout<Shape<_1,_4>>{});
    auto tiled_mma = make_tiled_mma(GFX928_16x16x8_F32F32F32F32_NT{},
                                    Layout<Shape<_1,_1,_1>>{},
                                    Layout<Shape<_2,_2,_1>>{});      
    auto tiled_copy = make_tiled_copy_A(Copy_Atom<GFX928_DS_READ_DS_M32x8_B32, uint32_t>{},
                                      tiled_mma);
    #if 0
      print_layout(smem_layout);
      print(tiled_copy);
      using test_copy = Copy_Atom<GFX928_DS_READ_DS_M32x8_B32, uint32_t>;
      print_layout(test_copy::ValLayoutSrc{});
      print_layout(test_copy::ValLayoutDst{});
      printf("block size:%d\n",int(size(tiled_copy)));
    #endif

    ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      tiled_copy,
      smem_layout);
    thrust::host_vector<uint32_t> h_out = d_out;
    for (int i = 0; i < size(smem_layout); ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    }
  }
  #endif
}

