#include "hip/hip_runtime.h"
/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cute/layout.hpp"
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// #include "data.h"

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif
template <typename T>
void cpu_compare(const T &x, const T &y, int threshold) {
  using namespace cute;

  if (size(x) != size(y)) {
    fprintf(stderr, "lenght not equal x = %d, y = %d\n", size(x), size(y));
    exit(9);
  }

  int n = size(x);
  int diff_max = 0;
  int diff_count = 0;
  for (int i = 0; i < n; ++i) {
    int v0 = x(i);
    int v1 = y(i);
    // printf("[%d]: %f %f\n",i,v0,v1);
    diff_max = max(diff_max, abs(v0 - v1));

    if (abs(v0 - v1) > threshold) {
        printf("[%d]: %d %d\n",i,v0,v1);
        ++diff_count;
    }
  }
  if (diff_count > 0) {
    printf("check fail: max_diff = %d, diff_count = %d\n", diff_max,
           diff_count);
  } else {
    printf("cpu check ok\n");
  }
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;
  using X = Underscore;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrA) ==   shape(tCsA));                // (MMA,MMA_M,MMA_K)
  CUTE_STATIC_ASSERT_V(  shape(tCrB) ==   shape(tCsB));                // (MMA,MMA_N,MMA_K)
  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // Copy rmem to smem
  copy(tArA, tAsA);
  copy(tBrB, tBsB);
  __syncthreads();

  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory AND register memory
  //   Data is read from global to registers, then to shared via the tA|tB partitions
  //   Data is then copied from shared to registers in multiple waves via the tC partitions
  //     and gemm(.) operates on the current register wave
  //

  // Load A, B shmem->regs for k_block=0
  copy(tCsA(_,_,0), tCrA(_,_,0));
  copy(tCsB(_,_,0), tCrB(_,_,0));
  auto K_TILE_MAX  = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Pipeline the k-mode of the block registers
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Copy rmem to smem
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
      }

      // Copy smem to rmem for k_block+1
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(tCsA(_,_,k_block_next), tCrA(_,_,k_block_next));
      copy(tCsB(_,_,k_block_next), tCrB(_,_,k_block_next));
      if (k_block == 0)
      {
        // Copy gmem to rmem for k_tile+1
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
      }

      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    } // k_block
  } // k_tile

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        hipStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<32>{};
  auto bN = Int<32>{};
  auto bK = Int<32>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_16,_16>>{},  // Thr layout 16x16 m-major
                                    Layout<Shape< _1,_1>>{}); // Val layout  1x1 m-major
  
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TB>{},
                                    Layout<Shape<_16,_16>>{},  // Thr layout 16x16 n-major
                                    Layout<Shape< _1,_1>>{}); // Val layout  1x1 n-major

  TiledMMA mmaC = make_tiled_mma(GFX928_16x16x32_I32U8U8I32_NT{},
                                  Layout<Shape<_2,_2,_1>>{},  // Layout in Thr
                                  Layout<Shape<_1,_1,_1>>{});   // Layout in Val
  
#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        hipStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<32>{};
  auto bN = Int<32>{};
  auto bK = Int<32>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));        // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));        // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx



  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_16,_16>>{}, // Thr layout 32x16 m-major
                                    Layout<Shape< _1,_1>>{}); // Val layout  1x1 m-major
  
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TB>{},
                                    Layout<Shape<_16,_16>>{}, // Thr layout 32x16 n-major
                                    Layout<Shape< _1,_1>>{}); // Val layout  1x1 n-major

  TiledMMA mmaC = make_tiled_mma(GFX928_16x16x32_I32U8U8I32_NT{},
                                  Layout<Shape<_2,_2,_1>>{},  // Layout in Thr
                                  Layout<Shape<_1,_1,_1>>{}); // Layout in Val
#if 0
  print(copyA);
  print(copyB);
  print(mmaC);

#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);

}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     hipStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}


int main(int argc, char** argv)
{
  hipDeviceProp_t props;
  hipError_t error = hipGetDeviceProperties(&props, 0);
  if (error != hipSuccess) {
    std::cerr << "hipGetDeviceProperties() returned an error: " << hipGetErrorString(error) << std::endl;
    return -1;
  }

  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'N';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'T';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = cute::uint8_t;
  using TB = cute::uint8_t;
  using TC = int;
  using TI = int;

  TI alpha = TI(1.0);
  TI beta  = TI(0.0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( rand() % 127);
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( rand() % 127);
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);
  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 1;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //

  hipblasHandle_t handle;
  hipblasCreate(&handle);

  // Run once
  d_C = h_C;
  blam::cublas::gemm(handle,
                     transA == 'N' ? HIPBLAS_OP_N :HIPBLAS_OP_T, 
                     transB == 'N' ? HIPBLAS_OP_N :HIPBLAS_OP_T,
                     m, n, k,
                     &alpha,
                     d_A.data().get(), ldA,
                     d_B.data().get(), ldB,
                     &beta,
                     d_C.data().get(), m);
  CUTE_CHECK_LAST();

  thrust::host_vector<TC> cublas_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    blam::cublas::gemm(handle,
                       transA == 'N' ? HIPBLAS_OP_N :HIPBLAS_OP_T, 
                       transB == 'N' ? HIPBLAS_OP_N :HIPBLAS_OP_T,
                       m, n, k,
                       &alpha,
                       d_A.data().get(), ldA,
                       d_B.data().get(), ldB,
                       &beta,
                       d_C.data().get(), m);
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);

#else

  std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
    "If you would like to enable verification with cuBLAS, "
    "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
    "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS

  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;
  if(transA == 'N' && transB == 'T')
  {
    using namespace cute;
    TI * Dptr_host_cpu = (TI *)malloc(sizeof(TI) * m * n);
    TI * Dptr_host = (TI *)malloc(sizeof(TI) * m * n);
    TA * Dptr_host_A = &h_A[0];
    TB * Dptr_host_B = &h_B[0];
    auto tD_host_cpu =
      make_tensor(Dptr_host_cpu, make_shape(m, n), make_stride(1, m));
    auto tA = make_tensor(Dptr_host_A, make_shape(m, k), make_stride(1, m));
    auto tB = make_tensor(Dptr_host_B, make_shape(n, k), make_stride(1, n));


    int m = size<0>(tA);
    int n = size<0>(tB);
    int k = size<1>(tA);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int s = 0;

        for (int kk = 0; kk < k; ++kk) {
          TA v1 = tA(i, kk);
          TB v2 = tB(j, kk);
          s += (int)(v1 * v2);
        }
        tD_host_cpu(i, j) = s;
      }
    }

    hipMemcpy(Dptr_host, d_C.data().get(), sizeof(TI) * m * n, hipMemcpyDeviceToHost);
    auto tD_host_cute = make_tensor(Dptr_host, make_shape(m, n), make_stride(1, m));
    cpu_compare(tD_host_cute,tD_host_cpu,1);
  }
  else if(transA == 'T' && transB == 'N')
  {
    using namespace cute;
    TI * Dptr_host_cpu = (TI *)malloc(sizeof(TI) * m * n);
    TI * Dptr_host = (TI *)malloc(sizeof(TI) * m * n);
    TA * Dptr_host_A = &h_A[0];
    TB * Dptr_host_B = &h_B[0];
    auto tD_host_cpu =
      make_tensor(Dptr_host_cpu, make_shape(m, n), make_stride(1, m));
    auto tA = make_tensor(Dptr_host_A, make_shape(m, k), make_stride(k,1));
    auto tB = make_tensor(Dptr_host_B, make_shape(n, k), make_stride(k,1));

    int m = size<0>(tA);
    int n = size<0>(tB);
    int k = size<1>(tA);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int s = 0;

        for (int kk = 0; kk < k; ++kk) {
          TA v1 = tA(i, kk);
          TB v2 = tB(j, kk);
          s += (int)(v1 * v2);
        }
        tD_host_cpu(i, j) = s;
      }
    }

    hipMemcpy(Dptr_host, d_C.data().get(), sizeof(TI) * m * n, hipMemcpyDeviceToHost);
    auto tD_host_cute = make_tensor(Dptr_host, make_shape(m, n), make_stride(1, m));
    cpu_compare(tD_host_cute,tD_host_cpu,1);
  }


#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

  auto host_matrix_to_const_column_major_cute_tensor =
    [](const auto& X, int num_rows, int num_cols, int LDX) {
      const auto shape = cute::Shape<int, int>{num_rows, num_cols};
      const auto strides = cute::Stride<int, int>{1, LDX};
      return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
    };

  const auto A_view = host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
  // B^T is k x n, so B is n x k.
  const auto B_view = host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
  const auto C_computed_view = host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
  const auto C_expected_view = host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
  print_matrix_multiply_mollified_relative_error("float", A_view, B_view, C_computed_view, C_expected_view);

  for(int i=0; i<256; i++){
    printf("%f ", cute_result[i]);
    if(i % 16 == 15){
      printf("\n");
    }
  }
  printf("\n");
  for(int i=0; i<10; i++){
    printf("%f ", cute_result[i * (m * n / 10)]);
  }
  printf("\n\n");

  for(int i=0; i<256; i++){
    printf("%f ", cublas_result[i]);
    if(i % 16 == 15){
      printf("\n");
    }
  }
  printf("\n");
  for(int i=0; i<10; i++){
    printf("%f ", cublas_result[i * (m * n / 10)]);
  }
  printf("\n\n");

#endif // CUTLASS_ENABLE_CUBLAS
  return 0;
}
