#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>

#include "aiter_hip_common.h"
#include "m_grouped_gemm_nt_masked.h"
#include "intrinsic.h"
#include "m_grouped_gemm_config.h"


#define DEBUG 0

using half_t = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

/* module function args */
template <typename T>
struct HipFunctionArgs {
  uint32_t num_MBlocks;
  uint32_t num_NBlocks;
  T* ptr_C;                          // output
  const char* ptr_A;              // input
  const char* ptr_B;               // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  const int32_t* masked_m;       // tokens per expert
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_ase;               // weight scale stride expert
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  uint32_t numFullBlocks;
  uint32_t wgmRemainder1;
  uint32_t magicNumberWgmRemainder1;
  void *debugBuffer;
  const int32_t* signal;
};

namespace deepgemm {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  uint32_t expected_m_per_group, 
  uint32_t experts_num) {

  const int size_m = input.size(1); 
  const int size_n = b_scale.size(1);
  const int size_k = input.size(2);
  const int workgroupmapping = 8;	//



  /* grid sizes */
  size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  auto need_size_m = std::min(size_m, expected_m_per_group*1.3);
  // globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
  // globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
  // globalWorkSize[2] = experts_num;

  globalWorkSize[0] = 128;

  unsigned int problemNumGroupTiles0 = globalWorkSize[0];
  unsigned int problemNumGroupTiles1 = globalWorkSize[1];
  const unsigned smallNumMagicShift = 31; // bozo, review
  unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
  unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
  if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
  unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.num_MBlocks = DIVIDE(size_m, BLOCKN);
  hipFunctionArgs.num_NBlocks = DIVIDE(size_n, BLOCKM);
  
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = (const char*)b_qweight.data_ptr();
  hipFunctionArgs.ptr_B = (const char*)input.data_ptr();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  hipFunctionArgs.masked_m = masked_m.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_ase = b_scale.stride(0);
  hipFunctionArgs.stride_asm = b_scale.stride(1);
  hipFunctionArgs.stride_ask = b_scale.stride(2);
  hipFunctionArgs.stride_bse = a_scale.stride(0);
  hipFunctionArgs.stride_bsn = a_scale.stride(1);
  hipFunctionArgs.stride_bsk = a_scale.stride(2);
  hipFunctionArgs.numFullBlocks  = numFullBlocks;
  hipFunctionArgs.wgmRemainder1  = wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  std::cout << "_m_grouped_marlin_fp8_gemm_nt_masked_asm_impl "
          << " size_m: " << size_m
          << " need_size_m: " << need_size_m
          << " size_n: " << size_n
          << " size_k: " << size_k
          << " expected_m_per_group: " << expected_m_per_group
          << " experts_num: " << experts_num
          << std::endl;
  int bytes = 64;
  int debug_size = BLOCKDIM * bytes;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  char funcName[1024], fp8_groupgemm_co_file[1024];
  memset(funcName, 0x0, sizeof(funcName));
  memset(fp8_groupgemm_co_file, 0x0, sizeof(fp8_groupgemm_co_file));
  
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    sprintf(funcName, "DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_MASKED", BLOCKM, BLOCKN, BLOCKK);
    sprintf(fp8_groupgemm_co_file, "deepgemm_groupgemm_masked_fp8_marlin_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
  } else {
    TORCH_CHECK(false, "deepgemm_masked_fp8 only supports outputType in [BFloat16]");
  }

  static AiterAsmKernel moe_groupgemm_marlin_fp8(funcName, fp8_groupgemm_co_file);
  moe_groupgemm_marlin_fp8.launch_kernel_ext({
    &hipFunctionArgs,
    &argsSize,
    globalWorkSize[0],           // gdx
    globalWorkSize[1],           // gdy
    globalWorkSize[2],           // gdz
    localWorkSize[0],            // bdx: 12 wv64
    localWorkSize[1],            // bdy
    localWorkSize[2],            // bdz
    stream
  });
  
  #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d argsSize:%ld\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2], argsSize);
  printf("num_MBlocks:%p\nnum_NBlocks:%p\nptr_C:%p\nptr_A:%p\nptr_B:%p\nptr_A_scale:%p\nptr_B_scale:%p\nproblem_size:%p\nexperts_num:%p\n"
    "size_m:%p\nsize_n:%p\nsize_k:%p\nstride_ase:%p\nstride_asm:%p\nstride_ask:%p\nstride_bse:%p\nstride_bsn:%p\nstride_bsk:%p\nnumFullBlocks:%p\nwgmRemainder1:%p\nmagicNumberWgmRemainder1:%p\ndebugBuffer%p\n",
    &hipFunctionArgs.num_MBlocks, &hipFunctionArgs.num_NBlocks, &hipFunctionArgs.ptr_C,
    &hipFunctionArgs.ptr_A, &hipFunctionArgs.ptr_B, &hipFunctionArgs.ptr_A_scale,
    &hipFunctionArgs.ptr_B_scale,// &hipFunctionArgs.topk_weights, &hipFunctionArgs.expert_offsets, 
    &hipFunctionArgs.masked_m, &hipFunctionArgs.experts_num, &hipFunctionArgs.size_m,
    &hipFunctionArgs.size_n, &hipFunctionArgs.size_k, &hipFunctionArgs.stride_ase, &hipFunctionArgs.stride_asm, 
    &hipFunctionArgs.stride_ask, &hipFunctionArgs.stride_bse, &hipFunctionArgs.stride_bsn,
    &hipFunctionArgs.stride_bsk, &hipFunctionArgs.numFullBlocks, &hipFunctionArgs.wgmRemainder1, &hipFunctionArgs.magicNumberWgmRemainder1, &hipFunctionArgs.debugBuffer
  );
  printf("\ndebug:\n");
  hipMemcpy(debug_host, hipFunctionArgs.debugBuffer, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 8 == 0) printf("\n");
    if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    printf("%6d\t", p_debug[i]);
    // printf("%10d\t", p_debug[i]);
    // printf("%10.4f\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = split * i;
    // printf("tid:%d %3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   "%3d %3d %3d %3d %3d %3d %3d %3d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ],
    //   p_debug[index+4 ], p_debug[index+5 ], p_debug[index+6 ], p_debug[index+7 ],
    //   p_debug[index+8 ], p_debug[index+9 ], p_debug[index+10], p_debug[index+11],
    //   p_debug[index+12], p_debug[index+13], p_debug[index+14], p_debug[index+15]
    //   // ,
    //   // p_debug[index+16], p_debug[index+17], p_debug[index+18], p_debug[index+19],
    //   // p_debug[index+20], p_debug[index+21], p_debug[index+22], p_debug[index+23],
    //   // p_debug[index+24], p_debug[index+25], p_debug[index+26], p_debug[index+27],
    //   // p_debug[index+28], p_debug[index+29], p_debug[index+30], p_debug[index+31]
    // );
    // printf("tid:%d %6d %6d %6d %6d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ]
    // );
    // int8_t * tmp = reinterpret_cast<int8_t*>(p_debug + index);
    // printf("tid:%d %3d %3d %3d %3d 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X %3d %3d %3d %3d\n", i, 
    //   tmp[0], tmp[1], tmp[2], tmp[3],
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], p_debug[index+4 ], p_debug[index+5 ], 
    //   tmp[20], tmp[21], tmp[22], tmp[23] 
    // );
  }
  printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}


template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl_enableOverlap(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  torch::Tensor signal,
  uint32_t expected_m_per_group, 
  uint32_t experts_num) {

  const int size_m = input.size(1); 
  const int size_n = b_scale.size(1);
  const int size_k = input.size(2);
  const int workgroupmapping = 8;	//



  /* grid sizes */
  size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  auto need_size_m = std::min(size_m, expected_m_per_group*1.3);
  // globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
  // globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
  // globalWorkSize[2] = experts_num;

  globalWorkSize[0] = 128;

  unsigned int problemNumGroupTiles0 = globalWorkSize[0];
  unsigned int problemNumGroupTiles1 = globalWorkSize[1];
  const unsigned smallNumMagicShift = 31; // bozo, review
  unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
  unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
  if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
  unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.num_MBlocks = DIVIDE(size_m, BLOCKN);
  hipFunctionArgs.num_NBlocks = DIVIDE(size_n, BLOCKM);
  
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = (const char*)b_qweight.data_ptr();
  hipFunctionArgs.ptr_B = (const char*)input.data_ptr();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  hipFunctionArgs.masked_m = masked_m.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_ase = b_scale.stride(0);
  hipFunctionArgs.stride_asm = b_scale.stride(1);
  hipFunctionArgs.stride_ask = b_scale.stride(2);
  hipFunctionArgs.stride_bse = a_scale.stride(0);
  hipFunctionArgs.stride_bsn = a_scale.stride(1);
  hipFunctionArgs.stride_bsk = a_scale.stride(2);
  hipFunctionArgs.numFullBlocks  = numFullBlocks;
  hipFunctionArgs.wgmRemainder1  = wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;
  hipFunctionArgs.signal = signal.data_ptr<int32_t>();
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  std::cout << "_m_grouped_marlin_fp8_gemm_nt_masked_asm_impl "
          << " size_m: " << size_m
          << " need_size_m: " << need_size_m
          << " size_n: " << size_n
          << " size_k: " << size_k
          << " expected_m_per_group: " << expected_m_per_group
          << " experts_num: " << experts_num
          << std::endl;
  int bytes = 64;
  int debug_size = BLOCKDIM * bytes;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  char funcName[1024], fp8_groupgemm_co_file[1024];
  memset(funcName, 0x0, sizeof(funcName));
  memset(fp8_groupgemm_co_file, 0x0, sizeof(fp8_groupgemm_co_file));
  
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    sprintf(funcName, "DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_MASKED_OVERLAP", BLOCKM, BLOCKN, BLOCKK);
    sprintf(fp8_groupgemm_co_file, "deepgemm_groupgemm_masked_fp8_marlin_%dx%dx%d_TN_BF16_WGM8_OVERLAP.co", BLOCKM, BLOCKN, BLOCKK);
  } else {
    TORCH_CHECK(false, "deepgemm_masked_fp8 only supports outputType in [BFloat16]");
  }

  static AiterAsmKernel moe_groupgemm_marlin_fp8(funcName, fp8_groupgemm_co_file);
  moe_groupgemm_marlin_fp8.launch_kernel_ext({
    &hipFunctionArgs,
    &argsSize,
    globalWorkSize[0],           // gdx
    globalWorkSize[1],           // gdy
    globalWorkSize[2],           // gdz
    localWorkSize[0],            // bdx: 12 wv64
    localWorkSize[1],            // bdy
    localWorkSize[2],            // bdz
    stream
  });
  
  #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d argsSize:%ld\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2], argsSize);
  printf("num_MBlocks:%p\nnum_NBlocks:%p\nptr_C:%p\nptr_A:%p\nptr_B:%p\nptr_A_scale:%p\nptr_B_scale:%p\nproblem_size:%p\nexperts_num:%p\n"
    "size_m:%p\nsize_n:%p\nsize_k:%p\nstride_ase:%p\nstride_asm:%p\nstride_ask:%p\nstride_bse:%p\nstride_bsn:%p\nstride_bsk:%p\nnumFullBlocks:%p\nwgmRemainder1:%p\nmagicNumberWgmRemainder1:%p\ndebugBuffer%p\n",
    &hipFunctionArgs.num_MBlocks, &hipFunctionArgs.num_NBlocks, &hipFunctionArgs.ptr_C,
    &hipFunctionArgs.ptr_A, &hipFunctionArgs.ptr_B, &hipFunctionArgs.ptr_A_scale,
    &hipFunctionArgs.ptr_B_scale,// &hipFunctionArgs.topk_weights, &hipFunctionArgs.expert_offsets, 
    &hipFunctionArgs.masked_m, &hipFunctionArgs.experts_num, &hipFunctionArgs.size_m,
    &hipFunctionArgs.size_n, &hipFunctionArgs.size_k, &hipFunctionArgs.stride_ase, &hipFunctionArgs.stride_asm, 
    &hipFunctionArgs.stride_ask, &hipFunctionArgs.stride_bse, &hipFunctionArgs.stride_bsn,
    &hipFunctionArgs.stride_bsk, &hipFunctionArgs.numFullBlocks, &hipFunctionArgs.wgmRemainder1, &hipFunctionArgs.magicNumberWgmRemainder1, &hipFunctionArgs.debugBuffer
  );
  printf("\ndebug:\n");
  hipMemcpy(debug_host, hipFunctionArgs.signal, debug_size, hipMemcpyDeviceToHost);
  // using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 8 == 0) printf("\n");
    if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    printf("%6d\t", p_debug[i]);
    // printf("%10d\t", p_debug[i]);
    // printf("%10.4f\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = split * i;
    // printf("tid:%d %3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   "%3d %3d %3d %3d %3d %3d %3d %3d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ],
    //   p_debug[index+4 ], p_debug[index+5 ], p_debug[index+6 ], p_debug[index+7 ],
    //   p_debug[index+8 ], p_debug[index+9 ], p_debug[index+10], p_debug[index+11],
    //   p_debug[index+12], p_debug[index+13], p_debug[index+14], p_debug[index+15]
    //   // ,
    //   // p_debug[index+16], p_debug[index+17], p_debug[index+18], p_debug[index+19],
    //   // p_debug[index+20], p_debug[index+21], p_debug[index+22], p_debug[index+23],
    //   // p_debug[index+24], p_debug[index+25], p_debug[index+26], p_debug[index+27],
    //   // p_debug[index+28], p_debug[index+29], p_debug[index+30], p_debug[index+31]
    // );
    // printf("tid:%d %6d %6d %6d %6d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ]
    // );
    // int8_t * tmp = reinterpret_cast<int8_t*>(p_debug + index);
    // printf("tid:%d %3d %3d %3d %3d 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X %3d %3d %3d %3d\n", i, 
    //   tmp[0], tmp[1], tmp[2], tmp[3],
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], p_debug[index+4 ], p_debug[index+5 ], 
    //   tmp[20], tmp[21], tmp[22], tmp[23] 
    // );
  }
  printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}

// cuda

template<
    typename Element, 
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M, 
    uint16_t BLOCK_SIZE_N, 
    uint16_t BLOCK_SIZE_K, 
    uint16_t WARP_M, 
    uint16_t WARP_N, 
    uint16_t WARP_K,
    int SIZE_K,
    int STAGE>                                                                                                            
__forceinline__ __device__ void  gemm_nt_marlin_fp8_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    union_vec<Element,WARP_K/4> A_reg[][STAGE], // 2 = stage
    union_vec<Element,WARP_K/4> B_reg[][STAGE], // 2 = stage
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int actual_m, 
    const int size_m,
    const int size_n,
    const int size_k,
    const int bidx) {

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
    constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = (warp_id / warp_k_num) % warp_n_num;
    int warp_m_id = (warp_id / warp_k_num) / warp_n_num;



    //warp内部k连续[warpn, 64] warp外部在N方向连续 
    int k_start = warp_k_id * WARP_K;
    //int k_start_b = warp_k_id * WARP_K * size_n; // moe
    int k_start_b = warp_k_id * WARP_K * 16;
    const int stage_offset = warp_k_num * WARP_K;
    //const int stage_offset_b = warp_k_num * WARP_K * size_n; // moe
    const int stage_offset_b = warp_k_num * WARP_K * 16;
    
    // 提前读取token_idx到reg
    int g_row_A[WARP_M / MFMA_M]; 
    int g_row_B[WARP_N / MFMA_N];
    

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int m_offset = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
        m_offset = m_offset >= actual_m ? actual_m % BLOCK_SIZE_M : m_offset;
        g_row_A[m_tile] = m_offset * size_k + col_id * 16;
        //g_row_A[m_tile] = (m_tile * MFMA_M + row_id) * size_k + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        //g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N)* 64 + col_id * 16; //warp内部k连续[warpn, 64] warp外部在N方向连续 
        int n_offset = warp_n_id * WARP_N + row_id + n_tile * MFMA_N;
        g_row_B[n_tile] = (n_offset / 16) * 16 * size_k + (n_offset % 16) * 16 + col_id * 16 * 16;
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    // read 0
    // load A
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * 16 + k_start_b);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;
    
    for( ; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE, k_start_b += stage_offset_b * STAGE){

        #pragma unroll
        for(int i = 0; i < STAGE; ++i){
            ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
            // load A
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

            // load B
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                    //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

            ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){  
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac_fp8<Element>(
                          *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].uint8t_array[k_tile]), 
                          *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].uint8t_array[k_tile]), 
                          C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]); 
                    }
                }
            }
        } // stage end

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//
    if(k_start + stage_offset < size_k){ // size_k / block 为偶数
        //  --------------------------------------------- load mmac ---------------------------------- //
        int i = 0;
        if constexpr (STAGE == 4){ //TODO: STAGE为4 尾处理有四种可能方式 需要size_k来确定loop次数，size_k如果为动态传入会有寄存器溢出 目前写死解决 
            //constexpr int epilogue_tile =  ( 7168/BLOCK_SIZE_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;        
            constexpr int epilogue_tile =  ( SIZE_K/BLOCK_SIZE_K - (STAGE - 1)) % STAGE;   
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                        
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac_fp8<Element>(
                              *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].uint8t_array[k_tile]), 
                              *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].uint8t_array[k_tile]), 
                              C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        } else{ // two stage
            constexpr int epilogue_tile =  1;      
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){  

                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac_fp8<Element>(
                              *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].uint8t_array[k_tile]), 
                              *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].uint8t_array[k_tile]), 
                              C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        }
        //  --------------------------------------------- load mmac ---------------------------------- //

        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int ii = 1; ii < STAGE; ++ii){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac_fp8<Element>(
                          *(typename vec<Element, 8>::type*)(&A_reg[m_tile][(i+ii-1)%STAGE].uint8t_array[k_tile]), 
                          *(typename vec<Element, 8>::type*)(&B_reg[n_tile][(i+ii-1)%STAGE].uint8t_array[k_tile]), 
                          C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]); 
                    }


                }
            }

        }

    }
    else{ // todo 不为BLOCK_SIZE_K的倍数
        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int i = 1; i < STAGE; ++i){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac_fp8<Element>(
                          *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i-1].uint8t_array[k_tile]), 
                          *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i-1].uint8t_array[k_tile]), 
                          C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]); 
                    }
                }
            }
        }
    }
///////////////////////////////////////////////////////////执行规约/////////////////////////////////////////////////////////
    // warp间的规约 
    extern __shared__ float out_smem[]; // 声明lds信息
    if constexpr (warp_k_num > 1){

        constexpr int pading_n = WARP_N + 1;

        if(warp_k_id > 0){ //0和1 只需要 warpk_id为1去拷贝数据
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                   
                    *(floatx4*)(&out_smem[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n  + n_tile * 16 + col_id * 4 + (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * pading_n * BLOCK_SIZE_M])  = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];

                }
            }
        }

        __syncthreads();


        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num - 1; k_tile++){
                      floatx4 temp = *(floatx4*)(&out_smem[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n  + n_tile * 16 + col_id * 4 + (k_tile +  warp_n_id * (warp_k_num-1))  * pading_n * BLOCK_SIZE_M]);
                        #pragma unroll
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += temp[i];
        
                        }
                    }
                }
            }
        }
    }
}


template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K,
  int STAGES
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_GROUPGEMM_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int WARP_K_NUM = BLOCK_SIZE_K / WARP_K;
  const int bidx = blockIdx.z; // pid_m  m方向z
  const int bidy = blockIdx.y; // pid_n  n方向
  const int bidz = blockIdx.x; // pid_k  e方向
  const int real_m = masked_m[bidz];
  const int actual_m = ((real_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * BLOCK_SIZE_M; //往上取block的整数幂


  if(bidx * BLOCK_SIZE_M >= actual_m) return; // 对于无效的block,直接返回

  const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
  const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
  //const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * 64; // moe
  const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
  const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
  const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

  auto g_input = input + input_offset;
  auto g_qweight = qweight + qweight_offset;
  auto g_input_scale = input_scale + input_scale_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  OutputType* g_output = output + output_offset;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;

  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = (warp_id / warp_k_num) % warp_n_num;
  int warp_m_id = (warp_id / warp_k_num) / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds; // decode这里没用
  //OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds
  uint16_t* output_lds = (uint16_t*)&(smem); // 重复使用lds,留给output_lds
  
  union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 


  // device gemm todo 
  gemm_nt_marlin_fp8_decode<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
    (g_input, g_qweight, input_lds, qweight_lds,
     A_reg, B_reg, C_reg, warp_id, real_m, size_m, size_n, size_k, bidx);

  // write global mem
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      float a_scale = g_input_scale[warp_m_id * WARP_M + min_tile_m * mfma_m + row_id];
      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
        #pragma unroll  
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale * b_scale; 
          int m_offset = min_tile_m * mfma_m + warp_m_id * WARP_M + row_id;
          int index = m_offset * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + reg_id * 4 + col_id + ( m_offset )/2 * 2/*padding*/;
          output_lds[index] = f32_to_output<OutputType>(value);
          // #if defined(__gfx938__)
          // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
          // #endif
        }
      }
    }
  }

  __syncthreads();

  
  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      if(m_idx < real_m) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 ]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  } 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 目前只适配m方向只有一个block和只有一个warp的情况
template <uint16_t BLOCK_N>
struct Scheduler {
public:
    int bidy;
    int bidz;
    int current_idx;
    int experts_num;
    int n_blocks;

public:
    // 构造函数，根据定义的变量进行初始化
    __device__ Scheduler(int _current_idx,int _bidy, int _bidz, int _experts_num, int _n_blocks)
        : current_idx(_current_idx),
          bidy(_bidy),
          bidz(_bidz),
          experts_num(_experts_num),
          n_blocks(_n_blocks)
    {}

 
    __device__ bool get_next_block() {
        get_swizzled_block_idx();

        if(bidz >= experts_num) return false;
        
        return true;  // 找到有效tile
    }
  
    __device__ int get_bidy() const {
        return bidy;
    }

    __device__ int get_bidz() const {
        return bidz;
    }
  
    __device__ void get_swizzled_block_idx() {
        bidy = current_idx % n_blocks;
        bidz = current_idx / n_blocks;
    }

    __device__ void advance() {
        current_idx += gridDim.x;  // 跳过gridDim.x个索引，实现不同block的任务交错
    }
};


////持久化内核 默认BLOCKM = WARPM = 16   M方向多次loop完成M的迭代
template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K,
  int STAGES
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_PERSISTENT_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int WARP_K_NUM = BLOCK_SIZE_K / WARP_K;


  Scheduler<BLOCK_SIZE_N> scheduler(blockIdx.x, 0, 0, experts_num, size_n / BLOCK_SIZE_N);

  while(scheduler.get_next_block()){

  
    int bidy = scheduler.get_bidy(); // pid_n  n方向
    int bidz = scheduler.get_bidz(); // pid_k  e方向
    
    const int real_m = masked_m[bidz];
    const int actual_m = ((real_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * BLOCK_SIZE_M; //往上取block的整数幂
    const int m_loop = actual_m / BLOCK_SIZE_M; // M方向需要loop的次数

    if(real_m == 0) // 对于无效的block,直接返回
    {
      scheduler.advance();
      continue;
    } 

    // M方向的loop
    for(int bidx = 0; bidx < m_loop; ++bidx){
      const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
      const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
      //const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * 64; // moe
      const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
      const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
      const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

      auto g_input = input + input_offset;
      auto g_qweight = qweight + qweight_offset;
      auto g_input_scale = input_scale + input_scale_offset;
      auto g_weight_scale = weight_scale + weight_scale_offset;
      OutputType* g_output = output + output_offset;

      int warp_id_vec = threadIdx.x / 64; //warp id in a block
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
      int lane_id = threadIdx.x & 63; // thread_id
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;
      constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
      constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;

      int warp_k_id = warp_id % warp_k_num;
      int warp_n_id = (warp_id / warp_k_num) % warp_n_num;
      int warp_m_id = (warp_id / warp_k_num) / warp_n_num;

      extern __shared__ Element smem[]; // 声明lds信息
      Element* input_lds = (Element*)&(smem); // decode这里没用
      Element* qweight_lds = input_lds; // decode这里没用
      //OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds
      uint16_t* output_lds = (uint16_t*)&(smem); // 重复使用lds,留给output_lds
      
      union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
      union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
      floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 


      // device gemm todo 
      gemm_nt_marlin_fp8_decode<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
        (g_input, g_qweight, input_lds, qweight_lds,
        A_reg, B_reg, C_reg, warp_id, real_m, size_m, size_n, size_k, bidx);

      // write global mem
      if(warp_k_id == 0 || warp_k_num == 1)
      {
        for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
          float a_scale = g_input_scale[warp_m_id * WARP_M + min_tile_m * mfma_m + row_id];
          for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
            #pragma unroll  
            for(int reg_id = 0; reg_id < 4; reg_id++){
              float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
              float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale * b_scale; 
              int m_offset = min_tile_m * mfma_m + warp_m_id * WARP_M + row_id;
              int index = m_offset * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + reg_id * 4 + col_id + ( m_offset )/2 * 2/*padding*/;
              output_lds[index] = f32_to_output<OutputType>(value);
              // #if defined(__gfx938__)
              // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
              // #endif
            }
          }
        }
      }

      __syncthreads();

      
      {
        // 最大化利用dwordx4 通用
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
        using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
        int m_idx = threadIdx.x / N_thread;  
        int n_idx = threadIdx.x % N_thread;  
        for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
          if(m_idx < real_m) {
            *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
            //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 ]);
            *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
          }
        } 
      } 
      // lgkmcnt_wait_barrier(0);
      // __syncthreads();
    } // M方向的loop
    scheduler.advance();
  } // schedule loop end
}


template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_fp8_masked_decode(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K) * (BLOCK_SIZE_M / WARP_M);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  
  uint32_t need_size_m = std::min(size_m, (uint32_t)(params.expected_m_per_group * 1.3));
  gridDim.z = DIVIDE(need_size_m, BLOCK_SIZE_M);
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.x = params.experts_num;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int paddingN = (WARP_N + 1) * (BLOCK_SIZE_N / WARP_N);
  //const int lds_size = std::max(BLOCK_SIZE_M * paddingN * (BLOCK_SIZE_K / WARP_K - 1) * 4, BLOCK_SIZE_M * BLOCK_SIZE_N * 4);
  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;
  
  // 模板化size_k是为了避免四级流水寄存器溢出
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_GROUPGEMM_MASKED<uint8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const uint8_t*)params.ptr_A, 
      (const uint8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_GROUPGEMM_MASKED<uint8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const uint8_t*)params.ptr_A, 
      (const uint8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
      
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_fp8_masked_decode_persistent(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K) * (BLOCK_SIZE_M / WARP_M);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  
  static int DCU_CUS = []() {
      const char* env = std::getenv("DEEPGEMM_GPU_CUS");
      return env ? std::strtol(env, nullptr, 10) : 64;
  }();

  int parallel = 2;
  gridDim.z = 1;
  gridDim.y = 1;
  gridDim.x = parallel * DCU_CUS;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int paddingN = (WARP_N + 1) * (BLOCK_SIZE_N / WARP_N);
  //const int lds_size = std::max(BLOCK_SIZE_M * paddingN * (BLOCK_SIZE_K / WARP_K - 1) * 4, BLOCK_SIZE_M * BLOCK_SIZE_N * 4);
  const int lds_size =  BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;  //待优化减少
  
  // 模板化size_k是为了避免四级流水寄存器溢出
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_PERSISTENT_MASKED<uint8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const uint8_t*)params.ptr_A, 
      (const uint8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_FP8_DECODE_MARLIN_TN_PERSISTENT_MASKED<uint8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const uint8_t*)params.ptr_A, 
      (const uint8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
      
}




// cuda end


torch::Tensor m_grouped_marlin_fp8_gemm_nt_masked(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m, 
  int expected_m_per_group,
  bool enable_overlap,
  std::optional<torch::Tensor> signal,
  int mode) {
  
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  uint32_t experts_num = b_qweight.sizes()[0];
  const int size_m = input.size(1); 
  const int size_n = b_scale.size(1);  // 直接用缩放因子的shape来获取size_n，避免了之前从b_qweight获取size_n时需要marlin重排后shape不匹配的问题
  const int size_k = input.size(2);
  if(mode >= 1000) {
    if (mode == 1000) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
          _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale,
            masked_m, expected_m_per_group, experts_num);
        }
      );
    } else if(mode == 1001) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
          _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 128, 128, scalar_t, 768>(
            input, b_qweight, output, a_scale, b_scale,
            masked_m, expected_m_per_group, experts_num);
        }
      );
    }else if(mode == 1002) {
      if(enable_overlap ){    // enable sbo down gemm
      // if(enable_overlap && (size_n ==7168 || size_n == 8192)){    // enable sbo down gemm
        if (signal) {
          AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            output.scalar_type(), "m_groupgemm_fp8_TN_impl_enableOverlap",[&] {
              _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl_enableOverlap<256, 64, 128, scalar_t, 512>(
                input, b_qweight, output, a_scale, b_scale,
                masked_m, *signal, expected_m_per_group, experts_num);
            }
          );
        }else{
          TORCH_CHECK(false, mode, " enable_overlap is true but signal is not provided.");
        }
        
      }else{ // 不启用sbo，或者n不满足条件的情况，仍然使用原来的kernel
        AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          output.scalar_type(), "m_groupgemm_fp8_TN_impl",[&] {
            _m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 64, 128, scalar_t, 512>(
              input, b_qweight, output, a_scale, b_scale,
              masked_m, expected_m_per_group, experts_num);
          }
        );
      }
      
    }
  } else if(mode <= 500){

    GroupGemmParams<uint8_t, bhalf_t> params(
      (uint8_t*)input.data_ptr(),
      (uint8_t*)b_qweight.data_ptr(), 
      (bhalf_t*)output.data_ptr(),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      masked_m.data_ptr<int32_t>(),
      experts_num,
      size_m,
      size_n,
      size_k,
      expected_m_per_group
    );

    //launch_grouped_marlin_w8a8_fp8_masked_decode_persistent<16, 32, 64, 16, 16, 64, 2, uint8_t, bhalf_t>(params);

    //查找 kernel map
    if(size_k == 7168){ // gemm1
      auto it = kernel_maps_groupgemm1_fp8_decode.find(mode);
      if (it != kernel_maps_groupgemm1_fp8_decode.end()) {
          it->second(params);
      } else {
          // 未找到对应的 mode，使用默认配置
          printf("DeepGEMM_Decode: No matching kernel configuration found for mode %d, using default settings\n", mode);
          launch_grouped_marlin_w8a8_fp8_masked_decode<16, 32, 64, 16, 16, 64, 2, uint8_t, bhalf_t>(params);
      }
    } else { // gemm2
      auto it = kernel_maps_groupgemm2_fp8_decode.find(mode);
      if (it != kernel_maps_groupgemm2_fp8_decode.end()) {
          it->second(params);
      } else {
          // 未找到对应的 mode，使用默认配置
          printf("DeepGEMM_Decode: No matching kernel configuration found for mode %d, using default settings\n", mode);
          launch_grouped_marlin_w8a8_fp8_masked_decode<16, 32, 64, 16, 16, 64, 2, uint8_t, bhalf_t>(params);
      }
    }

  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
  return output;
}

} // namespace deepgemm