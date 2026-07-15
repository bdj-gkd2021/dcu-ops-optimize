#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>
#include <mutex>
#include <queue>

#include "aiter_hip_common.h"
#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"

using half_t  = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

struct __attribute__((packed)) GemmInfoHost {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    void* d;
    void* c;
    int8_t* a;
    int8_t* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t  alpha[16];
    int8_t   beta[16];
    float* scaleA;
    float* scaleB;
};

template <typename T>
struct __attribute__((packed)) GemmInfo {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    T* d;
    T* c;
    int8_t* a;
    int8_t* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t  alpha[16];
    int8_t   beta[16];
    float* scaleA=nullptr;
    float* scaleB=nullptr;
};

namespace deepgemm {

constexpr int POOL_SIZE = 1024;

struct PoolEntry {
    GemmInfoHost* h_ptr;
    void* d_ptr;
};

static std::vector<PoolEntry> g_pool;
static std::mutex g_pool_mutex;
static std::condition_variable g_pool_cv;
static bool g_pool_initialized = false;
static std::once_flag g_pool_init_flag;

static hipStream_t g_callback_stream;
static hipEvent_t g_callback_event;
static std::once_flag g_callback_stream_init_flag;

static void init_callback_stream() {
    std::call_once(g_callback_stream_init_flag, [&]() {
        hipStreamCreate(&g_callback_stream);
        hipEventCreate(&g_callback_event);
    });
}

static void init_pool() {
    std::call_once(g_pool_init_flag, [&]() {
        g_pool.reserve(POOL_SIZE);
        for (int i = 0; i < POOL_SIZE; i++) {
            PoolEntry entry;
            hipHostMalloc(&entry.h_ptr, sizeof(GemmInfoHost), hipHostMallocMapped);
            hipHostGetDevicePointer(&entry.d_ptr, entry.h_ptr, 0);
            g_pool.push_back(entry);
        }
        g_pool_initialized = true;
    });
}

inline PoolEntry acquire_from_pool() {
    init_pool();
    std::unique_lock<std::mutex> lock(g_pool_mutex);
    g_pool_cv.wait(lock, [&]() { return !g_pool.empty(); });
    auto entry = g_pool.back();
    g_pool.pop_back();
    return entry;
}

inline void release_to_pool(PoolEntry entry) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.push_back(entry);
    g_pool_cv.notify_one();
}

void _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    torch::Tensor token_per_expert,
    uint32_t experts_num) {

    std::vector<GemmInfo<bhalf_t>> d_prob_vec;

    const int token_num = input.size(0);
    const int size_n = b_qweight.size(1) * 16;
    const int size_k = input.size(1);
    int32_t lda = size_k;
    int32_t ldb = size_k;
    int32_t ldc = size_n;
    int32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    int8_t* inp = input.data_ptr<int8_t>();
    int8_t* weight = b_qweight.data_ptr<int8_t>();
    int32_t* m_indices_ptr = m_indices.data_ptr<int32_t>();
    bhalf_t* output_ptr = reinterpret_cast<bhalf_t *>(output.data_ptr());
    float* scaleA = a_scale.data_ptr<float>();
    float* scaleB = b_scale.data_ptr<float>();

    size_t      size_nk = size_k * size_n;
    size_t stride_b = size_n * size_k;

    int32_t* token_per_expert_ptr = token_per_expert.data_ptr<int32_t>();
    int32_t start_token = 0;
    for (int i = 0, j = 0; i < token_num; ++j) {
        GemmInfo<bhalf_t> d_prob;
        int32_t expertId = m_indices_ptr[i];
        int32_t actual_size_m = token_per_expert_ptr[expertId];
        size_t stride_a = actual_size_m * size_k;
        size_t stride_c = actual_size_m * size_n;
        size_t stride_d = actual_size_m * size_n;
        d_prob.m = size_n;
        d_prob.n = actual_size_m;
        d_prob.batch = 1;
        d_prob.k = size_k;
        d_prob.strideD1 = ldd;
        d_prob.strideD2 = stride_d;
        d_prob.strideC1 = ldc;
        d_prob.strideC2 = stride_c;
        d_prob.strideA1 = ldb;
        d_prob.strideA2 = stride_b;
        d_prob.strideB1 = lda;
        d_prob.strideB2 = stride_a;

        memcpy(d_prob.alpha, &d_alpha, sizeof(float));
        memcpy(d_prob.beta,  &d_beta, sizeof(float));

        d_prob.a = weight + expertId * stride_b;
        //std::cout << std::hex << reinterpret_cast<void*>(d_prob.a) << std::endl;
        d_prob.b = inp + start_token * lda;
        // std::cout << std::hex << reinterpret_cast<void*>(d_prob.b) << std::endl;
        d_prob.c = output_ptr + start_token * ldc;
        d_prob.d = output_ptr + start_token * ldd;
        d_prob.scaleA = scaleB + expertId * ldd;
        d_prob.scaleB = scaleA + start_token;

        start_token += actual_size_m;
        i += actual_size_m;
        d_prob_vec.push_back(d_prob);
    }
    //printf("d_pro_vec size : %d\n", d_prob_vec.size());
    size_t localWorkSize[3] = {768, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    for (int i = 0; i < d_prob_vec.size(); ++i) {
        // printf("MNK %d,%d,%d,%d,GM=%d,", d_prob_vec[i].m, d_prob_vec[i].n, 1, d_prob_vec[i].k, i);
        // printf(" wgRight%d\n",wgTile);
        size_t size_m = d_prob_vec[i].m;
        size_t size_n = d_prob_vec[i].n;
        int wg_M = DIVIDE(size_m, 256);
        int wg_N = DIVIDE(size_n, 256);
        wgTile += wg_M * wg_N;
    }


        struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;
    }hipFunctionArgs;

    uint8_t* d_argsPtr;
    size_t arg_size =  d_prob_vec.size() * sizeof(GemmInfo<bhalf_t>);
    hipMalloc((void**)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, d_prob_vec.data(), arg_size, hipMemcpyHostToDevice);

    hipFunctionArgs.gemm_count = d_prob_vec.size();
    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    //hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    char func_name_fp[1024], fp16_co_file_tn[1024], bf16_co_file_tn[1024], func_name_bf[1024];
    memset(func_name_fp, 0x0, sizeof(func_name_fp));
    memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
    memset(func_name_bf, 0x0, sizeof(func_name_bf));
    memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));

    sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", 256, 256, 128);
    sprintf(bf16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_NOPAD_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", 256, 256, 128);      static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    impl_ptr = &deepgemm_w8a8_bf16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_w8a8_gemm_nt_contig_asm_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,                   // (M)
    torch::Tensor b_scale,                   // (E, N)
    torch::Tensor m_indices,
    uint32_t experts_num) {

    auto entry = acquire_from_pool();
    const int size_m = input.size(0);        //(M, K)
    const int size_n = b_qweight.size(1) * 16;    //(E, N // 16, K * 16)
    const int size_k = input.size(1);

    uint32_t lda = size_k;
    uint32_t ldb = size_k;
    uint32_t ldc = size_n;
    uint32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    size_t stride_a = size_m * size_k;
    size_t stride_b = size_n * size_k;
    size_t stride_c = size_m * size_n;
    size_t stride_d = size_m * size_n;
    entry.h_ptr->m = size_n;
    entry.h_ptr->n = size_m;
    entry.h_ptr->batch = 1;
    entry.h_ptr->k = size_k;
    entry.h_ptr->strideD1 = ldd;
    entry.h_ptr->strideD2 = stride_d;
    entry.h_ptr->strideC1 = ldc;
    entry.h_ptr->strideC2 = stride_c;
    entry.h_ptr->strideA1 = ldb;
    entry.h_ptr->strideA2 = stride_b;
    entry.h_ptr->strideB1 = lda;
    entry.h_ptr->strideB2 = stride_a;
    memcpy(entry.h_ptr->alpha, &d_alpha, sizeof(float));
    memcpy(entry.h_ptr->beta,  &d_beta, sizeof(float));
    entry.h_ptr->a = b_qweight.data_ptr<int8_t>();
    entry.h_ptr->b = input.data_ptr<int8_t>();
    entry.h_ptr->d = output.data_ptr();
    entry.h_ptr->scaleA = b_scale.data_ptr<float>();
    entry.h_ptr->scaleB = a_scale.data_ptr<float>();

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    void* d_argsPtr = entry.d_ptr;

    size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    int wg_M = DIVIDE(size_m, BLOCKM);
    int wg_N = DIVIDE(size_n, BLOCKN);
    wgTile += wg_M * wg_N;

        struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;
    }hipFunctionArgs;

    hipFunctionArgs.gemm_count = 1;
    hipFunctionArgs.DeviceUserArguments = (void const*)d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    static char func_name_fp[1024];
    static char fp16_co_file_tn[1024];
    static char func_name_bf[1024];
    static char bf16_co_file_tn[1024];
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        memset(func_name_fp, 0x0, sizeof(func_name_fp));
        memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
        memset(func_name_bf, 0x0, sizeof(func_name_bf));
        memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));
        sprintf(func_name_fp, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16", BLOCKM, BLOCKN, BLOCKK);
        sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", BLOCKM, BLOCKN, BLOCKK);
        sprintf(fp16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16.co", BLOCKM, BLOCKN, BLOCKK);
        sprintf(bf16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", BLOCKM, BLOCKN, BLOCKK);
    });

        static AiterAsmKernel deepgemm_w8a8_fp16_tn(func_name_fp, fp16_co_file_tn);
        static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    if (output.scalar_type() == at::ScalarType::BFloat16)
        impl_ptr = &deepgemm_w8a8_bf16_tn;
    else
        impl_ptr = &deepgemm_w8a8_fp16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });

    init_callback_stream();
    hipEventRecord(g_callback_event, stream);
    hipStreamWaitEvent(g_callback_stream, g_callback_event, 0);

    PoolEntry* e = new PoolEntry(entry);
    hipStreamAddCallback(g_callback_stream, [](hipStream_t, hipError_t, void* userData) {
        PoolEntry* e = static_cast<PoolEntry*>(userData);
        release_to_pool(*e);
        delete e;
    }, e, 0);

    return;
}

torch::Tensor m_grouped_w8a8_gemm_nt_contiguous(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  int mode) {

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    uint32_t experts_num = b_qweight.sizes()[0];
    if (mode == 1000) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else if (mode == 100) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 128, 128, scalar_t, 512>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else {
        TORCH_CHECK(false, mode, " mode error.");
    }
  return output;
}

torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contiguous(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  torch::Tensor token_per_expert) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    uint32_t experts_num = b_qweight.sizes()[0];
    _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert, experts_num
    );
    return output;
  }
}
