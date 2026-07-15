

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 64
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Input validation macros (consistent with flash_api.cpp and flash_api_sparse.cpp)
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
using uint8x4_t = __attribute__( (__vector_size__(4 * sizeof(uint8_t)) )) uint8_t;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using half8_t = __attribute__( (__vector_size__(8 * sizeof(_Float16)) )) _Float16;
using v4bh = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using float4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using float2_t = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
using intx2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using intx4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;

static constexpr int LDS_size = 65536;
static constexpr int max_tmp_offset=4000000;
static constexpr int signal_tmp_offset=8000000;
static constexpr int streamk_max_block=160*8;
static constexpr int out_tmp_offset=signal_tmp_offset+streamk_max_block*2;
// static constexpr int PARTITION_SIZE=512;
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1); 
    u += 0x8000; 
    out = u>>16;
  }
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}


inline __device__ float uint82float(const uint8_t& input) {
#if (defined(__gfx938__) )
  return __builtin_hcu_cvt_f32_fp8(input,false,0,0);
#else
  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));
  return c10::detail::fp32_from_bits(result);
#endif
}
template<typename scalar_t,bool is_e4m3>
__forceinline__ __device__ scalar_t uint82half(const uint8_t& input) {
  union uf16{
    uint16_t as_bits;
    _Float16 as_value;
  } ;
  union uf32 {
    uint32_t as_bits;
    float as_value;
  };
  if constexpr(!is_e4m3){
    uf16 u16;
    u16.as_bits = (uint16_t)input << 8;
    if constexpr(std::is_same<scalar_t, _Float16>::value){
      return u16.as_value;
    }
    else{
      uf32 u32;
      u32.as_value = (float)u16.as_value;
      return u32.as_bits>>16;
    }
  }
  else{
    uf32 u32;
    u32.as_value = uint82float(input);
    if constexpr(std::is_same<scalar_t, _Float16>::value){
      return (_Float16)(u32.as_value);
    }
    else{
      return (uint16_t)(u32.as_bits >> 16);
    }
  }
}
template <bool is_e4m3>
static __device__ int to_f8_from_f32(float v1,float v2,float v3,float v4) {
  int val=0;
  #if (defined(__gfx938__) )
  if constexpr(is_e4m3){
    val = __builtin_hcu_cvt_pk_fp8_f32(v1,v2,val,false);
    val = __builtin_hcu_cvt_pk_fp8_f32(v3,v4,val,true);
  }
  else{
    val = __builtin_hcu_cvt_pk_bf8_f32(v1,v2,val,false);
    val = __builtin_hcu_cvt_pk_bf8_f32(v3,v4,val,true);
  }
  #endif
  return val;
}

template <bool is_e4m3>
static __device__ float4_t to_fp32_from_fp8(int val) {
  float4_t ret;
  #if (defined(__gfx938__) )
  if constexpr(is_e4m3){
    ret[0] = __builtin_hcu_cvt_f32_fp8(val,false,0,0);
    ret[1] = __builtin_hcu_cvt_f32_fp8(val,false,0,1);
    ret[2] = __builtin_hcu_cvt_f32_fp8(val,false,0,2);
    ret[3] = __builtin_hcu_cvt_f32_fp8(val,false,0,3);
  }
  else{
    ret[0] = __builtin_hcu_cvt_f32_bf8(val,false,0,0);
    ret[1] = __builtin_hcu_cvt_f32_bf8(val,false,0,1);
    ret[2] = __builtin_hcu_cvt_f32_bf8(val,false,0,2);
    ret[3] = __builtin_hcu_cvt_f32_bf8(val,false,0,3);
  }
  #endif
  return ret;
}

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define Output_Type_SWITCH(SRC_DTYPE, ...)     \
  [&] {                                       \
    if (SRC_DTYPE == at::ScalarType::Half) {  \
      using scalar_t=_Float16;                \
      return __VA_ARGS__();                   \
    }else {                                   \
      using scalar_t=uint16_t;                \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define Input_Type_SWITCH(scalar_t,qdtype,kdtype,...)   \
  [&] {                                                 \
    if(qdtype==torch::kFloat8_e5m2){                    \
      constexpr bool is_e4m3=false;                     \
      using q_type = uint8_t;                           \
      return __VA_ARGS__();                             \
    }else if(qdtype==torch::kFloat8_e4m3fn){            \
      constexpr bool is_e4m3=true;                      \
      using q_type = uint8_t;                           \
      return __VA_ARGS__();                             \
    }else if(kdtype==torch::kFloat8_e5m2){              \
      constexpr bool is_e4m3=false;                     \
      using q_type = scalar_t;                          \
      return __VA_ARGS__();                             \
    }else{                                              \
      constexpr bool is_e4m3=true;                      \
      using q_type = scalar_t;                          \
      return __VA_ARGS__();                             \
    }                                                   \
  }()

#define REUSEKV_SWITCH(reusekv,...)                     \
[&] {                                                   \
    if (reusekv==48){                                   \
        constexpr static int REUSE_KV_TIMES = 48;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==36){                             \
        constexpr static int REUSE_KV_TIMES = 36;       \
        return __VA_ARGS__();                           \
        }else if (reusekv==32){                         \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==24){                             \
        constexpr static int REUSE_KV_TIMES = 24;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==16){                             \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==8){                              \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    }else                {                              \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()

#define HEADSIZE_SWITCH(headsize,...)                   \
[&] {                                                   \
    if (headsize==64){                                  \
        constexpr static int HEAD_SIZE = 64;            \
        return __VA_ARGS__();                           \
    }else if(headsize==128){                            \
        constexpr static int HEAD_SIZE = 128;           \
        return __VA_ARGS__();                           \
    }else if(headsize==192){                            \
        constexpr static int HEAD_SIZE = 192;           \
        return __VA_ARGS__();                           \
    }else {                                             \
        constexpr static int HEAD_SIZE = 256;           \
        return __VA_ARGS__();                           \
    }                                                   \
}()

static std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    return raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.
}

static const std::string device_name=get_device_name();

static inline int get_env_(const char *env_var) {
  if (char *value = std::getenv(env_var)) {
    return atoi(value);
  }
  return 0;
}

static const int PA_USE_STREAMK = get_env_("PA_USE_STREAMK");
static const int PA_MAX_BLOCKS = get_env_("PA_MAX_BLOCKS");
static const int PA_PRINT_PARAM = get_env_("PA_PRINT_PARAM");
static const int PA_PARTITION_SIZE = get_env_("PA_PARTITION_SIZE");

template<int vec> 
struct half4vec{
  half4_t data[vec];
};
using half4x2 = half4vec<2>;
using half4x4 = half4vec<4>;

template<int vec> 
struct int2vec{
  intx2 data[vec];
};

template<int vec> 
struct uint8x4vec{
  uint8x4_t data[vec];
};

using uint8x4x2 = uint8x4vec<2>;
using uint8x4x4 = uint8x4vec<4>;

template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
  int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }
  if (lane == 0) {
    red_smem[warp] = sum;
  }
  __syncthreads();
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }
  return __shfl(sum, 0);
}

template<bool is_e4m3>
inline __device__ void builtin_amdgcn_mmac(const intx2& reg_a, const intx2& reg_b, float4_t& reg_c)
{
  #if (defined(__gfx938__) )
  if constexpr(is_e4m3){
    reg_c=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(reg_a,reg_b,reg_c,false,false);
  }else{
    reg_c=__builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(reg_a,reg_b,reg_c,false,false);
  }
  #endif
}
  
template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const q_type* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const uint8_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const uint8_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili) {  // ★ Attention Sinks: [num_heads] scalar_t ★
#if (defined(__gfx938__) )
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride=BLOCK_SIZE*HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if(num_partitions<=partition_idx)return ;
  constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
  constexpr bool q_is_fp8 = std::is_same<q_type, uint8_t>::value;
  constexpr float scale = (HEAD_SIZE==64?0.125f:(HEAD_SIZE==128? 0.0883883476f:(HEAD_SIZE==192?0.0721687836f:0.0625f)))*1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  float k_scale=scale;
  float v_scale=1.0;
  float q_scale=1.0;
  if(k_scale_ptr!=nullptr){
    k_scale*=(*k_scale_ptr);
  }
  if(q_scale_ptr!=nullptr){
    q_scale=*q_scale_ptr;
  }
  if(v_scale_ptr!=nullptr){
    v_scale=*v_scale_ptr;
  }
  k_scale*=q_scale;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = blockIdx.x;
  const int head_idx=num_queries_per_kv/mtp * kv_head_idx;

  constexpr int reuse_group=(REUSE_KV_TIMES-1)/4+1;
  constexpr int Mloop=(REUSE_KV_TIMES-1)/16+1;
  extern __shared__ char shared_mem[];
  scalar_t* logits = reinterpret_cast<scalar_t*>(shared_mem);
  float* s_max = reinterpret_cast<float*>(shared_mem + sizeof(scalar_t)*num_queries_per_kv*PARTITION_SIZE);
  float* s_logit = s_max + num_queries_per_kv * NUM_WARPS;
  float* max_out = s_logit+NUM_WARPS;
  float* expsum_out = max_out+num_queries_per_kv;
  // ★ Attention Sinks: load s_aux to shared memory ★
  __shared__ scalar_t smem_s_aux[64];
  if (s_aux_ptr != nullptr) {
    if (thread_idx < num_heads) {
      smem_s_aux[thread_idx] = s_aux_ptr[thread_idx];
    }
    __syncthreads();
  }
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const q_type* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  
  float alibi_slope[reuse_group]={0.f};
  if (has_abili){
    for(int i=0;i<reuse_group;i++){
      int reuse_kv_idx=rows+i*4;
      if(reuse_kv_idx<num_queries_per_kv) alibi_slope[i]=alibi_slopes[head_idx+reuse_kv_idx]*1.4426950408889634;
    }
  }
  float qk_max[reuse_group];
  for(int i=0;i<reuse_group;i++){
    qk_max[i]=-FLT_MAX;
  }
  intx4 q_vec[Mloop][HEAD_SIZE/64];
  q_type* s_q = reinterpret_cast<q_type*>(shared_mem);
  {
    int head_offset = HEAD_SIZE*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE;i+=NUM_THREADS*8){
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;

      if constexpr (q_is_fp8){
        *reinterpret_cast<intx2*>(s_q+i)=*reinterpret_cast<const intx2*>(q_ptr+qoffset);
      }
      else{
        *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
      }
    }
  }
  __syncthreads();
  for(int m=0;m<Mloop;m++){
    for(int i=0;i<HEAD_SIZE/64;i++){
      int head_idx_=rowid+16*m;
      if(head_idx_<num_queries_per_kv) {
        if constexpr(q_is_fp8){
          q_vec[m][i]=*reinterpret_cast<const intx4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
        }
        else{
          auto q_temp = *reinterpret_cast<const half4x4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
          scalar_t *q_temp_ptr=(scalar_t*)&q_temp;
          q_vec[m][i][0]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[0])/q_scale,to_float(q_temp_ptr[1])/q_scale,to_float(q_temp_ptr[2])/q_scale,to_float(q_temp_ptr[3])/q_scale);
          q_vec[m][i][1]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[4])/q_scale,to_float(q_temp_ptr[5])/q_scale,to_float(q_temp_ptr[6])/q_scale,to_float(q_temp_ptr[7])/q_scale);
          q_vec[m][i][2]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[8])/q_scale,to_float(q_temp_ptr[9])/q_scale,to_float(q_temp_ptr[10])/q_scale,to_float(q_temp_ptr[11])/q_scale);
          q_vec[m][i][3]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[12])/q_scale,to_float(q_temp_ptr[13])/q_scale,to_float(q_temp_ptr[14])/q_scale,to_float(q_temp_ptr[15])/q_scale);
        }
      }
      else q_vec[m][i]={0,0,0,0};
    }
  }
  __syncthreads();

  const int start_block_idx = partition_idx * PARTITION_SIZE / BLOCK_SIZE;
  const int end_block_idx =MIN(start_block_idx + PARTITION_SIZE / BLOCK_SIZE, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //compute q*k
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*HEAD_SIZE;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          intx4 k_vec=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*HEAD_SIZE+rows*16);
          intx2 *k_vec_2 = (intx2*)&k_vec;
          for(int m=0;m<Mloop;m++){
            intx2 *q_vec_2 = (intx2*)(&q_vec[m][i]);
            builtin_amdgcn_mmac<is_e4m3>(k_vec_2[0],q_vec_2[0],qk_vec[m]);
            builtin_amdgcn_mmac<is_e4m3>(k_vec_2[1],q_vec_2[1],qk_vec[m]);
          }
        }
        #pragma unroll
        for(int i=0;i<reuse_group;i++){
          int reuse_kv_idx=rows+i*4;
          int m = reuse_kv_idx/16;
          int ii = i%4;
          if(reuse_kv_idx<num_queries_per_kv){
            qk_vec[m][ii]*=k_scale;
            const int token_idx = block_idx * BLOCK_SIZE+rowid + b;
            if (has_abili){
              float alibi=alibi_slope[i] * (token_idx - seq_len + 1);
              qk_vec[m][ii] += alibi;
            }
            if(token_idx >= seq_len) {
              int seq_len_pad=DIVIDE_ROUND_UP(seq_len,8)*8;
              if(token_idx<seq_len_pad) from_float(logits[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx],-INFINITY);
              else logits[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx]=0;
            }
            else{
              scalar_t temp;
              if (mtp>1){
                int casual = mtp - reuse_kv_idx * mtp / num_queries_per_kv ;
                if(token_idx+casual>seq_len)qk_vec[m][ii]=-INFINITY;
              }
              from_float(temp,qk_vec[m][ii]);
              logits[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              qk_max[i] = fmaxf(qk_max[i], to_float(temp));
              // if(partition_idx==0)printf("tid=%d,tokenid=%d,reuse_kv_idx=%d,m=%d,ii=%d,qk=%f\n",thread_idx,token_idx,reuse_kv_idx,m,i,qk_vec[m][ii]);
            }
          }
        }
      }
    }
  }
  // compute max
  #pragma unroll
  for (int mask = 8; mask >= 1; mask /= 2) {
    #pragma unroll
    for(int r=0;r<reuse_group;r++){
      qk_max[r]=fmaxf(qk_max[r],__shfl_xor(qk_max[r],mask));
    }
  }
  #pragma unroll
  for(int r=0;r<reuse_group;r++){
    if(rowid==0&&r*4+rows<num_queries_per_kv){
      s_max[(r*4+rows)*NUM_WARPS+warp_idx] = qk_max[r];
    }
  }
  __syncthreads();

  if(PARTITION_SIZE==256){
    for(int lineid = warp_idx;lineid<REUSE_KV_TIMES/2;lineid+=NUM_WARPS){
      int half_lane = lane%32;
      int which_half = lane/32;
      int real_line=lineid*2+which_half;
      if(real_line<num_queries_per_kv){
        float qk_max_tmp;
        float exp_sum=0;
        if(half_lane==0){
          int smax_offset = real_line*4;
          qk_max_tmp=s_max[smax_offset];
          for(int i=1;i<4;i++){
            qk_max_tmp=fmaxf(qk_max_tmp,s_max[smax_offset+i]);
          }
        }
        qk_max_tmp=__shfl(qk_max_tmp,which_half*32);
        int seq_len_pad = DIVIDE_ROUND_UP(num_tokens,8);
        using f16x8_t = __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(half_lane<seq_len_pad){
          f16x8_t logit16 = *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii])-qk_max_tmp);
            exp_sum+=logit32[ii];
          }
          // printf("tid=%d,logit32=%.4f,%.4f,%.4f,%.4f, %.4f,%.4f,%.4f,%.4f\n",thread_idx,logit32[0],logit32[1],logit32[2],logit32[3],logit32[4],logit32[5],logit32[6],logit32[7]);
        }
        for (int mask = 16; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(half_lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8)=logit16;
          if(num_partitions>1&&half_lane==0){
            max_out[real_line] = qk_max_tmp;
            expsum_out[real_line] = exp_sum;
          }
        }
      }
    }
  }
  else if(PARTITION_SIZE==512){
    for(int lineid = warp_idx;lineid<num_queries_per_kv;lineid+=NUM_WARPS){
      if(lineid<num_queries_per_kv){
        float qk_max_tmp;
        float exp_sum=0;
        if(lane==0){
          int smax_offset = lineid*4;
          qk_max_tmp=s_max[smax_offset];
          for(int i=1;i<4;i++){
            qk_max_tmp=fmaxf(qk_max_tmp,s_max[smax_offset+i]);
          }
        }
        qk_max_tmp=__shfl(qk_max_tmp,0);
        int seq_len_pad = DIVIDE_ROUND_UP(num_tokens,8);
        using f16x8_t = __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(lane<seq_len_pad){
          f16x8_t logit16 = *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii])-qk_max_tmp);
            exp_sum+=logit32[ii];
          }
          // printf("tid=%d,logit32=%.4f,%.4f,%.4f,%.4f, %.4f,%.4f,%.4f,%.4f\n",thread_idx,logit32[0],logit32[1],logit32[2],logit32[3],logit32[4],logit32[5],logit32[6],logit32[7]);
        }
        for (int mask = 32; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8)=logit16;
          if(num_partitions>1&&lane==0){
            max_out[lineid] = qk_max_tmp;
            expsum_out[lineid] = exp_sum;
          }
        }
      }
    }
  }

  __syncthreads();
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE, 16*NUM_WARPS);//2
  constexpr int GROUPS=reuse_group*4;
  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float4_t accs[Mloop][NUM_ROWS_PER_THREAD];
  for(int m=0;m<Mloop;m++){
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[m][i] = {0.f,0.f,0.f,0.f};
    }
  }
  constexpr int vecsize=BLOCK_SIZE/32;//2
  using int_vec = int2vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    intx2 logits_vec[Mloop][vecsize];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m][i]={0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        for(int k=0;k<vecsize;k++){
          auto l_temp = *reinterpret_cast<half8_t*>(logits + real_row * PARTITION_SIZE+token_idx - start_token_idx + k*8);
          scalar_t *l_temp_ptr=(scalar_t*)&l_temp;
          logits_vec[m][k][0]=to_f8_from_f32<is_e4m3>(to_float(l_temp_ptr[0]),to_float(l_temp_ptr[1]),to_float(l_temp_ptr[2]),to_float(l_temp_ptr[3]));
          logits_vec[m][k][1]=to_f8_from_f32<is_e4m3>(to_float(l_temp_ptr[4]),to_float(l_temp_ptr[5]),to_float(l_temp_ptr[6]),to_float(l_temp_ptr[7]));
        }
      }
    }
    const uint8_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                          kv_head_idx * kv_head_stride;
    if(partition_idx<num_partitions-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*16+rowid*BLOCK_SIZE;
        int_vec v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*16+rowid*BLOCK_SIZE;
        int_vec v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
        if (block_idx == num_seq_blocks - 1) {
          uint8_t* v_vec_ptr = reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for (int j = 0; j < 16; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : 0;
          }
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
  }
  {
    scalar_t* out_ptr_base;
    int out_offset;
    if(num_partitions>1){
      out_offset=max_num_partitions*HEAD_SIZE;
      out_ptr_base=out_tmp+out_tmp_offset + seq_idx * num_heads * out_offset + head_idx*out_offset+partition_idx * HEAD_SIZE;
    }
    else{
      out_offset=HEAD_SIZE;
      out_ptr_base=out + seq_idx * num_heads  * HEAD_SIZE + head_idx*HEAD_SIZE;
    } 
    int head_offset = num_queries_per_kv/mtp;
    for(int g=0;g<reuse_group;g++){
      int reusekvid=g*4+rows;
      if(reusekvid<num_queries_per_kv){
        int out_head = reusekvid/head_offset*num_kv_heads*head_offset + reusekvid%head_offset;
        scalar_t* out_ptr = out_ptr_base + out_head*out_offset;
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx = rowid+16*warp_idx + i * WARP_SIZE;
          from_float(*(out_ptr + row_idx), accs[reusekvid/16][i][g%4]*v_scale);
          // if(reusekvid==0)printf("patition=%d,tid=%d,i=%d,g=%d,acc=%f\n",partition_idx,thread_idx,i,g,accs[i][g]);
        }
      }
    }
    if (num_partitions>1&&thread_idx < num_queries_per_kv){
      int out_head = thread_idx/head_offset*num_kv_heads*head_offset + thread_idx%head_offset;
      int offset = seq_idx * num_heads * max_num_partitions + (head_idx+out_head) * max_num_partitions + partition_idx;
      float * exp_sums=reinterpret_cast<float*>(out_tmp);
      float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
      *(exp_sums+offset)=expsum_out[thread_idx];
      *(max_logits+offset)=max_out[thread_idx];
    }
  }
#endif
}

template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS, 1) void paged_attention_combine(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    scalar_t* out_tmp,  // [num_seqs, num_heads,
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions,
    int num_heads,
    int PARTITION_SIZE) {
  extern __shared__ char shared_mem[];
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  const int lane = threadIdx.x;
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if(num_partitions==1)return;
  float* shared_exp_sums=reinterpret_cast<float*>(shared_mem);
  float* shared_max_logits=shared_exp_sums+num_partitions;
  float max_logit = -FLT_MAX;
  float global_exp_sum = 0.0f;
  int offset = seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  const float * exp_sums=reinterpret_cast<float*>(out_tmp);
  const float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
  const float* max_logits_ptr = max_logits + offset;
  const float* exp_sums_ptr = exp_sums + offset;
  const scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  const scalar_t* tmp_out_ptr = out_tmp + out_tmp_offset + offset* HEAD_SIZE;
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l; 
    max_logit = fmaxf(max_logit,l);
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    float rescaled_exp_sum = exp_sums_ptr[i] * __builtin_amdgcn_exp2f(shared_max_logits[i] - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    global_exp_sum += __shfl_xor(global_exp_sum, mask);
  }
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);
  constexpr int vec_size_o=HEAD_SIZE/64;
  constexpr int vec_size = vec_size_o==3?4:vec_size_o;
  using half_vec= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  using float_vec= __attribute__( (__vector_size__(vec_size * sizeof(float)) )) float;
  float_vec acc = {0.0f};
  half_vec acc_half;
  if(lane<HEAD_SIZE/vec_size){
    for (int j = 0; j < num_partitions; ++j) {
      half_vec tout= *(half_vec*)(tmp_out_ptr + j * HEAD_SIZE + lane * vec_size);
      float temp_sum=shared_exp_sums[j]*inv_global_exp_sum;
      #pragma unroll
      for(int i=0;i<vec_size;i++){
        acc[i] += to_float(tout[i])*temp_sum;
      }
    }
    #pragma unroll
    for(int i=0;i<vec_size;i++){
      scalar_t temp;
      from_float(temp,acc[i]);
      acc_half[i]=temp;
    }
    *(half_vec*)(out_ptr+lane*vec_size)=acc_half;
  }
}

static int get_reusekv(int qhead,int kv_head){
  if(qhead>kv_head*36) return 48;
  if(qhead>kv_head*32) return 36;//glm4.7 mtp 3
  if(qhead>kv_head*24) return 32;
  if(qhead>kv_head*16) return 24;
  if(qhead>kv_head*8) return 16;
  if(qhead>kv_head*4)return 8;
  return  4;
}

void paged_attention_938(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    float *tmp_out_ptr,
    int PARTITION_SIZE)  // ★ Attention Sinks ★
{
  int max_num_blocks_per_seq = block_tables.size(1);
  int num_seqs = query.size(0);
  int mtp = query.size(1);
  int block_size=key_cache.size(2);
  int num_heads = query.size(2)*mtp;
  int num_kv_heads = key_cache.size(1);
  int head_size = query.size(3);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);
  int num_blocks=key_cache.size(0);
  const float* alibi_slopes_ptr =alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr()):nullptr;
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  auto* out_ptr =  out.data_ptr();
  const float* q_scale_ptr = q_scale? reinterpret_cast<const float*>(q_scale.value().data_ptr()):nullptr;
  const float* k_scale_ptr = k_scale? reinterpret_cast<const float*>(k_scale.value().data_ptr()):nullptr;
  const float* v_scale_ptr = v_scale? reinterpret_cast<const float*>(v_scale.value().data_ptr()):nullptr;

  // Attention Sinks: validate and set s_aux_ptr
  const void* s_aux_ptr = nullptr;
  if (s_aux_.has_value()) {
    auto s_aux = s_aux_.value();
    // ★ s_aux must match Q/K/V dtype (Element type) for mixed precision
    TORCH_CHECK(s_aux.dtype() == query.dtype(),
                "s_aux must have the same dtype as query. Got s_aux dtype: ", s_aux.dtype(),
                ", query dtype: ", query.dtype());
    TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                "s_aux must have dtype float16 or bfloat16 (to match query). Got: ", s_aux.dtype());
    TORCH_CHECK(num_heads <= 64,
                "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
    CHECK_DEVICE(s_aux);
    CHECK_SHAPE(s_aux, num_heads);
    CHECK_CONTIGUOUS(s_aux);
    s_aux_ptr = s_aux.data_ptr();
  }
  auto* query_ptr = query.data_ptr();
  auto* key_cache_ptr = key_cache.data_ptr();
  auto* value_cache_ptr = value_cache.data_ptr();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 reduce_grid(num_heads, num_seqs);
 
  dim3 grid;
  grid.x = num_kv_heads;
  grid.y = num_seqs;
  int reusekv=get_reusekv(num_heads,num_kv_heads);
  int headsize=query.size(3);
  AT_ASSERTM(headsize%64==0 && headsize<=256, "Page Attention head size must be 64, 128, 192 or 256");
  AT_ASSERTM(num_heads<=num_kv_heads*48, "Page Attention qheads*mtp/kvheads must be smaller than 48");
  HEADSIZE_SWITCH(headsize,[&]{
    Output_Type_SWITCH(out.dtype(),[&]{
      Input_Type_SWITCH(scalar_t,query.dtype(),key_cache.dtype(),[&] {
        REUSEKV_SWITCH(reusekv,[&] {
          BOOL_SWITCH(block_size==64,is_block64,[&]{
            constexpr int BLOCK_SIZE = (is_block64?64:128);
              // constexpr int HEAD_SIZE=128;
              // using scalar_t=uint16_t;
              // constexpr bool is_e4m3=true;
              // constexpr static int REUSE_KV_TIMES = 4;
              // constexpr bool has_abili=false;
              // constexpr bool use_mtp=false;
              constexpr static int NUM_THREADS = 256;
              constexpr static int NUM_WARPS = NUM_THREADS / WARP_SIZE;
              int real_reuse_times = num_heads/num_kv_heads;
              int other_use = (real_reuse_times*NUM_WARPS+NUM_WARPS+ real_reuse_times*2)*sizeof(float);
              int shared_mem_size=PARTITION_SIZE*sizeof(scalar_t)*real_reuse_times+other_use;
              int max_num_partitions=DIVIDE_ROUND_UP(max_seq_len,PARTITION_SIZE);
              grid.z = max_num_partitions;
              dim3 block(NUM_THREADS);
              if(PA_PRINT_PARAM)printf("sizeof(q)=%d,shared_mem_size=%d,HEAD_SIZE=%d,num_thread=%d,grid={%d,%d,%d},qhead=%d,kvhead=%d,seq=%d,batch=%d,PARTITION_SIZE=%d,max_num_partitions=%d\n",
                                        sizeof(q_type),shared_mem_size,HEAD_SIZE,NUM_THREADS,grid.x,grid.y,grid.z,num_heads,num_kv_heads,max_seq_len,num_seqs,PARTITION_SIZE,max_num_partitions);
              paged_attention_kernel<scalar_t,q_type,is_e4m3,HEAD_SIZE,BLOCK_SIZE,NUM_THREADS,REUSE_KV_TIMES><<<grid,block,shared_mem_size,stream>>>(
                (scalar_t*)out_ptr,(scalar_t*)tmp_out_ptr, (q_type*)query_ptr,(uint8_t*) key_cache_ptr, (uint8_t*)value_cache_ptr,
                num_heads, num_kv_heads, block_tables_ptr, seq_lens_ptr,max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride,
                q_scale_ptr,k_scale_ptr, v_scale_ptr,max_num_partitions,PARTITION_SIZE,(const scalar_t*)s_aux_ptr,mtp,alibi_slopes_ptr!=nullptr);
              if(max_num_partitions>1){
                paged_attention_combine<scalar_t,HEAD_SIZE,64><<<dim3(num_heads,num_seqs),64,4*2*max_num_partitions,stream>>>(
                  (scalar_t*)out_ptr,(scalar_t*)tmp_out_ptr,seq_lens_ptr,max_num_partitions,num_heads,PARTITION_SIZE);
              }
          });
        });
      });
    });
  });
}
 
