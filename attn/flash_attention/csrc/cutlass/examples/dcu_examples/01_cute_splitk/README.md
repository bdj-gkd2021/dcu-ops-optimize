This example provides gfx928 and gfx906 cute gemm splitk serial using cute.

## Quick start

```shell

# build the gemm splitk serial
hipcc -o gfx928_gemm_tensor_op gfx928_gemm_tensor_op.cu -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include -I../../

hipcc -o gfx928_splitk_gemm gfx928_splitk_gemm.cu -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include -I../../

hipcc -o gfx906_gemm_simt gfx906_gemm_simt.cu -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include -I../../


# build the gemm splitk parallel
hipcc -o gfx906_gemm_simt_parallel gfx906_gemm_simt_parallel.cu -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include -I../../

hipcc -o gfx928_splitk_parallel_gemm gfx928_splitk_parallel_gemm.cu -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include -I../../

# Run the gemm with default M = 256 N = 256 K = 32 slice_k = 1
./gfx928_gemm_tensor_op

# Run according to problem size that user wants. eg: M = 128 N = 128 K =128 slice_k = 2
./gfx928_gemm_tensor_op 128 128 128 2 

```

Current restrictions

This experimental example has the following restrictions:

1. when changing blockShape, global memory tiled shape should be also modified.
2. Only FP16 is supported currently
3. Matrix C and D must be column major.

```
在Splitk Parallel mode中，可以根据精度需求自主选择累加中间结果的类型，涉及以下改动：
1. Gemm Kernel中CollectiveEpilogue的ElementC_类型可以设置为高精度类型
2. 如果中间结果使用高精度，则reduction kernel按需设置新的EpilogueOp，EpilogueOp中的ElementOutput_是自主选择的最终输出结果类型；
ReductionOp中的Element_类型需要传入类型与中间结果类型一致

