This example provides gfx928 cute gemm pipline using cute.

## Quick start

```shell

# build the gemm
hipcc -o gfx928_gemm_tensor_op gfx928_gemm_tensor_op.cu -O3  -DHIP_ENABLE_WARP_SYNC_BUILTINS  -DDCU_ASM -std=c++17 -I../ -I../../../include -I../../../tools/util/include 

# Run the gemm with default M = 256  N = 256  K = 32 
./gfx928_gemm_tensor_op

# Run the gemm perf with default M = 4096 N = 4096 K =128 iterations = 1000
./gfx928_gemm_tensor_op Y 1000

```

Current restrictions

This experimental example has the following restrictions:

1. when changing blockShape, global memory tiled shape should be also modified.
2. Only FP16 is supported currently
3. Matrix A must be column major, matrix B must be row major, matrices C and D must be column major.

```
