This example provides gfx928 cute gemm pipline using cute.

## Quick start

```shell

# build the gemm

    CC_FILES=$(shell find ./ -name "*.cu")

    EXE_FILES=$(CC_FILES:.cu=)

    all:$(EXE_FILES)

    %:%.cu
        hipcc -o $@ $< -O3 -DHIP_ENABLE_WARP_SYNC_BUILTINS -DDCU_ASM -std=c++17  -I../../../include -I./ -I../../../tools/util/include

    clean:
        rm -rf $(EXE_FILES)
```

# Run the gemm with default

./hgemm 256 256 128 N T

```

Current restrictions

This experimental example has the following restrictions:

1. when changing blockShape, global memory tiled shape should be also modified.
2. Matrix A must be column major, matrix B must be row major, matrices C and D must be column major.

```
