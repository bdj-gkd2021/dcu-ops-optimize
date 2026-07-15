#!/bin/bash

if [ ! -d "./build" ]; then
    echo "create build"
    mkdir ./build
fi

cd ./build

# enable hipblas
cmake .. -DCUTLASS_ENABLE_CUBLAS=1

make -j32 VERBOSE=1 && make install