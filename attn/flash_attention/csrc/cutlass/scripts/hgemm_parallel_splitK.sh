./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=2 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=4 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=8 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=16 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=32 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=64 --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 

./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=4 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=8 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=16 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=32 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=64 --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column

./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=4 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=8 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=16 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=32 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=64 --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column

./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=4 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=8 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=16 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=32 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=64 --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column

./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=4 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=8 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=16 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=32 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=f16_sgemm_f16 --split_k_mode=parallel --split_k_slices=64 --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
