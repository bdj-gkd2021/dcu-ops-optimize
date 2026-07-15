# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=3328 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1664 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1664 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=3328 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1664 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1664 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=3328 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1664 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1664 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=832 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=3328 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1664 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1664 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=832 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=2816 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=1408 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=1408 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=704 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=2816 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=1408 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=1408 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=704 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=2816 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=1408 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=1408 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=704 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=2816 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2560 --n=1408 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=1408 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1280 --n=704 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1232 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=3072 --n=3152 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1232 --k=1536 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1232 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1232 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=16 --k=16 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=16 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=16 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3152 --k=2304 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3152 --k=3072 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3152 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=512 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=3072 --n=768 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1536 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=2048 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=16 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=768 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=2304 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3072 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=768 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1536 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=16 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2304 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=3072 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1232 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3152 --k=3072 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=768 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=12288 --k=1536 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=6144 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=12288 --k=4608 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=12288 --k=4608 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=1536 --k=12288 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=12288 --k=6144 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=2048 --n=6144 --k=12288 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=12288 --n=4608 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1536 --n=12288 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=12288 --n=6144 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=256 --n=256 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=512 --n=512 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=6144 --n=12288 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=6144 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=12288 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
# ./tools/profiler/cutlass_profiler --kernels=hgemm --m=32768 --n=16384 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column

./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=3328 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1664 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1664 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=3328 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1664 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1664 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=832 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=3328 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1664 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1664 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=832 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=3328 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1664 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1664 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=832 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=2816 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=1408 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=1408 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=704 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=2816 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=1408 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=1408 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=704 --k=4096 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=2816 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=1408 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=1408 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=704 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=2816 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2560 --n=1408 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=1408 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1280 --n=704 --k=4096 --A=f16:row --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1232 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=3072 --n=3152 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1232 --k=1536 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1232 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1232 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=16 --k=16 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=16 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=16 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3152 --k=2304 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3152 --k=3072 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3152 --k=768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=512 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=3072 --n=768 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1536 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=2048 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=1232 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=16 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=768 --k=16 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=2304 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3072 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=768 --k=3152 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1536 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=16 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2304 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=3072 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1232 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=1232 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=16 --k=512 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3152 --k=3072 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=768 --n=3152 --k=768 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=4608 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=12288 --k=1536 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=6144 --k=12288 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=12288 --k=4608 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=12288 --k=4608 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=1536 --k=12288 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=12288 --k=6144 --A=f16:column --B=f16:row --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=2048 --n=6144 --k=12288 --A=f16:column --B=f16:row --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=12288 --n=4608 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1536 --n=12288 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=12288 --n=6144 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=2048 --A=f16:row --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column 
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=128 --n=128 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=256 --n=256 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=512 --n=512 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=6144 --n=12288 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=6144 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=12288 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=128 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=256 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=512 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=1024 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=2048 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=4096 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=8192 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=16384 --A=f16:column --B=f16:column --C=f16:column --D=f16:column
./tools/profiler/cutlass_profiler --kernels=sgemm --m=32768 --n=16384 --k=32768 --A=f16:column --B=f16:column --C=f16:column --D=f16:column

