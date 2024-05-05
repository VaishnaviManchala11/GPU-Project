LAPACK_PATH:=/apps/x86-64/apps/spack_0.19.1/spack/opt/spack/linux-rocky8-zen3/gcc-11.3.0/netlib-lapack-3.10.1-vyp45bfucjjiiatmvtsi6xrnbymjuidi

tiled_mm: c++_tiled_mm.cu
	nvcc -I../common \
        -I${LAPACK_PATH}/include \
        -L${LAPACK_PATH}/lib64 \
        -Xlinker -rpath=${LAPACK_PATH}/lib64 \
    c++_tiled_mm.cu -o tiled_mm \
    -lcblas

clean:
	rm -f tiled_mm
