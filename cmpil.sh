#rm -rf build
#mkdir build

cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="80" -DCUTLASS_NVCC_ARCHS=80  ..
make -j40
