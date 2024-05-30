rm -rf build
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="80"   ..
make -j40
