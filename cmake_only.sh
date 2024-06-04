rm -rf build
mkdir build
cd build
#cmake -DCMAKE_CUDA_ARCHITECTURES="80"   ..

cmake -DCMAKE_CUDA_ARCHITECTURES="80" -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../
make
#make -j40
