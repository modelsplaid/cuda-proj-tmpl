add_library(cudaCommon "")
target_sources(cudaCommon PRIVATE
  my_cuda_utils.cu
  OpenMPTimer.cpp
  SimpleTimer.cpp)

target_include_directories(cudaCommon
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

##########tzq
target_include_directories(cudaCommon
  PUBLIC
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/include"
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/tools/util/include"
)

list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})

list(APPEND NVCC_FLAGS "-O3")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")
list(APPEND NVCC_FLAGS "-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
list(APPEND NVCC_FLAGS "-expt-relaxed-constexpr")
list(APPEND NVCC_FLAGS "--use_fast_math")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++17)

########

target_compile_options(cudaCommon PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(cudaCommon PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

add_library(cuda::common ALIAS cudaCommon)
