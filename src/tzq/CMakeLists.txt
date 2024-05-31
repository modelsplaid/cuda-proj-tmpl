add_library(tzqTest "")
target_sources(tzqTest PRIVATE
  tzq_cuda.cu)

target_include_directories(tzqTest
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_include_directories(tzqTest
  PUBLIC
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/include"
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/tools/util/include"
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/blocked_flash"
)
#/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/blocked_flash/build/libblockedflash.a

target_compile_options(tzqTest PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(tzqTest PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)


add_library(cuda::tzqtest ALIAS tzqTest)
