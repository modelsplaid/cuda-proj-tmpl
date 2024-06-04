
message("******** add subdir common")
add_subdirectory(common)
add_subdirectory(tzq)
add_subdirectory(sum_kernels)
message("******** done add subdir common")

add_executable(atten_cuda "")
target_sources(atten_cuda PRIVATE atten.cu)


target_include_directories(atten_cuda
  PUBLIC
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/include"
  "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/third_party/cutlass/tools/util/include"
)


# the following line shouldn't be necessary as target_link_libraries
# with OpenMP::OpenMP_CXX should pass the correct flags for OpenMP
# but apparently, this is KO when using nvcc :(
# You can remove that if OpenMP is not required
# checked using cmake 3.14.3 (October 2019)
target_compile_options(atten_cuda PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  --expt-relaxed-constexpr
  )

target_link_libraries(atten_cuda PUBLIC CUDA::cublas)
target_link_libraries(atten_cuda PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(atten_cuda PUBLIC cuda::common)
target_link_libraries(atten_cuda PUBLIC cuda::tzqtest)

target_link_libraries(atten_cuda PUBLIC "/data/tzq/DeepSpeed-Kernels/dskernels/inf_flash_attn/blocked_flash/build/libblockedflash.a")
