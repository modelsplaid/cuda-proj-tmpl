set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
include_directories(/usr/include/python3.7)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

set(SRCS add2_ops.cpp add2_kernel.cu)
add_library(add2 SHARED ${SRCS})

target_sources(add2 PRIVATE
  add2_ops.cpp)

target_include_directories(add2
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_include_directories(add2
  PUBLIC
  "/data/tzq/tzq_cutlass_exp/cuda-proj-tmpl/src/sum_kernels"
)

target_compile_options(add2 PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(add2 PUBLIC OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
endif(OpenMP_CXX_FOUND)

#target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

add_library(cuda::tzqadd2 ALIAS add2)
