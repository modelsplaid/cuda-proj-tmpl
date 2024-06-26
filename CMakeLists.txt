# cmake 3.22 is necessary to have FindCUDAToolkit.cmake
# actually find cublas library when using nvc++ compiler (from nvhpc
# package)
cmake_minimum_required(VERSION 3.18)

#
# check if we want to apply cmake policy CMP0104, i.e.
# raise an error if CUDA_ARCHITECTURES is empty, when using
# nvc++ compiler
#
message("Using CMake version ${CMAKE_VERSION}")
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("Using cmake policy CMP0104 OLD")
    cmake_policy(SET CMP0104 OLD)
else()
    message("Using cmake policy CMP0104 NEW")
    cmake_policy(SET CMP0104 NEW)
endif()

message("******CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

project(cuda-proj-tmpl LANGUAGES CXX C)

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  if(${CMAKE_VERSION} VERSION_LESS "3.22.0")
    message(FATAL_ERROR "NVHPC requires cmake version >= 3.22.0")
  endif()
endif()

###### tzq added
set(CMAKE_CUDA_STANDARD 17)
######

# custom cmake macros location

message("****** INSERT CMAKE_MODULE_PATH  to find cuda libs cmake")

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("${CMAKE_MODULE_PATH}")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")
message("${CMAKE_MODULE_PATH}")

message("****** Done insert")

# Prevent from build in source tree
include(preventBuildInSource)
#prevent_build_in_source()

# Init build type: Release, Debug, ...
include(initBuildType)

# useful for debugging cmake
include(print_target_properties)

# a simple macro to build executable avoiding copy paste
#include(build_macro)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#####################################################################
enable_language(CUDA)

# FindCUDAToolkit.cmake is able to correctly find cuda library
# either when using nvcc from cuda toolkit or nvc++ from nvhpc
# only in cmake 3.22.0
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.22.0" )
  message("Using CUDAToolkit macros")
  find_package(CUDAToolkit REQUIRED)
else()
  message("Using CUDALibs macros")
  find_package(CUDALibs REQUIRED COMPONENTS CUBLAS)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# other third party libraries
find_package(OpenMP)

# we also need extra stuff to make sure compile flags are correctly
# passed to nvcc / host compiler
include(tzq_test)
include(protect_nvcc_flags)
include(protect_pthread_flag)


##########################################################

# build some executable
message("******** add subdir src")
add_subdirectory(src)
message("******** done add subdir src")
# Testing
#enable_testing()

#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("// ${PROJECT_NAME} version : ${PROJECT_VERSION}")
message("//===================================================")
message("  CMake version          : ${CMAKE_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CXX      STANDARD  : ${CMAKE_CXX_STANDARD}")
message("  CUDA CXX STANDARD  : ${CMAKE_CUDA_STANDARD}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("")
message("  OpenMP compile flags : ${OpenMP_CXX_FLAGS}")
message("")
#message(" Thrid party libraries")
#message("  OpenGL found : ${OPENGL_FOUND}")
