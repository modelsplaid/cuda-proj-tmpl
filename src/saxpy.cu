/**
 * Compute saxpy
 * - on CPU : serial and OpenMP version
 * - on GPU : first using CUDA, then library CuBLAS
 *
 * compare timings.
 *
 */

// =========================
// standard imports
// =========================
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// =========================
// CUDA imports
// =========================
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/array.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>

#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include "tzq_cuda.h"

#include "flash.h"
#include "kernel_traits.h"
#include "flash_fwd_kernel.h"
#include "flash_fwd_launch_template.h"
#include "static_switch.h"

#include "flash_api.cu"

void ds_kernel(){
  // todo: here
  Flash_fwd_params *ffp;
  cudaStream_t stream;
  //Flash_
  using elem_type = cutlass::half_t;
  //run_mha_fwd(*ffp,stream);

}

// =========================
// main routine
// =========================
int main (int argc, char **argv)
{

  ds_kernel();
  //gemm();
  //test_type();
  //test_arr();
  //multi_add();
  //test_tensor();
  //tzqtest();
  //test_tensor_gemm();
  //test_type();
  return 0;
}