#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "tzq_cuda.h"

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



void tzqtest(){
    fprintf(stderr,"tzq test here \n");

}

void test_tensor_gemm(){

  int M = 6;
  int N = 3;
  int K = 2;

  float alpha = 2.0f; //1.5f;
  float beta = -3.0f; //-1.25f;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::reference::host::TensorFill(A.host_view(),1_hf);

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::reference::host::TensorFill(B.host_view(),2_hf);

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> D({M, N});

  cutlass::reference::host::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementA and LayoutA
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementB and LayoutB
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementC and LayoutC
    float,                                           // scalar type (alpha and beta)
    float> gemm_op;                                  // internal accumulation type

  gemm_op(
    {M, N, K},             // problem size
    alpha,                 // alpha scalar
    A.host_view(),         // TensorView to host memory
    B.host_view(),         // TensorView to host memory
    beta,                  // beta scalar
    C.host_view(),         // TensorView to host memory
    D.host_view());        // TensorView to device memory


  cutlass::TensorView<cutlass::half_t, cutlass::layout::ColumnMajor> view=D.host_view();

  std::cout<<"D: \n"<<view;

}

int gemm() {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm80                       // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::half_t const *ptrA = A.device_data();
  cutlass::half_t const *ptrB = B.device_data();
  cutlass::half_t const *ptrC = C.device_data();
  cutlass::half_t       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  return 0;
}

__global__ void kernel(cutlass::half_t x){
  printf("device: %f \n",x*2);
}

void test_type(){
  cutlass::half_t x=0.5_hf;
  //std::cin>>x;
  std::cout<<"Host: "<<2.0_hf*x<<std::endl;
  kernel<<<dim3(1,1),dim3(1,1,1)>>>(x);

  cudaError_t status=cudaDeviceSynchronize();

}


void test_arr(){

  // cutlass::Array<cutlass::half_t, 3> elements;
  cutlass::Array<float, 3> elements;
  elements[0]=3;
  elements[1]=9;
  elements[2]=87;
  CUTLASS_PRAGMA_UNROLL                        // required to ensure array remains in registers
  for (auto x : elements) {
    printf("%d, %f \n", int(x), double(x));   // explictly convert to int64_t or double
  }

}


void multi_add(){
  static int const kN=8;
  using namespace cutlass;
  using namespace std;

  //std::array<float,9> arr;

  // for(std::array<float,9>::iterator itr=arr.begin();itr<arr.end();itr++){
  //   *itr=9;
  //   cout<<"*itr: "<<*itr<<endl;
  // }

  Array<half_t,kN> a;
  Array<half_t,kN> b;
  Array<half_t,kN> c;
  Array<half_t,kN> d;
  
  for(auto itr=a.begin();itr!=a.end();itr++){
    cout<<"*itr: "<<*itr<<endl;
  }
  
  multiply_add<Array<half_t,kN>> mad_op;
  d=mad_op(a,b,c);

  for(auto i :d){

    cout<<"i: "<<i<<endl;
  }

}

void test_tensor(){
  using namespace std;
  using namespace cutlass;

  int rows =3;
  int columns=3;
  float x=3.14;

  // fill 
  HostTensor<float,layout::ColumnMajor> tensor({rows,columns});
  reference::host::TensorFill(tensor.host_view(),x);
  cutlass::TensorView<float, cutlass::layout::ColumnMajor> view = tensor.host_view();
  std::cout<<view<<endl;

  // write at {i,j}
  float idx=0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {

      // Write the element at location {i, j} in host memory
      tensor.host_ref().at({i, j}) = idx;

      idx += 0.5f;
    } 
  }

  // copy host mem to device mem
  tensor.sync_device();

  std::cout<<view<<endl;
  float *device_ptr = tensor.device_data();

}
