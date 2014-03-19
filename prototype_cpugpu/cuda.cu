#include <cassert>
#include "testclass.h"

__global__
void do_work_kernel(int *vector) {
  vector[threadIdx.x] *= 10;
}

void gpu::TestClass::DoWork() {
  int *vector_gpu;
  cudaMalloc(&vector_gpu, 10*sizeof(int));
  cudaMemcpy(vector_gpu, vector, 10*sizeof(int), cudaMemcpyHostToDevice);
  do_work_kernel<<<1, 10>>>(vector_gpu);
  cudaMemcpy(vector, vector_gpu, 10*sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaGetLastError() == cudaSuccess);
}