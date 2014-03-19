/**
 * @file backend.cu
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <iostream>
#include "backend/cuda/backend.h"

namespace VECGEOM_NAMESPACE {

cudaError_t CudaCheckError(const cudaError_t err) {
  if (err != cudaSuccess) {
    std::cout << "CUDA reported error with message: \""
              << cudaGetErrorString(err) << "\"\n";
  }
  return err;
}

cudaError_t CudaCheckError() {
  return CudaCheckError(cudaGetLastError());
}

void CudaAssertError(const cudaError_t err) {
  assert(CudaCheckError(err) == cudaSuccess);
}

void CudaAssertError() {
  CudaAssertError(cudaGetLastError());
}

void* AllocateOnGpu(const int size) {
  void *ptr;
  CudaAssertError(cudaMalloc(&ptr, size));
  return ptr;
}

} // End global namespace