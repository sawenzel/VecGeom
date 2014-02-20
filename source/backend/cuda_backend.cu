#include "backend/cuda_backend.cuh"

namespace vecgeom {

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

} // End namespace vecgeom