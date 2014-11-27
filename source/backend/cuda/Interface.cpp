/// \file Interface.cu
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 
#include "backend/cuda/Interface.h"

#include <cassert>
#include <stdio.h>

#include <cuda.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

// maybe you need also helpers
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper utility functions

namespace vecgeom {
namespace cxx {

cudaError_t CudaCheckError(const cudaError_t err) {
  if (err != cudaSuccess) {
    printf("CUDA reported error with message: \"%s\"\n",
           cudaGetErrorString(err));
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

cudaError_t CudaMalloc(void** ptr, unsigned size) {
  return cudaMalloc(ptr, size);
}

cudaError_t CudaCopyToDevice(void* tgt, void const* src, unsigned size) {
  return cudaMemcpy(tgt, src, size, cudaMemcpyHostToDevice);
}

cudaError_t CudaCopyFromDevice(void* tgt, void const* src, unsigned size) {
  return cudaMemcpy(tgt, src, size, cudaMemcpyDeviceToHost);
}

cudaError_t CudaFree(void* ptr) {
  return cudaFree(ptr);
}

} // End namespace cuda

} // End namespace vecgeom
