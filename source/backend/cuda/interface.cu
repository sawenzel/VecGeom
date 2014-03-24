/**
 * @file interface.cu
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <cassert>
#include <iostream>
 
#include "backend/cuda/interface.h"
#include "volumes/placed_volume.h"
#include "volumes/logical_volume.h"

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

// Class specific functions

__global__
void CudaManagerPrintGeometryKernel(
    vecgeom_cuda::VPlacedVolume const *const world) {
  world->logical_volume()->PrintContent();
}

void CudaManagerPrintGeometry(VPlacedVolume const *const world) {
  CudaManagerPrintGeometryKernel<<<1, 1>>>(
    reinterpret_cast<vecgeom_cuda::VPlacedVolume const*>(world)
  );
  CudaAssertError();
  cudaDeviceSynchronize();
}

} // End namespace vecgeom