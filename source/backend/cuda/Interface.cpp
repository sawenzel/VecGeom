/// \file Interface.cu
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/backend/cuda/Interface.h"

#include <stdio.h>
#include <iostream>

#include <cuda.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

// maybe you need also helpers
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper utility functions

namespace vecgeom {

#ifdef VECCORE_CUDA
namespace cxx {
#else
inline namespace cxx {
#endif

cudaError_t CudaCheckError(const cudaError_t err)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA reported error with message: \"" << cudaGetErrorString(err) << "\"\n";
  }
  return err;
}

/**
 * Retrieve the last cuda error and print it in case of problems.
 * This clears the error state of the cuda API.
 * @return The cuda error state.
 */
cudaError_t CudaCheckError()
{
  return CudaCheckError(cudaGetLastError());
}

/**
 * Assert that `err == cudaSuccess` in debug builds.
 * Print the error and terminate the program if it's not.
 * The function is a no-op in release builds.
 * @return The cuda error state.
 */
void CudaAssertError(const cudaError_t err)
{
  assert(CudaCheckError(err) == cudaSuccess);
}

/**
 * In debug builds, retrieve the last cuda error and assert that it is cudaSuccess.
 * The error will not be cleared in release builds. Use CudaCheckError() for this.
 */
void CudaAssertError()
{
  assert(CudaCheckError(cudaGetLastError()) == cudaSuccess);
}

cudaError_t CudaMalloc(void **ptr, unsigned size)
{
  return cudaMalloc(ptr, size);
}

cudaError_t CudaCopyToDevice(void *tgt, void const *src, unsigned size)
{
  return cudaMemcpy(tgt, src, size, cudaMemcpyHostToDevice);
}

cudaError_t CudaCopyFromDevice(void *tgt, void const *src, unsigned size)
{
  return cudaMemcpy(tgt, src, size, cudaMemcpyDeviceToHost);
}

cudaError_t CudaCopyFromDeviceAsync(void *dst, void const * src, unsigned size, cudaStream_t stream)
{
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

cudaError_t CudaFree(void *ptr)
{
  return cudaFree(ptr);
}

cudaError_t CudaDeviceSetStackLimit(unsigned size)
{
  return cudaDeviceSetLimit(cudaLimitStackSize, size);
}

cudaError_t CudaDeviceSetHeapLimit(unsigned size)
{
  return cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
}

} // End namespace cuda

} // End namespace vecgeom
