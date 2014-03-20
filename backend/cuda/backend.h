/**
 * @file cuda/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_CUDABACKEND_H_
#define VECGEOM_BACKEND_CUDABACKEND_H_

#include <cassert>
#include "base/global.h"
#include "backend/cuda/interface.h"
#include "backend/scalar/backend.h"

namespace vecgeom_cuda {

struct kCuda {
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  const static bool early_returns = false;
  const static precision_v kOne = 1.0;
  const static precision_v kZero = 0.0;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
};

typedef kCuda::int_v       CudaInt;
typedef kCuda::precision_v CudaPrecision;
typedef kCuda::bool_v      CudaBool;

static const unsigned kThreadsPerBlock = 256;

// Auxiliary GPU functions

VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
int ThreadIndex() {
  return blockDim.x * gridDim.x * blockIdx.y
         + blockDim.x * blockIdx.x
         + threadIdx.x;
}

/**
 * Initialize with the number of threads required to construct the necessary
 * block and grid dimensions to accommodate all threads.
 */
struct LaunchParameters {
  dim3 block_size;
  dim3 grid_size;
  LaunchParameters(const unsigned threads) {
    // Blocks always one dimensional
    block_size.x = kThreadsPerBlock;
    block_size.y = 1;
    block_size.z = 1;
    // Grid becomes two dimensions at large sizes
    const unsigned blocks = 1 + (threads - 1) / kThreadsPerBlock;
    grid_size.z = 1;
    if (blocks <= 1<<16) {
      grid_size.x = blocks;
      grid_size.y = 1;
    } else {
      int dim = static_cast<int>(sqrt(static_cast<double>(blocks)) + 0.5);
      grid_size.x = dim;
      grid_size.y = dim;
    }
  }
};

} // End global namespace

namespace vecgeom {

template <typename Type>
Type* AllocateOnGpu(const unsigned size) {
  Type *ptr;
  CudaAssertError(cudaMalloc((void**)&ptr, size));
  return ptr;
}

template <typename Type>
Type* AllocateOnGpu() {
  return AllocateOnGpu<Type>(sizeof(Type));
}

template <typename Type>
void FreeFromGpu(Type *const ptr) {
  CudaAssertError(cudaFree(ptr));
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    cudaMemcpy(tgt, src, size, cudaMemcpyHostToDevice)
  );
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt) {
  CopyToGpu<Type>(src, tgt, sizeof(Type));
}

template <typename Type>
void CopyFromGpu(Type const * const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    cudaMemcpy(tgt, src, size, cudaMemcpyDeviceToHost)
  );
}

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt) {
  CopyFromGpu<Type>(src, tgt, sizeof(Type));
}

void CudaCheckMemory(size_t *const free_memory, size_t *const total_memory) {
  CudaAssertError(cudaMemGetInfo(free_memory, total_memory));
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_CUDABACKEND_H_