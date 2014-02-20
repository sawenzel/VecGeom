#ifndef VECGEOM_BACKEND_CUDABACKEND_H_
#define VECGEOM_BACKEND_CUDABACKEND_H_

#include <iostream>
#include <cassert>
#include "base/utilities.h"
#include "base/types.h"
#include "backend/backend.h"

namespace vecgeom {

template <>
struct Impl<kCuda> {
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  const static bool early_returns = false;
  const static precision_v kOne = 1.0;
  const static precision_v kZero = 0.0;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
};

typedef Impl<kCuda>::int_v       CudaInt;
typedef Impl<kCuda>::precision_v CudaPrecision;
typedef Impl<kCuda>::bool_v      CudaBool;

static const int kThreadsPerBlock = 256;

// Auxiliary GPU functions

VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
int ThreadIndex() {
  return blockDim.x * gridDim.x * blockIdx.y
         + blockDim.x * blockIdx.x
         + threadIdx.x;
}

cudaError_t CudaCheckError(const cudaError_t err);

cudaError_t CudaCheckError();

void CudaAssertError(const cudaError_t err);

void CudaAssertError();
/**
 * Initialize with the number of threads required to construct the necessary
 * block and grid dimensions to accommodate all threads.
 */
struct LaunchParameters {
  dim3 block_size;
  dim3 grid_size;
  LaunchParameters(const int threads) {
    // Blocks always one dimensional
    block_size.x = kThreadsPerBlock;
    block_size.y = 1;
    block_size.z = 1;
    // Grid becomes two dimensions at large sizes
    const int blocks = 1 + (threads - 1) / kThreadsPerBlock;
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

VECGEOM_CUDA_HEADER_HOST
void* AllocateOnGpu(const int size);

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
Type* AllocateOnGpu() {
  Type *ptr;
  CudaAssertError(cudaMalloc((void**)&ptr, sizeof(Type)));
  return ptr;
}

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
void FreeFromGpu(Type *const ptr) {
  CudaAssertError(cudaFree(ptr));
}

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
void CopyToGPU(Type const *const src, Type *const tgt, const int count) {
  CudaAssertError(
    cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyHostToDevice)
  );
}

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
void CopyFromGPU(Type const * const src, Type *const tgt, const int count) {
  CudaAssertError(
    cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyDeviceToHost)
  );
}

// Microkernels

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CondAssign(const bool cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  *output = (cond) ? thenval : elseval;
}

template <typename Type1, typename Type2>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void MaskedAssign(const bool cond,
                  Type1 const &thenval, Type2 *const output) {
  *output = (cond) ? thenval : *output;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Abs(Type const &val) {
  return fabs(val);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Sqrt(Type const &val) {
  return sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_CUDABACKEND_H_