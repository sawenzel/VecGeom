#ifndef VECGEOM_BACKEND_CUDABACKEND_H_
#define VECGEOM_BACKEND_CUDABACKEND_H_

#include "base/utilities.h"
#include "base/types.h"
#include "backend/backend.h"

namespace vecgeom {

template <>
struct Impl<kCuda> {
  typedef double precision;
  typedef int    int_v;
  typedef double double_v;
  typedef bool bool_v;
  const static bool early_returns = false;
  const static double_v kOne = 1.0;
  const static double_v kZero = 0.0;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
};

typedef Impl<kCuda>::int_v    CudaInt;
typedef Impl<kCuda>::double_v CudaDouble;
typedef Impl<kCuda>::bool_v   CudaBool;

static const int kThreadsPerBlock = 256;

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

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
VECGEOM_INLINE
static Type* AllocateOnGPU(const int count) {
  Type *ptr;
  cudaMalloc((void**)&ptr, count*sizeof(Type));
  return ptr;
}

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
VECGEOM_INLINE
void CopyToGPU(Type const * const src, Type * const tgt, const int count) {
  cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyHostToDevice);
}

template <typename Type>
VECGEOM_CUDA_HEADER_HOST
VECGEOM_INLINE
void CopyFromGPU(Type const * const src, Type * const tgt, const int count) {
  cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyDeviceToHost);
}

// Microkernels

template <ImplType it, typename Type>
VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
void CondAssign(typename Impl<it>::bool_v const &cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  *output = (cond) ? thenval : elseval;
}

template <ImplType it, typename Type1, typename Type2>
VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
void MaskedAssign(typename Impl<it>::bool_v const &cond,
                  Type1 const &thenval, Type2 *const output) {
  *output = (cond) ? thenval : *output;
}

template <>
VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
CudaDouble Abs<kCuda, CudaDouble>(CudaDouble const &val) {
  return fabs(val);
}

template <>
VECGEOM_CUDA_HEADER_DEVICE
VECGEOM_INLINE
CudaDouble Sqrt<kCuda, CudaDouble>(CudaDouble const &val) {
  return sqrt(val);
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_CUDABACKEND_H_