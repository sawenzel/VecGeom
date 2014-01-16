#ifndef LIBRARYCUDA_H
#define LIBRARYCUDA_H

#include "mm_malloc.h"
#include "LibraryGeneric.h"

static const int threads_per_block = 512;

template <>
struct ImplTraits<kCuda> {
  typedef CudaFloat float_t;
  typedef int       int_v;
  typedef CudaFloat float_v;
  typedef bool      bool_v;
  const static bool early_return = true;
  const static bool_v kZero = 0;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
};

typedef ImplTraits<kCuda>::int_v  CudaInt;
typedef ImplTraits<kCuda>::bool_v CudaBool;

struct LaunchParameters {
  dim3 block_size;
  dim3 grid_size;
  LaunchParameters(const int threads) {
    // Blocks always one dimensional
    block_size.x = threads_per_block;
    block_size.y = 1;
    block_size.z = 1;
    // Grid becomes two dimensions at large sizes
    const int blocks = 1 + (threads - 1) / threads_per_block;
    grid_size.z = 1;
    if (blocks <= 1<<16) {
      grid_size.x = blocks;
      grid_size.y = 1;
    } else {
      int dim = int(sqrt(double(blocks)) + 0.5);
      grid_size.x = dim;
      grid_size.y = dim;
    }
  }
};

__host__
inline __attribute__((always_inline))
void CheckCudaError() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA reported error with message: \""
              << cudaGetErrorString(err) << "\"\n";
    assert(err != cudaSuccess);
  }
}
__host__ __device__
inline __attribute__((always_inline))
Vector3D<CudaFloat> DeviceVector(
    Vector3D<double> const &vec) {
  return Vector3D<CudaFloat>(CudaFloat(vec[0]),
                             CudaFloat(vec[1]),
                             CudaFloat(vec[2]));
}

__device__
inline __attribute__((always_inline))
int ThreadIndex() {
  return blockDim.x * gridDim.x * blockIdx.y
         + blockDim.x * blockIdx.x
         + threadIdx.x;
}

template <typename Type>
__host__
inline __attribute__((always_inline))
static Type* AllocateOnGPU(const int count) {
  Type *ptr;
  cudaMalloc((void**)&ptr, count*sizeof(Type));
  return ptr;
}

template <typename Type>
__host__
inline __attribute__((always_inline))
void CopyToGPU(Type const * const src, Type * const tgt, const int count) {
  cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyHostToDevice);
}

template <typename Type>
__host__
inline __attribute__((always_inline))
void CopyFromGPU(Type const * const src, Type * const tgt, const int count) {
  cudaMemcpy(tgt, src, count*sizeof(Type), cudaMemcpyDeviceToHost);
}

template <>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
CudaFloat Abs<kCuda, CudaFloat>(CudaFloat const &val) {
  return fabs(val);
}

// Use scalar implementation

// template <typename Type>
// __host__ __device__
// inline __attribute__((always_inline))
// Type CondAssign(const CudaBool &cond,
//                 const Type &thenval, const Type &elseval) {
//   return (cond) ? thenval : elseval;
// }

// template <typename Type>
// __host__ __device__
// inline __attribute__((always_inline))
// void MaskedAssign(const CudaBool &cond,
//                   const Type &thenval, Type &output) {
//   output = (cond) ? thenval : output;
// }

#endif /* LIBRARYCUDA_H */