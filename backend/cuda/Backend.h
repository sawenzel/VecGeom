/// \file cuda/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_CUDABACKEND_H_
#define VECGEOM_BACKEND_CUDABACKEND_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/cuda/Interface.h"

#include <cassert>

namespace vecgeom_cuda {

struct kCuda {
  typedef int       int_v;
  typedef Precision precision_v;
  typedef bool      bool_v;
  typedef Inside_t  inside_v;
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
  return blockDim.x * blockIdx.x
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
    // Grid becomes two dimensional at large sizes
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

#endif // VECGEOM_BACKEND_CUDABACKEND_H_